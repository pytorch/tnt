# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import contextlib
import inspect
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import typing_extensions
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    FullyShardedDataParallel as FSDP,
)
from torch.profiler import record_function
from torchtnt.utils.version import is_torch_version_geq_2_0

if is_torch_version_geq_2_0():
    from torch.distributed._composable_state import _get_module_state
    from torch.distributed.fsdp._common_utils import _FSDPState


from pyre_extensions import none_throws
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import ActivePhase, EntryPoint, State
from torchtnt.framework.unit import AppStateMixin
from torchtnt.utils.lr_scheduler import TLRScheduler
from torchtnt.utils.progress import Progress
from typing_extensions import Self

_logger: logging.Logger = logging.getLogger(__name__)


# Helper functions common across the loops
def _is_done(
    progress: Progress, max_epochs: Optional[int], max_steps: Optional[int]
) -> bool:
    return (max_steps is not None and progress.num_steps_completed >= max_steps) or (
        max_epochs is not None and progress.num_epochs_completed >= max_epochs
    )


def _is_epoch_done(
    progress: Progress, max_steps_per_epoch: Optional[int], max_steps: Optional[int]
) -> bool:
    return (max_steps is not None and progress.num_steps_completed >= max_steps) or (
        max_steps_per_epoch is not None
        and progress.num_steps_completed_in_epoch >= max_steps_per_epoch
    )


def _maybe_set_distributed_sampler_epoch(
    # pyre-ignore: Missing parameter annotation [2]
    dataloader: Iterable[Any],
    current_epoch: int,
) -> None:
    """Set epoch of distributed sampler in dataloader, if applicable.
    See: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    """
    # Set current training epoch for any DistributedSampler in dataloader
    if isinstance(dataloader, torch.utils.data.DataLoader) and isinstance(
        dataloader.sampler,
        torch.utils.data.distributed.DistributedSampler,
    ):
        dataloader.sampler.set_epoch(current_epoch)


def _set_module_training_mode(
    modules: Dict[str, nn.Module], mode: bool
) -> Dict[str, bool]:
    """Returns states to allow for a reset at the end of the loop."""
    prior_module_train_states = {}
    for name, module in modules.items():
        prior_module_train_states[name] = module.training
        module.train(mode)
    return prior_module_train_states


def _reset_module_training_mode(
    modules: Dict[str, nn.Module], prior_modes: Dict[str, bool]
) -> None:
    # Reset training mode for modules at the end of the epoch
    # This ensures that side-effects made by the loop are reset before
    # returning back to the user
    for name, module in modules.items():
        if name in prior_modes:
            module.train(prior_modes[name])


@contextmanager
# pyre-fixme[3]: Return type must be annotated.
def _get_timing_context(state: State, event_name: str, skip_timer: bool = False):
    """Returns a context manager that records an event to a :class:`~torchtnt.utils.timer.Timer` and to PyTorch Profiler."""
    timer_context = (
        state.timer.time(event_name)
        if state.timer and not skip_timer
        else contextlib.nullcontext()
    )
    profiler_context = record_function(event_name)
    with timer_context, profiler_context:
        yield (timer_context, profiler_context)


def _run_callback_fn(
    callbacks: List[Callback],
    fn_name: str,
    state: State,
    *args: Any,
    **kwargs: Any,
) -> None:
    for cb in callbacks:
        fn = getattr(cb, fn_name)
        if not callable(fn):
            raise ValueError(f"Invalid callback method name provided: {fn_name}")
        with _get_timing_context(state, f"{cb.name}.{fn_name}"):
            fn(state, *args, **kwargs)


def log_api_usage(entry_point: str) -> None:
    torch._C._log_api_usage_once(f"torchtnt.framework.{entry_point}")


def _step_requires_iterator(step_func: Callable[[State, object], object]) -> bool:
    """
    Helper function to evaluate whether the loops should pass the data iterator to the `_step`
    functions, or whether the loop should call `next(data_iter)` and pass a single batch to process.

    This is closely tied to the Unit's corresponding step function signature.
    """
    argspec = inspect.getfullargspec(step_func)
    annotations = argspec.annotations
    if "data" not in annotations:
        _logger.warning(
            f"Expected step function to have an annotated argument named ``data``. Found {annotations}."
        )
        return False
    annotated_type = annotations["data"]
    return typing_extensions.get_origin(annotated_type) is collections.abc.Iterator


def _is_fsdp_module(module: torch.nn.Module) -> bool:
    if isinstance(module, FSDP):
        return True

    if is_torch_version_geq_2_0():
        # Also check for composable FSDP API
        maybe_composable_state = _get_module_state(module)
        if maybe_composable_state is not None:
            return isinstance(maybe_composable_state, _FSDPState)

    return False


class _FSDPOptimizerWrapper:
    """
    Wrapper for FSDP optimizer to call specific FSDP optimizer state checkpointing APIs.
    """

    def __init__(
        self, module: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        self.module = module
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        optim_state_dict = FullyShardedDataParallel.optim_state_dict(
            self.module, self.optimizer
        )
        return optim_state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        optim_state_dict = FullyShardedDataParallel.optim_state_dict_to_load(
            self.module, self.optimizer, state_dict
        )
        self.optimizer.load_state_dict(optim_state_dict)


def _construct_tracked_optimizers(
    unit: AppStateMixin,
) -> Dict[str, Union[torch.optim.Optimizer, _FSDPOptimizerWrapper]]:
    """
    Constructs tracked optimizers. Handles optimizers working on FSDP modules, wrapping them in _FSDPOptimizerWrapper.
    """
    fsdp_tracked_optimizers: Dict[str, _FSDPOptimizerWrapper] = {}
    for module in unit.tracked_modules().values():
        if _is_fsdp_module(module):
            # find optimizers for module, if exists
            optimizer_list = _find_optimizers_for_module(
                module, unit.tracked_optimizers()
            )
            for optim_name, optimizer in optimizer_list:
                fsdp_tracked_optimizers[optim_name] = _FSDPOptimizerWrapper(
                    module, optimizer
                )

    # construct custom tracked optimizers with FSDP optimizers
    tracked_optimizers: Dict[
        str, Union[torch.optim.Optimizer, _FSDPOptimizerWrapper]
    ] = {
        key: value
        for key, value in unit.tracked_optimizers().items()
        if key not in fsdp_tracked_optimizers
    }
    tracked_optimizers.update(fsdp_tracked_optimizers)
    return tracked_optimizers


def _construct_tracked_optimizers_and_schedulers(
    unit: AppStateMixin,
) -> Dict[str, Union[torch.optim.Optimizer, _FSDPOptimizerWrapper, TLRScheduler]]:
    """
    Combines tracked optimizers and schedulers. Handles optimizers working on FSDP modules, wrapping them in _FSDPOptimizerWrapper.
    """
    # construct custom tracked optimizers with FSDP optimizers
    tracked_optimizers_and_schedulers = _construct_tracked_optimizers(unit)

    # add schedulers
    for lr_scheduler_attrib_name, lr_scheduler in unit.tracked_lr_schedulers().items():
        if lr_scheduler_attrib_name in tracked_optimizers_and_schedulers:
            _logger.warning(
                f'Key collision "{lr_scheduler_attrib_name}" detected between LR Scheduler and optimizer attribute names. Please ensure there are no identical attribute names, as they will override each other.'
            )
        # pyre-ignore: Incompatible parameter type [6]: In call `dict.__setitem__`, for 2nd positional argument, expected `Optimizer` but got `str`.
        tracked_optimizers_and_schedulers[lr_scheduler_attrib_name] = lr_scheduler

    # pyre-ignore: Incompatible return type [7]
    return tracked_optimizers_and_schedulers


def _find_optimizers_for_module(
    module: torch.nn.Module, optimizers: Dict[str, torch.optim.Optimizer]
) -> List[Tuple[str, torch.optim.Optimizer]]:
    """
    Given a module, returns a list of optimizers that are associated with it.
    """
    optimizer_list = []
    module_params = [param.data_ptr() for param in module.parameters()]
    for optim_name, optimizer in optimizers.items():
        optimizer_params = [
            param.data_ptr() for param in optimizer.param_groups[0]["params"]
        ]
        if all(module_param in optimizer_params for module_param in module_params):
            optimizer_list.append((optim_name, optimizer))
    return optimizer_list


def get_current_progress(state: State) -> Progress:
    """
    If state's entry point is fit, returns train progress. During fit, we want to return training progress even during eval, so that metrics can be compared easily across train and eval.
    Otherwise, checks the active phase, and returns the corresponding progress class.
    """
    if state.entry_point == EntryPoint.FIT or state.active_phase == ActivePhase.TRAIN:
        return none_throws(state.train_state).progress

    if state.active_phase == ActivePhase.EVALUATE:
        return none_throws(state.eval_state).progress
    else:
        return none_throws(state.predict_state).progress


class StatefulInt:
    """
    This wrapper is useful if there are additional values related to training
    progress that need to be saved during checkpointing.
    """

    def __init__(self, val: int) -> None:
        self.val = val

    def state_dict(self) -> Dict[str, Any]:
        return {"value": self.val}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.val = state_dict["value"]

    def __add__(self, other: int) -> Self:
        self.val += other
        return self

    def __sub__(self, other: int) -> Self:
        self.val -= other
        return self

    def __repr__(self) -> str:
        return f"StatefulInt({self.val})"
