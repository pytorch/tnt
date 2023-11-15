# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import inspect
import logging
from contextlib import contextmanager, nullcontext
from typing import (
    Callable,
    ContextManager,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import torch
import torch.nn as nn
import typing_extensions
from torch.profiler import record_function
from torchtnt.framework.state import State
from torchtnt.utils.progress import Progress

_logger: logging.Logger = logging.getLogger(__name__)
T = TypeVar("T")


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
    dataloader: Iterable[object],
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
def get_timing_context(
    state: State, event_name: str
) -> Generator[Tuple[ContextManager, ContextManager], None, None]:
    """
    Returns a context manager that records an event to a :class:`~torchtnt.utils.timer.Timer` and to PyTorch Profiler.

    Args:
        state: an instance of :class:`~torchtnt.framework.state.State`
        event_name: string identifier to use for timing
    """
    timer_context = (
        state.timer.time(event_name) if state.timer is not None else nullcontext()
    )
    profiler_context = record_function(event_name)
    with timer_context, profiler_context:
        yield (timer_context, profiler_context)


def log_api_usage(entry_point: str) -> None:
    torch._C._log_api_usage_once(f"torchtnt.framework.{entry_point}")


def _step_requires_iterator(step_func: Callable[[State, T], object]) -> bool:
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
