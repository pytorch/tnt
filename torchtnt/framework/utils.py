# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import inspect
import logging
from contextlib import contextmanager, nullcontext
from typing import Callable, ContextManager, Dict, Generator, List, Tuple, TypeVar

import torch
import typing_extensions
from torch.profiler import record_function
from torchtnt.framework.state import State

_logger: logging.Logger = logging.getLogger(__name__)
T = TypeVar("T")


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
