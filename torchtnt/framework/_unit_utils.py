# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import collections
import inspect
import logging
from typing import Callable, Dict, List, Tuple, TypeVar

import torch
import typing_extensions
from torchtnt.framework.state import State

_logger: logging.Logger = logging.getLogger(__name__)
T = TypeVar("T")


def _step_requires_iterator(step_func: Callable[[State, T], object]) -> bool:
    """
    Helper function to evaluate whether the get_next_X_batch method should pass the data iterator to the `X_step`
    functions, or whether get_next_X_batch should call `next(data_iter)` and pass a single batch to the step method.

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
