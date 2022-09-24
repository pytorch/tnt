# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import inspect
import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import typing_extensions
from torchtnt.runner.progress import Progress
from torchtnt.runner.state import State

_logger: logging.Logger = logging.getLogger(__name__)


# Helper functions common across the loops


def _check_loop_condition(name: str, val: Optional[int]) -> None:
    if val is not None and val < 0:
        raise ValueError(
            f"Invalid value provided for {name}. Expected a non-negative integer or None, but received {val}."
        )


def _is_done(progress: Progress, max_epochs: Optional[int]) -> bool:
    if max_epochs is None:
        # infinite training
        return False
    return progress.num_epochs_completed >= max_epochs


def _is_epoch_done(progress: Progress, max_steps_per_epoch: Optional[int]) -> bool:
    # No limit specified, so continue until the data iterator is exhausted
    if max_steps_per_epoch is None:
        return False
    return progress.num_steps_completed_in_epoch >= max_steps_per_epoch


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


def log_api_usage(entry_point: str) -> None:
    torch._C._log_api_usage_once(f"torchtnt.runner.{entry_point}")


def _step_requires_iterator(step_func: Callable[[State, object], object]) -> bool:
    """
    Helper function to evaluate whether the loops should pass the data iterator to the `_step`
    functions, or whether the loop should call `next(data_iter)` and pass a single batch to process.

    This is closely tied ot the Unit's corresponding step function signature.
    """
    argspec = inspect.getfullargspec(step_func)
    annotations = argspec.annotations
    if "data" not in annotations:
        _logger.warn(
            f"Expected step function to have an annotated argument named ``data``. Found {annotations}."
        )
        return False
    annotated_type = annotations["data"]
    return typing_extensions.get_origin(annotated_type) is collections.abc.Iterator
