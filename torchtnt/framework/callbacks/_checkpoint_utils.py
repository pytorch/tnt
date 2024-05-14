# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Any, cast, Dict, Union

from pyre_extensions import none_throws
from torchtnt.framework.callbacks.checkpointer_types import RestoreOptions
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import AppStateMixin, TEvalUnit, TTrainUnit
from torchtnt.utils.checkpoint import Phase

from torchtnt.utils.stateful import Stateful


# keys for use when checkpointing
_TRAIN_PROGRESS_STATE_KEY = "train_progress"
_TRAIN_DL_STATE_KEY = "train_dataloader"
_EVAL_PROGRESS_STATE_KEY = "eval_progress"


def _get_step_phase_mapping(
    state: State, unit: Union[TTrainUnit, TEvalUnit]
) -> Dict[Phase, int]:
    """
    Returns a mapping of phase to step, depending on the entrypoint.
    For FIT, it always includes train and eval progress.
    """
    step_mapping = {}

    if state.entry_point in (EntryPoint.TRAIN, EntryPoint.FIT):
        train_unit = cast(TTrainUnit, unit)
        step_mapping[Phase.TRAIN] = train_unit.train_progress.num_steps_completed

    if state.entry_point in (EntryPoint.EVALUATE, EntryPoint.FIT):
        eval_unit = cast(TEvalUnit, unit)
        step_mapping[Phase.EVALUATE] = eval_unit.eval_progress.num_steps_completed

    return step_mapping


def _prepare_app_state(unit: AppStateMixin) -> Dict[str, Any]:
    """Join together all of the tracked stateful entities to simplify registration of snapshottable states, deals with FSDP case"""
    app_state = unit.app_state()
    tracked_optimizers = unit._construct_tracked_optimizers()  # handles fsdp
    app_state.update(tracked_optimizers)
    return app_state


def _prepare_app_state_for_checkpoint(
    state: State, unit: AppStateMixin, intra_epoch: bool
) -> Dict[str, Stateful]:
    """
    Prepares the application state for checkpointing.
    """
    app_state = _prepare_app_state(unit)

    # for intra-epoch checkpointing, include dataloader states
    train_state = none_throws(state.train_state)
    train_dl = train_state.dataloader
    if intra_epoch and isinstance(train_dl, Stateful):
        app_state[_TRAIN_DL_STATE_KEY] = train_dl

    return app_state


def _prepare_app_state_for_restore(
    unit: AppStateMixin, restore_options: RestoreOptions
) -> Dict[str, Any]:
    """
    Prepares the application state for restoring from a checkpoint given a RestoreOptions.
    """
    app_state = _prepare_app_state(unit)

    restore_options = restore_options or RestoreOptions()
    if not restore_options.restore_train_progress:
        app_state.pop(_TRAIN_PROGRESS_STATE_KEY, None)

    if not restore_options.restore_eval_progress:
        app_state.pop(_EVAL_PROGRESS_STATE_KEY, None)

    if not restore_options.restore_optimizers:
        # remove all optimizer keys from app_state
        for optim_keys in unit.tracked_optimizers().keys():
            app_state.pop(optim_keys, None)

    if not restore_options.restore_lr_schedulers:
        # remove all lr scheduler keys from app_state
        for lr_scheduler_keys in unit.tracked_lr_schedulers().keys():
            app_state.pop(lr_scheduler_keys, None)

    return app_state
