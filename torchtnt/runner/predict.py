# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable, Optional

import torch

from torchtnt.runner.progress import Progress
from torchtnt.runner.state import EntryPoint, PhaseState, State
from torchtnt.runner.unit import PredictUnit, TPredictData
from torchtnt.runner.utils import (
    _check_loop_condition,
    _is_epoch_done,
    _reset_module_training_mode,
    _set_module_training_mode,
    log_api_usage,
)

logger: logging.Logger = logging.getLogger(__name__)


def predict(
    predict_unit: PredictUnit[TPredictData],
    dataloader: Iterable[TPredictData],
    *,
    max_steps_per_epoch: Optional[int] = None,
) -> State:
    """Makes a single pass through the predict dataloader."""
    log_api_usage("predict")
    state = State(
        entry_point=EntryPoint.PREDICT,
        predict_state=PhaseState(
            dataloader=dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
            progress=Progress(),
        ),
    )
    try:
        _predict_impl(state, predict_unit)
        logger.info("Finished predict")
        return state
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(e)
        predict_unit.on_exception(state, e)
        raise e


@torch.inference_mode()
def _predict_impl(
    state: State,
    predict_unit: PredictUnit[TPredictData],
) -> None:
    # input validation
    predict_state = state.predict_state
    if not predict_state:
        raise RuntimeError("Expected predict_state to be initialized!")
    max_steps_per_epoch = predict_state.max_steps_per_epoch
    _check_loop_condition("max_steps_per_epoch", max_steps_per_epoch)
    logger.info(f"Started predict with max_steps_per_epoch={max_steps_per_epoch}")

    # Set all modules to eval mode
    # access modules made available through _AppStateMixin
    tracked_modules = predict_unit.tracked_modules()
    prior_module_train_states = _set_module_training_mode(tracked_modules, False)

    predict_unit.on_predict_start(state)

    # Conditionally run this to avoid running this multiple times
    # in the case of resuming from a checkpoint mid-epoch
    if predict_state.progress.num_steps_completed_in_epoch == 0:
        predict_unit.on_predict_epoch_start(state)

    data_iter = iter(predict_state.dataloader)

    while not _is_epoch_done(predict_state.progress, predict_state.max_steps_per_epoch):
        try:
            # TODO: conditionally expose data iterator for use cases that require access during the step
            batch = next(data_iter)
            predict_state.step_output = predict_unit.predict_step(state, batch)
            # clear step_output to avoid retaining extra memory
            predict_state.step_output = None
            predict_state.progress.num_steps_completed_in_epoch += 1
            predict_state.progress.num_steps_completed += 1
        except StopIteration:
            break
    predict_unit.on_predict_epoch_end(state)

    # set progress counters for the next epoch
    predict_state.progress.num_epochs_completed += 1
    predict_state.progress.num_steps_completed_in_epoch = 0

    predict_unit.on_predict_end(state)

    # Reset training mode for modules at the end of the epoch
    # This ensures that side-effects made by the loop are reset before
    # returning back to the user
    _reset_module_training_mode(tracked_modules, prior_module_train_states)
