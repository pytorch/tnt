# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable, List, Optional

import torch
from pyre_extensions import none_throws

from torchtnt.framework._callback_handler import CallbackHandler
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import ActivePhase, EntryPoint, PhaseState, State
from torchtnt.framework.unit import TPredictData, TPredictUnit
from torchtnt.framework.utils import (
    _is_epoch_done,
    _reset_module_training_mode,
    _set_module_training_mode,
    get_timing_context,
    log_api_usage,
)
from torchtnt.utils.timer import get_timer_summary, TimerProtocol

logger: logging.Logger = logging.getLogger(__name__)


def predict(
    predict_unit: TPredictUnit,
    predict_dataloader: Iterable[TPredictData],
    *,
    max_steps_per_epoch: Optional[int] = None,
    callbacks: Optional[List[Callback]] = None,
    timer: Optional[TimerProtocol] = None,
) -> None:
    """
    The ``predict`` entry point takes in a :class:`~torchtnt.framework.unit.PredictUnit` object, a train dataloader (any Iterable), optional arguments to modify loop execution,
    and runs the prediction loop.

    Args:
        predict_unit: an instance of :class:`~torchtnt.framework.unit.PredictUnit` which implements `predict_step`.
        predict_dataloader: dataloader to be used during prediction, which can be *any* iterable, including PyTorch DataLoader, DataLoader2, etc.
        max_steps_per_epoch: the max number of steps to run per epoch. None means predict until the dataloader is exhausted.
        callbacks: an optional list of :class:`~torchtnt.framework.callback.Callback` s.
        timer: an optional Timer which will be used to time key events (using a Timer with CUDA synchronization may degrade performance).


    Below is an example of calling :py:func:`~torchtnt.framework.predict`.

    .. code-block:: python

        from torchtnt.framework.predict import predict

        predict_unit = MyPredictUnit(module=..., optimizer=..., lr_scheduler=...)
        predict_dataloader = torch.utils.data.DataLoader(...)
        predict(predict_unit, predict_dataloader, max_steps_per_epoch=20)

    Below is pseudocode of what the :py:func:`~torchtnt.framework.predict` entry point does.

    .. code-block:: text

        set unit's tracked modules to eval mode
        call on_predict_start on unit first and then callbacks
        while not done:
            call on_predict_epoch_start on unit first and then callbacks
            try:
                call get_next_predict_batch on unit
                call on_predict_step_start on callbacks
                call predict_step on unit
                increment step counter
                call on_predict_step_end on callbacks
            except StopIteration:
                break
        increment epoch counter
        call on_predict_epoch_end on unit first and then callbacks
        call on_predict_end on unit first and then callbacks
    """
    log_api_usage("predict")
    callback_handler = CallbackHandler(callbacks or [])
    state = State(
        entry_point=EntryPoint.PREDICT,
        predict_state=PhaseState(
            dataloader=predict_dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
        ),
        timer=timer,
    )
    try:
        _predict_impl(state, predict_unit, callback_handler)
        logger.info("Finished predict")
        if state.timer:
            logger.info(get_timer_summary(state.timer))
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(f"Exception during predict:\n{e}")
        predict_unit.on_exception(state, e)
        callback_handler.on_exception(state, predict_unit, e)
        raise e


@torch.inference_mode()
def _predict_impl(
    state: State,
    predict_unit: TPredictUnit,
    callback_handler: CallbackHandler,
) -> None:
    # input validation
    predict_state = none_throws(state.predict_state)

    state._active_phase = ActivePhase.PREDICT
    logger.info(
        f"Started predict with max_steps_per_epoch={predict_state.max_steps_per_epoch}"
    )

    # Set all modules to eval mode
    # access modules made available through AppStateMixin
    tracked_modules = predict_unit.tracked_modules()
    prior_module_train_states = _set_module_training_mode(tracked_modules, False)

    with get_timing_context(
        state, f"{predict_unit.__class__.__name__}.on_predict_start"
    ):
        predict_unit.on_predict_start(state)
    callback_handler.on_predict_start(state, predict_unit)

    # Conditionally run this to avoid running this multiple times
    # in the case of resuming from a checkpoint mid-epoch
    if predict_unit.predict_progress.num_steps_completed_in_epoch == 0:
        with get_timing_context(
            state, f"{predict_unit.__class__.__name__}.on_predict_epoch_start"
        ):
            predict_unit.on_predict_epoch_start(state)
        callback_handler.on_predict_epoch_start(state, predict_unit)

    with get_timing_context(state, "predict.iter(dataloader)"):
        data_iter = iter(predict_state.dataloader)
    step_input = data_iter

    prev_steps_in_epoch = predict_unit.predict_progress.num_steps_completed_in_epoch

    while not (
        state.should_stop
        or _is_epoch_done(
            predict_unit.predict_progress,
            predict_state.max_steps_per_epoch,
            predict_state.max_steps,
        )
    ):
        try:
            with get_timing_context(
                state, "predict.next(data_iter)"
            ), predict_state.iteration_timer.time("data_wait_time"):
                step_input = predict_unit.get_next_predict_batch(state, data_iter)
                callback_handler.on_predict_get_next_batch_end(state, predict_unit)

            with predict_state.iteration_timer.time("predict_iteration_time"):
                callback_handler.on_predict_step_start(state, predict_unit)
                predict_state._step_output = predict_unit.predict_step(
                    state, step_input
                )

                predict_unit.predict_progress.increment_step()
                callback_handler.on_predict_step_end(state, predict_unit)

                # clear step_output to avoid retaining extra memory
                predict_state._step_output = None
        except StopIteration:
            break

    # Possibly warn about an empty dataloader
    any_steps_completed = (
        abs(
            predict_unit.predict_progress.num_steps_completed_in_epoch
            - prev_steps_in_epoch
        )
        > 0
    )
    if not any_steps_completed:
        logger.warning("No steps completed during predict epoch!")

    # set progress counters for the next epoch
    predict_unit.predict_progress.increment_epoch()

    predict_unit.on_predict_epoch_end(state)
    callback_handler.on_predict_epoch_end(state, predict_unit)

    predict_unit.on_predict_end(state)
    callback_handler.on_predict_end(state, predict_unit)

    # Reset training mode for modules at the end of the epoch
    # This ensures that side-effects made by the loop are reset before
    # returning back to the user
    _reset_module_training_mode(tracked_modules, prior_module_train_states)
