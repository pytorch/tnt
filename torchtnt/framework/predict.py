# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable, List, Optional

import torch
from pyre_extensions import none_throws
from torchtnt.framework.callback import Callback

from torchtnt.framework.state import ActivePhase, EntryPoint, PhaseState, State
from torchtnt.framework.unit import TPredictData, TPredictUnit
from torchtnt.framework.utils import (
    _is_epoch_done,
    _reset_module_training_mode,
    _run_callback_fn,
    _set_module_training_mode,
    _step_requires_iterator,
    log_api_usage,
)
from torchtnt.utils.timer import get_timer_summary

logger: logging.Logger = logging.getLogger(__name__)


def init_predict_state(
    *,
    dataloader: Iterable[TPredictData],
    max_steps_per_epoch: Optional[int] = None,
) -> State:
    """
    Helper function that initializes a :class:`~torchtnt.framework.State` object for prediction.

    Args:
        dataloader: dataloader to be used during prediction.
        max_steps_per_epoch: the max number of steps to run per epoch. None means predict until the dataloader is exhausted.

    Returns:
        An initialized state object containing metadata.
    """

    return State(
        entry_point=EntryPoint.PREDICT,
        predict_state=PhaseState(
            dataloader=dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
        ),
    )


def predict(
    state: State,
    predict_unit: TPredictUnit,
    *,
    callbacks: Optional[List[Callback]] = None,
) -> None:
    """
    The ``predict`` entry point takes in a :class:`~torchtnt.framework.State` and :class:`~torchtnt.framework.PredictUnit` and runs the prediction loop over the data.

    Args:
        state: a State object containing metadata about the prediction run. This can be initialized using :func:`~torchtnt.framework.init_predict_state`.
        predict_unit: an instance of :class:`~torchtnt.framework.PredictUnit` which implements `predict_step`.
        callbacks: an optional list of callbacks.
    """
    log_api_usage("predict")
    callbacks = callbacks or []
    try:
        state._entry_point = EntryPoint.PREDICT
        _predict_impl(state, predict_unit, callbacks)
        logger.info("Finished predict")
        logger.debug(get_timer_summary(state.timer))
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(e)
        predict_unit.on_exception(state, e)
        _run_callback_fn(callbacks, "on_exception", state, predict_unit, e)
        raise e


@torch.inference_mode()
def _predict_impl(
    state: State,
    predict_unit: TPredictUnit,
    callbacks: List[Callback],
) -> None:
    # input validation
    predict_state = none_throws(state.predict_state)

    state._active_phase = ActivePhase.PREDICT
    logger.info(
        f"Started predict with max_steps_per_epoch={predict_state.max_steps_per_epoch}"
    )

    # Set all modules to eval mode
    # access modules made available through _AppStateMixin
    tracked_modules = predict_unit.tracked_modules()
    prior_module_train_states = _set_module_training_mode(tracked_modules, False)

    with state.timer.time(
        f"predict.{predict_unit.__class__.__name__}.on_predict_start"
    ):
        predict_unit.on_predict_start(state)
    _run_callback_fn(callbacks, "on_predict_start", state, predict_unit)

    # Conditionally run this to avoid running this multiple times
    # in the case of resuming from a checkpoint mid-epoch
    if predict_state.progress.num_steps_completed_in_epoch == 0:
        with state.timer.time(
            f"predict.{predict_unit.__class__.__name__}.on_predict_epoch_start"
        ):
            predict_unit.on_predict_epoch_start(state)
        _run_callback_fn(callbacks, "on_predict_epoch_start", state, predict_unit)

    data_iter = iter(predict_state.dataloader)
    step_input = data_iter

    pass_data_iter_to_step = _step_requires_iterator(predict_unit.predict_step)
    prev_steps_in_epoch = predict_state.progress.num_steps_completed_in_epoch

    while not (
        state.should_stop
        or _is_epoch_done(
            predict_state.progress,
            predict_state.max_steps_per_epoch,
            predict_state.max_steps,
        )
    ):
        try:
            if not pass_data_iter_to_step:
                # get the next batch from the data iterator
                with state.timer.time("predict.data_iter_next"):
                    step_input = next(data_iter)

            _run_callback_fn(callbacks, "on_predict_step_start", state, predict_unit)
            with state.timer.time(
                f"predict.{predict_unit.__class__.__name__}.predict_step"
            ):
                predict_state._step_output = predict_unit.predict_step(
                    state, step_input
                )
            predict_state.progress.increment_step()
            _run_callback_fn(callbacks, "on_predict_step_end", state, predict_unit)

            # clear step_output to avoid retaining extra memory
            predict_state._step_output = None
        except StopIteration:
            break

    # Possibly warn about an empty dataloader
    any_steps_completed = (
        abs(predict_state.progress.num_steps_completed_in_epoch - prev_steps_in_epoch)
        > 0
    )
    if not any_steps_completed:
        logger.warning("No steps completed during predict epoch!")

    # set progress counters for the next epoch
    predict_state.progress.increment_epoch()

    with state.timer.time(
        f"predict.{predict_unit.__class__.__name__}.on_predict_epoch_end"
    ):
        predict_unit.on_predict_epoch_end(state)
    _run_callback_fn(callbacks, "on_predict_epoch_end", state, predict_unit)

    with state.timer.time(f"predict.{predict_unit.__class__.__name__}.on_predict_end"):
        predict_unit.on_predict_end(state)
    _run_callback_fn(callbacks, "on_predict_end", state, predict_unit)

    # Reset training mode for modules at the end of the epoch
    # This ensures that side-effects made by the loop are reset before
    # returning back to the user
    _reset_module_training_mode(tracked_modules, prior_module_train_states)
