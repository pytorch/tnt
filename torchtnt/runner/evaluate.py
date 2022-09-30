# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable, List, Optional

import torch
from torchtnt.runner.callback import Callback

from torchtnt.runner.progress import Progress
from torchtnt.runner.state import EntryPoint, PhaseState, State
from torchtnt.runner.unit import EvalUnit, TEvalData
from torchtnt.runner.utils import (
    _check_loop_condition,
    _is_epoch_done,
    _reset_module_training_mode,
    _run_callback_fn,
    _set_module_training_mode,
    _step_requires_iterator,
    log_api_usage,
)
from torchtnt.utils.timer import get_timer_summary, Timer

logger: logging.Logger = logging.getLogger(__name__)


def evaluate(
    eval_unit: EvalUnit[TEvalData],
    dataloader: Iterable[TEvalData],
    callbacks: Optional[List[Callback]] = None,
    *,
    max_steps_per_epoch: Optional[int] = None,
) -> State:
    """Makes a single pass through the evaluation dataloader."""
    log_api_usage("evaluate")
    callbacks = callbacks or []
    state = State(
        entry_point=EntryPoint.EVALUATE,
        timer=Timer(),
        eval_state=PhaseState(
            dataloader=dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
            progress=Progress(),
        ),
    )
    try:
        _evaluate_impl(state, eval_unit, callbacks)
        logger.info("Finished evaluation")
        logger.debug(get_timer_summary(state.timer))
        return state
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(e)
        eval_unit.on_exception(state, e)
        _run_callback_fn(callbacks, "on_exception", state, eval_unit, e)
        raise e


@torch.inference_mode()
def _evaluate_impl(
    state: State,
    eval_unit: EvalUnit[TEvalData],
    callbacks: List[Callback],
) -> None:
    # input validation
    eval_state = state.eval_state
    if not eval_state:
        raise RuntimeError("Expected eval_state to be initialized!")
    max_steps_per_epoch = eval_state.max_steps_per_epoch
    _check_loop_condition("max_steps_per_epoch", max_steps_per_epoch)
    logger.info(f"Started evaluate with max_steps_per_epoch={max_steps_per_epoch}")

    # Set all modules to eval mode
    # access modules made available through _AppStateMixin
    tracked_modules = eval_unit.tracked_modules()
    prior_module_train_states = _set_module_training_mode(tracked_modules, False)

    with state.timer.time(f"eval.{eval_unit.__class__.__name__}.on_eval_start"):
        eval_unit.on_eval_start(state)
    _run_callback_fn(callbacks, "on_eval_start", state, eval_unit)

    # Conditionally run this to avoid running this multiple times
    # in the case of resuming from a checkpoint mid-epoch
    if eval_state.progress.num_steps_completed_in_epoch == 0:
        with state.timer.time(
            f"eval.{eval_unit.__class__.__name__}.on_eval_epoch_start"
        ):
            eval_unit.on_eval_epoch_start(state)
        _run_callback_fn(callbacks, "on_eval_epoch_start", state, eval_unit)

    data_iter = iter(eval_state.dataloader)
    step_input = data_iter

    # pyre-ignore[6]: Incompatible parameter type
    pass_data_iter_to_step = _step_requires_iterator(eval_unit.eval_step)
    prev_steps_in_epoch = eval_state.progress.num_steps_completed_in_epoch

    while not (
        state.should_stop
        or _is_epoch_done(
            eval_state.progress, eval_state.max_steps_per_epoch, eval_state.max_steps
        )
    ):
        try:
            if not pass_data_iter_to_step:
                # get the next batch from the data iterator
                with state.timer.time("eval.data_iter_next"):
                    step_input = next(data_iter)
            _run_callback_fn(callbacks, "on_eval_step_start", state, eval_unit)
            with state.timer.time(f"eval.{eval_unit.__class__.__name__}.eval_step"):
                eval_state.step_output = eval_unit.eval_step(state, step_input)
            _run_callback_fn(callbacks, "on_eval_step_end", state, eval_unit)
            # clear step_output to avoid retaining extra memory
            eval_state.step_output = None
            eval_state.progress.num_steps_completed_in_epoch += 1
            eval_state.progress.num_steps_completed += 1
        except StopIteration:
            break

    # Possibly warn about an empty dataloader
    any_steps_completed = (
        abs(eval_state.progress.num_steps_completed_in_epoch - prev_steps_in_epoch) == 0
    )
    if not any_steps_completed:
        logger.warning("No steps completed during evaluate epoch!")

    with state.timer.time(f"eval.{eval_unit.__class__.__name__}.on_eval_epoch_end"):
        eval_unit.on_eval_epoch_end(state)
    _run_callback_fn(callbacks, "on_eval_epoch_end", state, eval_unit)

    # set progress counters for the next epoch
    eval_state.progress.num_epochs_completed += 1
    eval_state.progress.num_steps_completed_in_epoch = 0

    with state.timer.time(f"eval.{eval_unit.__class__.__name__}.on_eval_end"):
        eval_unit.on_eval_end(state)
    _run_callback_fn(callbacks, "on_eval_end", state, eval_unit)

    # Reset training mode for modules at the end of the epoch
    # This ensures that side-effects made by the loop are reset before
    # returning back to the user
    _reset_module_training_mode(tracked_modules, prior_module_train_states)
