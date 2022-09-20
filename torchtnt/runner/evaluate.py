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
from torchtnt.runner.unit import EvalUnit, TEvalData
from torchtnt.runner.utils import (
    _check_loop_condition,
    _is_epoch_done,
    _reset_module_training_mode,
    _set_module_training_mode,
    log_api_usage,
)
from torchtnt.utils.timer import get_timer_summary, Timer

logger: logging.Logger = logging.getLogger(__name__)


def evaluate(
    eval_unit: EvalUnit[TEvalData],
    dataloader: Iterable[TEvalData],
    *,
    max_steps_per_epoch: Optional[int] = None,
) -> State:
    """Makes a single pass through the evaluation dataloader."""
    log_api_usage("evaluate")
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
        _evaluate_impl(state, eval_unit)
        logger.info("Finished evaluation")
        logger.debug(get_timer_summary(state.timer))
        return state
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(e)
        eval_unit.on_exception(state, e)
        raise e


@torch.inference_mode()
def _evaluate_impl(
    state: State,
    eval_unit: EvalUnit[TEvalData],
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

    # Conditionally run this to avoid running this multiple times
    # in the case of resuming from a checkpoint mid-epoch
    if eval_state.progress.num_steps_completed_in_epoch == 0:
        with state.timer.time(
            f"eval.{eval_unit.__class__.__name__}.on_eval_epoch_start"
        ):
            eval_unit.on_eval_epoch_start(state)

    data_iter = iter(eval_state.dataloader)

    while not (
        state.should_stop
        or _is_epoch_done(eval_state.progress, eval_state.max_steps_per_epoch)
    ):
        try:
            # TODO: conditionally expose data iterator for use cases that require access during the step
            with state.timer.time("eval.data_iter_next"):
                batch = next(data_iter)
            with state.timer.time(f"eval.{eval_unit.__class__.__name__}.eval_step"):
                eval_state.step_output = eval_unit.eval_step(state, batch)
            # clear step_output to avoid retaining extra memory
            eval_state.step_output = None
            eval_state.progress.num_steps_completed_in_epoch += 1
            eval_state.progress.num_steps_completed += 1
        except StopIteration:
            break
    with state.timer.time(f"eval.{eval_unit.__class__.__name__}.on_eval_epoch_end"):
        eval_unit.on_eval_epoch_end(state)

    # set progress counters for the next epoch
    eval_state.progress.num_epochs_completed += 1
    eval_state.progress.num_steps_completed_in_epoch = 0

    with state.timer.time(f"eval.{eval_unit.__class__.__name__}.on_eval_end"):
        eval_unit.on_eval_end(state)

    # Reset training mode for modules at the end of the epoch
    # This ensures that side-effects made by the loop are reset before
    # returning back to the user
    _reset_module_training_mode(tracked_modules, prior_module_train_states)
