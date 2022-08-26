# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable, Optional

import torch
from torchtnt.runner.evaluate import _evaluate_impl

from torchtnt.runner.progress import Progress
from torchtnt.runner.state import EntryPoint, PhaseState, State
from torchtnt.runner.unit import TrainUnit, TTrainData
from torchtnt.runner.utils import (
    _check_loop_condition,
    _is_done,
    _is_epoch_done,
    _reset_module_training_mode,
    _set_module_training_mode,
    log_api_usage,
)

logger: logging.Logger = logging.getLogger(__name__)

# Define training procedures: train() and train_epoch() are exposed as top-level, user-facing functions


@torch.enable_grad()
def train(
    train_unit: TrainUnit[TTrainData],
    dataloader: Iterable[TTrainData],
    *,
    max_epochs: Optional[int],
    max_steps_per_epoch: Optional[int] = None,
) -> State:
    log_api_usage("train")
    state = State(
        entry_point=EntryPoint.TRAIN,
        train_state=PhaseState(
            dataloader=dataloader,
            max_epochs=max_epochs,
            max_steps_per_epoch=max_steps_per_epoch,
            progress=Progress(),
        ),
    )
    try:
        _train_impl(state, train_unit)
        logger.info("Finished train")
        return state
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(f"Exception during train\n: {e}")
        train_unit.on_exception(state, e)
        raise e


def _train_impl(
    state: State,
    train_unit: TrainUnit[TTrainData],
) -> None:
    train_state = state.train_state
    if not train_state:
        raise RuntimeError("Expected train_state to be initialized!")

    max_steps_per_epoch = train_state.max_steps_per_epoch
    _check_loop_condition("max_steps_per_epoch", train_state.max_steps_per_epoch)
    max_epochs = train_state.max_epochs
    _check_loop_condition("max_epochs", train_state.max_epochs)
    logger.info(
        f"Started train with max_epochs={max_epochs} and max_steps_per_epoch={max_steps_per_epoch}"
    )

    # TODO: add profiling & timing

    # Set all modules to train() mode
    # access modules made available through _AppStateMixin
    tracked_modules = train_unit.tracked_modules()
    prior_module_train_states = _set_module_training_mode(tracked_modules, True)

    train_unit.on_train_start(state)
    while not _is_done(train_state.progress, train_state.max_epochs):
        _train_epoch_impl(state, train_unit)

    train_unit.on_train_end(state)

    # Reset training mode for modules at the end of the epoch
    # This ensures that side-effects made by the loop are reset before
    # returning back to the user
    _reset_module_training_mode(tracked_modules, prior_module_train_states)


@torch.enable_grad()
def train_epoch(
    train_unit: TrainUnit[TTrainData],
    dataloader: Iterable[TTrainData],
    *,
    max_steps_per_epoch: Optional[int] = None,
) -> State:
    state = State(
        entry_point=EntryPoint.TRAIN,
        train_state=PhaseState(
            dataloader=dataloader,
            max_epochs=1,
            max_steps_per_epoch=max_steps_per_epoch,
            progress=Progress(),
        ),
    )

    try:
        _check_loop_condition("max_steps_per_epoch", max_steps_per_epoch)
        logger.info(
            f"Started train_epoch with max_steps_per_epoch={max_steps_per_epoch}"
        )
        _train_epoch_impl(
            state,
            train_unit,
        )
        logger.info("Finished train")
        return state
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(f"Exception during train_epoch\n: {e}")
        train_unit.on_exception(state, e)
        raise e


def _train_epoch_impl(state: State, train_unit: TrainUnit[TTrainData]) -> None:
    logger.info("Started train epoch")

    # Set all modules to train() mode
    # access modules made available through _AppStateMixin
    tracked_modules = train_unit.tracked_modules()
    prior_module_train_states = _set_module_training_mode(tracked_modules, True)

    train_state = state.train_state
    assert train_state is not None

    evaluate_every_n_steps = None
    evaluate_every_n_epochs = None
    if state.eval_state and state.eval_state.evaluate_every_n_steps:
        evaluate_every_n_steps = state.eval_state.evaluate_every_n_steps
    if state.eval_state and state.eval_state.evaluate_every_n_epochs:
        evaluate_every_n_epochs = state.eval_state.evaluate_every_n_epochs

    # Check the progress to conditionally run this
    # to avoid running this multiple times
    # in the case of resuming from a checkpoint mid-epoch
    if train_state.progress.num_steps_completed_in_epoch == 0:
        train_unit.on_train_epoch_start(state)

    data_iter = iter(train_state.dataloader)

    while not _is_epoch_done(train_state.progress, train_state.max_steps_per_epoch):
        try:
            # TODO: conditionally expose data iterator for use cases that require access during the step
            batch = next(data_iter)
            train_state.step_output = train_unit.train_step(state, batch)
            # clear step_output to avoid retaining extra memory
            train_state.step_output = None
            train_state.progress.num_steps_completed_in_epoch += 1
            train_state.progress.num_steps_completed += 1

            if (
                evaluate_every_n_steps
                and train_state.progress.num_steps_completed_in_epoch
                % evaluate_every_n_steps
                == 0
            ):
                _evaluate_impl(
                    state,
                    # pyre-ignore: Incompatible parameter type [6]
                    train_unit,
                )

        except StopIteration:
            break
    train_unit.on_train_epoch_end(state)

    # set progress counters for the next epoch
    train_state.progress.num_epochs_completed += 1
    train_state.progress.num_steps_completed_in_epoch = 0

    if (
        evaluate_every_n_epochs
        and train_state.progress.num_epochs_completed % evaluate_every_n_epochs == 0
    ):
        _evaluate_impl(
            state,
            # pyre-ignore: Incompatible parameter type [6]
            train_unit,
        )

    # Reset training mode for modules at the end of the epoch
    # This ensures that side-effects made by the loop are reset before
    # returning back to the user
    _reset_module_training_mode(tracked_modules, prior_module_train_states)

    logger.info("Ended train epoch")
