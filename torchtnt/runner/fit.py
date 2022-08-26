# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable, Optional

from torchtnt.runner.progress import Progress
from torchtnt.runner.state import EntryPoint, PhaseState, State
from torchtnt.runner.train import _train_epoch_impl
from torchtnt.runner.unit import EvalUnit, TEvalData, TrainUnit, TTrainData
from torchtnt.runner.utils import _check_loop_condition, _is_done, log_api_usage

logger: logging.Logger = logging.getLogger(__name__)


def fit(
    unit: TrainUnit[TTrainData],
    train_dataloader: Iterable[TTrainData],
    eval_dataloader: Iterable[TEvalData],
    *,
    max_epochs: Optional[int],
    max_train_steps_per_epoch: Optional[int] = None,
    max_eval_steps_per_epoch: Optional[int] = None,
    evaluate_every_n_steps: Optional[int] = None,
    evaluate_every_n_epochs: Optional[int] = 1,
) -> State:
    """Function that interleaves training & evaluation."""
    log_api_usage("fit")
    state = State(
        entry_point=EntryPoint.FIT,
        train_state=PhaseState(
            progress=Progress(),
            dataloader=train_dataloader,
            max_epochs=max_epochs,
            max_steps_per_epoch=max_train_steps_per_epoch,
        ),
        eval_state=PhaseState(
            progress=Progress(),
            dataloader=eval_dataloader,
            max_steps_per_epoch=max_eval_steps_per_epoch,
            evaluate_every_n_steps=evaluate_every_n_steps,
            evaluate_every_n_epochs=evaluate_every_n_epochs,
        ),
    )
    try:
        _fit_impl(state, unit)
        return state
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(e)
        unit.on_exception(state, e)
        raise e


def _fit_impl(
    state: State,
    unit: TrainUnit[TTrainData],
) -> None:
    # input validation
    if not isinstance(unit, TrainUnit):
        raise TypeError("Expected module to implement TrainUnit interface.")
    if not isinstance(unit, EvalUnit):
        raise TypeError("Expected module to implement EvalUnit interface.")

    train_state = state.train_state
    if not train_state:
        raise RuntimeError("Expected train_state to be initialized")
    eval_state = state.eval_state
    if not eval_state:
        raise RuntimeError("Expected eval_state to be initialized")

    _check_loop_condition("max_epochs", train_state.max_epochs)
    _check_loop_condition("max_train_steps_per_epoch", train_state.max_steps_per_epoch)
    _check_loop_condition("max_eval_steps_per_epoch", eval_state.max_steps_per_epoch)
    _check_loop_condition("evaluate_every_n_steps", eval_state.evaluate_every_n_steps)
    _check_loop_condition("evaluate_every_n_epochs", eval_state.evaluate_every_n_epochs)
    logger.info(
        f"Started fit with max_epochs={train_state.max_epochs}"
        f"max_train_steps_per_epoch={train_state.max_steps_per_epoch}"
        f"max_eval_steps_per_epoch={eval_state.max_steps_per_epoch}"
        f"evaluate_every_n_steps={eval_state.evaluate_every_n_steps}"
        f"evaluate_every_n_epochs={eval_state.evaluate_every_n_epochs}"
    )

    unit.on_train_start(state)

    while not _is_done(train_state.progress, train_state.max_epochs):
        _train_epoch_impl(state, unit)

    unit.on_train_end(state)
