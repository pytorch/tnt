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
from torchtnt.runner.train import _train_epoch_impl
from torchtnt.runner.unit import EvalUnit, TEvalData, TrainUnit, TTrainData
from torchtnt.runner.utils import _check_loop_condition, _is_done

logger: logging.Logger = logging.getLogger(__name__)


def fit(
    train_unit: TrainUnit,
    eval_unit: EvalUnit,
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
    torch._C._log_api_usage_once("torchtnt.runner.fit")
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
        _check_loop_condition("max_epochs", max_epochs)
        _check_loop_condition("max_train_steps_per_epoch", max_train_steps_per_epoch)
        _check_loop_condition("max_eval_steps_per_epoch", max_eval_steps_per_epoch)
        _check_loop_condition("evaluate_every_n_steps", evaluate_every_n_steps)
        _check_loop_condition("evaluate_every_n_epochs", evaluate_every_n_epochs)
        logger.info(
            f"Started fit with max_epochs={max_epochs}"
            f"max_train_steps_per_epoch={max_train_steps_per_epoch}"
            f"max_eval_steps_per_epoch={max_eval_steps_per_epoch}"
            f"evaluate_every_n_steps={evaluate_every_n_steps}"
            f"evaluate_every_n_epochs={evaluate_every_n_epochs}"
        )
        _fit_impl(state, train_unit, eval_unit)
        return state
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(e)
        train_unit.on_exception(state, e)
        eval_unit.on_exception(state, e)
        raise e


def _fit_impl(
    state: State,
    train_unit: TrainUnit,
    eval_unit: EvalUnit,
) -> None:
    train_unit.on_train_start(state)

    train_state = state.train_state
    assert train_state is not None

    while not _is_done(train_state.progress, train_state.max_epochs):
        _train_epoch_impl(state, train_unit, eval_unit)

    # delete step_output to avoid retaining extra memory
    train_state.step_output = None

    train_unit.on_train_end(state)
