# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable, List, Optional

from pyre_extensions import none_throws
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import EntryPoint, PhaseState, State
from torchtnt.framework.train import _train_epoch_impl
from torchtnt.framework.unit import (
    EvalUnit,
    TEvalData,
    TrainUnit,
    TTrainData,
    TTrainUnit,
)
from torchtnt.framework.utils import _is_done, _run_callback_fn, log_api_usage
from torchtnt.utils.timer import get_timer_summary

logger: logging.Logger = logging.getLogger(__name__)


def init_fit_state(
    train_dataloader: Iterable[TTrainData],
    eval_dataloader: Iterable[TEvalData],
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    max_train_steps_per_epoch: Optional[int] = None,
    max_eval_steps_per_epoch: Optional[int] = None,
    evaluate_every_n_steps: Optional[int] = None,
    evaluate_every_n_epochs: Optional[int] = 1,
) -> State:
    """
    Helper function that initializes a :class:`~torchtnt.framework.State` object for fitting.

    Args:
        train_dataloader: dataloader to be used during training.
        eval_dataloader: dataloader to be used during evaluation.
        max_epochs: the max number of epochs to run for training. ``None`` means no limit (infinite training) unless stopped by max_steps.
        max_steps: the max number of steps to run for training. ``None`` means no limit (infinite training) unless stopped by max_epochs.
        max_train_steps_per_epoch: the max number of steps to run per epoch for training. None means train until the dataloader is exhausted.
        evaluate_every_n_steps: how often to run the evaluation loop in terms of training steps.
        evaluate_every_n_epochs: how often to run the evaluation loop in terms of training epochs.

    Returns:
        An initialized state object containing metadata.
    """

    return State(
        entry_point=EntryPoint.FIT,
        train_state=PhaseState(
            dataloader=train_dataloader,
            max_epochs=max_epochs,
            max_steps=max_steps,
            max_steps_per_epoch=max_train_steps_per_epoch,
        ),
        eval_state=PhaseState(
            dataloader=eval_dataloader,
            max_steps_per_epoch=max_eval_steps_per_epoch,
            evaluate_every_n_steps=evaluate_every_n_steps,
            evaluate_every_n_epochs=evaluate_every_n_epochs,
        ),
    )


def fit(
    state: State, unit: TTrainUnit, *, callbacks: Optional[List[Callback]] = None
) -> None:
    """
    The ``fit`` entry point interleaves training and evaluation loops.

    Args:
        state: a :class:`~torchtnt.framework.State` object containing metadata about the fitting run.
         :func:`~torchtnt.framework.init_fit_state` can be used to initialize a state object.
        unit: an instance that subclasses both :class:`~torchtnt.framework.unit.TrainUnit` and :class:`~torchtnt.framework.unit.EvalUnit`,
         implementing :meth:`~torchtnt.framework.TrainUnit.train_step` and :meth:`~torchtnt.framework.EvalUnit.eval_step`.
        callbacks: an optional list of callbacks.
    """
    log_api_usage("fit")
    callbacks = callbacks or []

    try:
        state._entry_point = EntryPoint.FIT
        _fit_impl(state, unit, callbacks)
        logger.debug(get_timer_summary(state.timer))
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(e)
        unit.on_exception(state, e)
        _run_callback_fn(callbacks, "on_exception", state, unit, e)
        raise e


def _fit_impl(
    state: State,
    unit: TTrainUnit,
    callbacks: List[Callback],
) -> None:
    # input validation
    if not isinstance(unit, TrainUnit):
        raise TypeError("Expected unit to implement TrainUnit interface.")
    if not isinstance(unit, EvalUnit):
        raise TypeError("Expected unit to implement EvalUnit interface.")

    train_state = none_throws(state.train_state)
    eval_state = none_throws(state.eval_state)

    logger.info(
        f"Started fit with max_epochs={train_state.max_epochs} "
        f"max_steps={train_state.max_steps} "
        f"max_train_steps_per_epoch={train_state.max_steps_per_epoch} "
        f"max_eval_steps_per_epoch={eval_state.max_steps_per_epoch} "
        f"evaluate_every_n_steps={eval_state.evaluate_every_n_steps} "
        f"evaluate_every_n_epochs={eval_state.evaluate_every_n_epochs} "
    )

    with state.timer.time(f"train.{unit.__class__.__name__}.on_train_start"):
        unit.on_train_start(state)
    _run_callback_fn(callbacks, "on_train_start", state, unit)

    while not (
        state.should_stop
        or _is_done(train_state.progress, train_state.max_epochs, train_state.max_steps)
    ):
        _train_epoch_impl(state, unit, callbacks)

    with state.timer.time(f"train.{unit.__class__.__name__}.on_train_end"):
        unit.on_train_end(state)
    _run_callback_fn(callbacks, "on_train_end", state, unit)
