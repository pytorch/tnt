# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable, List, Optional

from pyre_extensions import none_throws
from torchtnt.framework._callback_handler import CallbackHandler
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
from torchtnt.framework.utils import _get_timing_context, _is_done, log_api_usage
from torchtnt.utils.timer import get_timer_summary, Timer

logger: logging.Logger = logging.getLogger(__name__)


def fit(
    unit: TTrainUnit,
    train_dataloader: Iterable[TTrainData],
    eval_dataloader: Iterable[TEvalData],
    *,
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    max_train_steps_per_epoch: Optional[int] = None,
    max_eval_steps_per_epoch: Optional[int] = None,
    evaluate_every_n_steps: Optional[int] = None,
    evaluate_every_n_epochs: Optional[int] = 1,
    callbacks: Optional[List[Callback]] = None,
    auto_timing: bool = False,
) -> None:
    """
    The ``fit`` entry point interleaves training and evaluation loops. The ``fit`` entry point takes in an object which subclasses both :class:`~torchtnt.framework.TrainUnit` and :class:`~torchtnt.framework.EvalUnit`, train and eval dataloaders (any Iterables), optional arguments to modify loop execution,
    and runs the fit loop.

    Args:
        unit: an instance that subclasses both :class:`~torchtnt.framework.unit.TrainUnit` and :class:`~torchtnt.framework.unit.EvalUnit`,
         implementing :meth:`~torchtnt.framework.TrainUnit.train_step` and :meth:`~torchtnt.framework.EvalUnit.eval_step`.
        train_dataloader: dataloader to be used during training, which can be *any* iterable, including PyTorch DataLoader, DataLoader2, etc.
        eval_dataloader: dataloader to be used during evaluation, which can be *any* iterable, including PyTorch DataLoader, DataLoader2, etc.
        max_epochs: the max number of epochs to run for training. ``None`` means no limit (infinite training) unless stopped by max_steps.
        max_steps: the max number of steps to run for training. ``None`` means no limit (infinite training) unless stopped by max_epochs.
        max_train_steps_per_epoch: the max number of steps to run per epoch for training. None means train until ``train_dataloader`` is exhausted.
        max_eval_steps_per_epoch: the max number of steps to run per epoch for evaluation. None means evaluate until ``eval_dataloader`` is exhausted.
        evaluate_every_n_steps: how often to run the evaluation loop in terms of training steps.
        evaluate_every_n_epochs: how often to run the evaluation loop in terms of training epochs.
        callbacks: an optional list of callbacks.
        auto_timing: whether to automatically time the training and evaluation loop, using the state's timer (enabling auto_timing may degrade performance).

    Below is an example of calling :py:func:`~torchtnt.framework.fit`.

    .. code-block:: python

        from torchtnt.framework import fit

        fit_unit = MyFitUnit(module=..., optimizer=..., lr_scheduler=...)
        train_dataloader = torch.utils.data.DataLoader(...)
        eval_dataloader = torch.utils.data.DataLoader(...)
        fit(fit_unit, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, max_epochs=4)

    Below is pseudocode of what the :py:func:`~torchtnt.framework.fit` entry point does.

    .. code-block:: text

        set unit's tracked modules to train mode
        call on_train_start on unit first and then callbacks
        while training is not done:
            while epoch is not done:
                call on_train_epoch_start on unit first and then callbacks
                try:
                    data = next(dataloader)
                    call on_train_step_start on callbacks
                    call train_step on unit
                    increment step counter
                    call on_train_step_end on callbacks
                    if should evaluate after this step:
                        run eval loops
                except StopIteration:
                    break
            increment epoch counter
            call on_train_epoch_end on unit first and then callbacks
            if should evaluate after this epoch:
                run eval loop
        call on_train_end on unit first and then callbacks
    """
    log_api_usage("fit")
    callback_handler = CallbackHandler(callbacks or [])
    state = State(
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
        timer=None if not auto_timing else Timer(),
    )
    try:
        _fit_impl(state, unit, callback_handler)
        if state.timer:
            logger.info(get_timer_summary(state.timer))
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(e)
        unit.on_exception(state, e)
        callback_handler.on_exception(state, unit, e)
        raise e


def _fit_impl(
    state: State,
    unit: TTrainUnit,
    callback_handler: CallbackHandler,
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

    with _get_timing_context(state, f"{unit.__class__.__name__}.on_train_start"):
        unit.on_train_start(state)
    callback_handler.on_train_start(state, unit)

    while not (
        state.should_stop
        or _is_done(unit.train_progress, train_state.max_epochs, train_state.max_steps)
    ):
        _train_epoch_impl(state, unit, callback_handler)

    with _get_timing_context(state, f"{unit.__class__.__name__}.on_train_end"):
        unit.on_train_end(state)
    callback_handler.on_train_end(state, unit)
