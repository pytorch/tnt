# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable, List, Optional

import torch
from pyre_extensions import none_throws
from torchtnt.framework import AutoUnit
from torchtnt.framework.callback import Callback
from torchtnt.framework.evaluate import _evaluate_impl
from torchtnt.framework.state import ActivePhase, EntryPoint, PhaseState, State
from torchtnt.framework.unit import TTrainData, TTrainUnit
from torchtnt.framework.utils import (
    _get_timing_context,
    _is_done,
    _is_epoch_done,
    _maybe_set_distributed_sampler_epoch,
    _reset_module_training_mode,
    _run_callback_fn,
    _set_module_training_mode,
    _step_requires_iterator,
    log_api_usage,
)
from torchtnt.utils.timer import get_timer_summary, Timer

logger: logging.Logger = logging.getLogger(__name__)


def init_train_state(
    *,
    dataloader: Iterable[TTrainData],
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    max_steps_per_epoch: Optional[int] = None,
    auto_timing: bool = False,
) -> State:
    """
    ``init_train_state`` is a helper function that initializes a :class:`~torchtnt.framework.State` object for training. This :class:`~torchtnt.framework.State` object
    can then be passed to the :func:`~torchtnt.framework.train` entry point.

    Args:
        dataloader: dataloader to be used during training, which can be *any* iterable, including PyTorch DataLoader, DataLoader2, etc.
        max_epochs: the max number of epochs to run. ``None`` means no limit (infinite training) unless stopped by max_steps.
        max_steps: the max number of steps to run. ``None`` means no limit (infinite training) unless stopped by max_epochs.
        max_steps_per_epoch: the max number of steps to run per epoch. None means train until the dataloader is exhausted.
        auto_timing: whether to automatically time the training loop, using the state's timer (enabling auto_timing may degrade performance).

    Returns:
        An initialized state object containing metadata.

    Below is an example of calling :py:func:`~torchtnt.framework.init_train_state` and :py:func:`~torchtnt.framework.train` together.

    .. code-block:: python

        from torchtnt.framework import init_train_state, train

        train_unit = MyTrainUnit(module=..., optimizer=..., lr_scheduler=...)
        dataloader = torch.utils.data.DataLoader(...)
        state = init_train_state(dataloader=dataloader, max_epochs=4)
        train(state, train_unit)

    """

    return State(
        entry_point=EntryPoint.TRAIN,
        train_state=PhaseState(
            dataloader=dataloader,
            max_epochs=max_epochs,
            max_steps=max_steps,
            max_steps_per_epoch=max_steps_per_epoch,
        ),
        timer=None if not auto_timing else Timer(),
    )


@torch.enable_grad()
def train(
    state: State,
    train_unit: TTrainUnit,
    *,
    callbacks: Optional[List[Callback]] = None,
) -> None:
    """
    The ``train`` entry point takes in a :class:`~torchtnt.framework.State` object, a :class:`~torchtnt.framework.TrainUnit` object, and an optional list of :class:`~torchtnt.framework.Callback` s,
    and runs the training loop. The :class:`~torchtnt.framework.State` object can be initialized with :func:`~torchtnt.framework.init_train_state`.

    Args:
        state: a :class:`~torchtnt.framework.State` object containing metadata about the training run.
        train_unit: an instance of :class:`~torchtnt.framework.TrainUnit` which implements `train_step`.
        callbacks: an optional list of callbacks.

    Below is an example of calling :py:func:`~torchtnt.framework.init_train_state` and :py:func:`~torchtnt.framework.train` together.

    .. code-block:: python

        from torchtnt.framework import init_train_state, train

        train_unit = MyTrainUnit(module=..., optimizer=..., lr_scheduler=...)
        dataloader = torch.utils.data.DataLoader(...)
        state = init_train_state(dataloader=dataloader, max_epochs=4)
        train(state, train_unit)

    Below is pseudocode of what the :py:func:`~torchtnt.framework.train` entry point does.

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
                except StopIteration:
                    break
            increment epoch counter
            call on_train_epoch_end on unit first and then callbacks
        call on_train_end on unit first and then callbacks
    """
    log_api_usage("train")
    callbacks = callbacks or []
    try:
        state._entry_point = EntryPoint.TRAIN
        _train_impl(state, train_unit, callbacks)
        logger.info("Finished train")
        if state.timer:
            logger.info(get_timer_summary(state.timer))
    except Exception as e:
        # TODO: log for diagnostics
        logger.info(f"Exception during train\n: {e}")
        train_unit.on_exception(state, e)
        _run_callback_fn(callbacks, "on_exception", state, train_unit, e)
        raise e


def _train_impl(
    state: State,
    train_unit: TTrainUnit,
    callbacks: List[Callback],
) -> None:
    train_state = none_throws(state.train_state)

    logger.info(
        f"Started train with max_epochs={train_state.max_epochs}, max_steps={train_state.max_steps}, max_steps_per_epoch={train_state.max_steps_per_epoch}"
    )
    state._active_phase = ActivePhase.TRAIN

    # Set all modules to train() mode
    # access modules made available through _AppStateMixin
    tracked_modules = train_unit.tracked_modules()
    prior_module_train_states = _set_module_training_mode(tracked_modules, True)

    with _get_timing_context(state, f"{train_unit.__class__.__name__}.on_train_start"):
        train_unit.on_train_start(state)
    _run_callback_fn(callbacks, "on_train_start", state, train_unit)

    while not (
        state.should_stop
        or _is_done(train_state.progress, train_state.max_epochs, train_state.max_steps)
    ):
        _train_epoch_impl(state, train_unit, callbacks)

    with _get_timing_context(state, f"{train_unit.__class__.__name__}.on_train_end"):
        train_unit.on_train_end(state)
    _run_callback_fn(callbacks, "on_train_end", state, train_unit)

    # Reset training mode for modules at the end of the epoch
    # This ensures that side-effects made by the loop are reset before
    # returning back to the user
    _reset_module_training_mode(tracked_modules, prior_module_train_states)


def _train_epoch_impl(
    state: State,
    train_unit: TTrainUnit,
    callbacks: List[Callback],
) -> None:
    logger.info("Started train epoch")
    state._active_phase = ActivePhase.TRAIN

    # Set all modules to train() mode
    # access modules made available through _AppStateMixin
    tracked_modules = train_unit.tracked_modules()
    prior_module_train_states = _set_module_training_mode(tracked_modules, True)

    train_state = none_throws(state.train_state)

    evaluate_every_n_steps = None
    evaluate_every_n_epochs = None
    if state.eval_state:
        if state.eval_state.evaluate_every_n_steps:
            evaluate_every_n_steps = state.eval_state.evaluate_every_n_steps
        if state.eval_state.evaluate_every_n_epochs:
            evaluate_every_n_epochs = state.eval_state.evaluate_every_n_epochs

    # Check the progress to conditionally run this
    # to avoid running this multiple times
    # in the case of resuming from a checkpoint mid-epoch
    if train_state.progress.num_steps_completed_in_epoch == 0:
        with _get_timing_context(
            state, f"{train_unit.__class__.__name__}.on_train_epoch_start"
        ):
            train_unit.on_train_epoch_start(state)
        _run_callback_fn(callbacks, "on_train_epoch_start", state, train_unit)

    _maybe_set_distributed_sampler_epoch(
        train_state.dataloader, train_state.progress.num_epochs_completed
    )

    with _get_timing_context(state, "train.iter(dataloader)"):
        data_iter = iter(train_state.dataloader)
    step_input = data_iter

    pass_data_iter_to_step = _step_requires_iterator(train_unit.train_step)
    is_auto_unit = isinstance(train_unit, AutoUnit)

    prev_steps_in_epoch = train_state.progress.num_steps_completed_in_epoch

    while not (
        state.should_stop
        or _is_epoch_done(
            train_state.progress, train_state.max_steps_per_epoch, train_state.max_steps
        )
    ):
        try:
            if not pass_data_iter_to_step:
                # get the next batch from the data iterator
                with _get_timing_context(state, "train.next(data_iter)"):
                    step_input = next(data_iter)

            _run_callback_fn(callbacks, "on_train_step_start", state, train_unit)

            with _get_timing_context(
                state,
                f"{train_unit.__class__.__name__}.train_step",
                skip_timer=is_auto_unit,
                # skip timer if train_unit is a subclass of AutoUnit because there is additional timing in the AutoUnit, and all timing should be mutually exclusive
            ):
                train_state._step_output = train_unit.train_step(state, step_input)

            train_state.progress.increment_step()
            _run_callback_fn(callbacks, "on_train_step_end", state, train_unit)

            # clear step_output to avoid retaining extra memory
            train_state._step_output = None

            if (
                evaluate_every_n_steps
                and train_state.progress.num_steps_completed % evaluate_every_n_steps
                == 0
            ):
                _evaluate_impl(
                    state,
                    # pyre-ignore: Incompatible parameter type [6]
                    train_unit,
                    callbacks,
                )
                logger.info("Finished evaluation. Resuming training epoch")
                state._active_phase = ActivePhase.TRAIN

        except StopIteration:
            break

    # Possibly warn about an empty dataloader
    any_steps_completed = (
        abs(train_state.progress.num_steps_completed_in_epoch - prev_steps_in_epoch) > 0
    )
    if not any_steps_completed:
        logger.warning("No steps completed during train epoch!")

    # set progress counters for the next epoch
    train_state.progress.increment_epoch()

    with _get_timing_context(
        state, f"{train_unit.__class__.__name__}.on_train_epoch_end"
    ):
        train_unit.on_train_epoch_end(state)
    _run_callback_fn(callbacks, "on_train_epoch_end", state, train_unit)

    if (
        evaluate_every_n_epochs
        and train_state.progress.num_epochs_completed % evaluate_every_n_epochs == 0
    ):
        _evaluate_impl(
            state,
            # pyre-ignore: Incompatible parameter type [6]
            train_unit,
            callbacks,
        )
        state._active_phase = ActivePhase.TRAIN

    # Reset training mode for modules at the end of the epoch
    # This ensures that side-effects made by the loop are reset before
    # returning back to the user
    _reset_module_training_mode(tracked_modules, prior_module_train_states)

    logger.info("Ended train epoch")
