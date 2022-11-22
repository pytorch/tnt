# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sized
from typing import Iterable, Optional

from pyre_extensions import none_throws

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.distributed import get_global_rank
from tqdm.auto import tqdm


class TQDMProgressBar(Callback):
    """
    A callback for progress bar visualization in training, evaluation, and prediction.
    It is initialized only on rank 0 in distributed environments.

    Args:
        refresh_rate: Determines at which rate (in number of steps) the progress bars get updated.
    """

    def __init__(self, refresh_rate: int = 1) -> None:
        self._refresh_rate = refresh_rate

        self._train_progress_bar: Optional[tqdm] = None
        self._eval_progress_bar: Optional[tqdm] = None
        self._predict_progress_bar: Optional[tqdm] = None

    def on_train_epoch_start(self, state: State, unit: TTrainUnit) -> None:
        train_state = none_throws(state.train_state)
        self._train_progress_bar = _create_progress_bar(
            train_state.dataloader,
            desc="Train Epoch",
            num_epochs_completed=train_state.progress.num_epochs_completed,
            num_steps_completed=train_state.progress.num_steps_completed,
            max_steps=train_state.max_steps,
            max_steps_per_epoch=train_state.max_steps_per_epoch,
        )

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        train_state = none_throws(state.train_state)
        if self._train_progress_bar is not None:
            _update_progress_bar(
                self._train_progress_bar,
                train_state.progress.num_steps_completed,
                self._refresh_rate,
            )

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        train_state = none_throws(state.train_state)
        if self._train_progress_bar is not None:
            _close_progress_bar(
                self._train_progress_bar,
                train_state.progress.num_steps_completed,
                self._refresh_rate,
            )

    def on_eval_epoch_start(self, state: State, unit: TEvalUnit) -> None:
        eval_state = none_throws(state.eval_state)
        self._eval_progress_bar = _create_progress_bar(
            eval_state.dataloader,
            desc="Eval Epoch",
            num_epochs_completed=eval_state.progress.num_epochs_completed,
            num_steps_completed=eval_state.progress.num_steps_completed,
            max_steps=eval_state.max_steps,
            max_steps_per_epoch=eval_state.max_steps_per_epoch,
        )

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        eval_state = none_throws(state.eval_state)
        if self._eval_progress_bar is not None:
            _update_progress_bar(
                self._eval_progress_bar,
                eval_state.progress.num_steps_completed,
                self._refresh_rate,
            )

    def on_eval_epoch_end(self, state: State, unit: TEvalUnit) -> None:
        eval_state = none_throws(state.eval_state)
        if self._eval_progress_bar is not None and state.eval_state:
            _close_progress_bar(
                self._eval_progress_bar,
                eval_state.progress.num_steps_completed,
                self._refresh_rate,
            )

    def on_predict_epoch_start(self, state: State, unit: TPredictUnit) -> None:
        predict_state = none_throws(state.predict_state)
        self._predict_progress_bar = _create_progress_bar(
            predict_state.dataloader,
            desc="Predict Epoch",
            num_epochs_completed=predict_state.progress.num_epochs_completed,
            num_steps_completed=predict_state.progress.num_steps_completed,
            max_steps=predict_state.max_steps,
            max_steps_per_epoch=predict_state.max_steps_per_epoch,
        )

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        predict_state = none_throws(state.predict_state)
        if self._predict_progress_bar is not None:
            _update_progress_bar(
                self._predict_progress_bar,
                predict_state.progress.num_steps_completed,
                self._refresh_rate,
            )

    def on_predict_epoch_end(self, state: State, unit: TPredictUnit) -> None:
        predict_state = none_throws(state.predict_state)
        if self._predict_progress_bar is not None:
            _close_progress_bar(
                self._predict_progress_bar,
                predict_state.progress.num_steps_completed,
                self._refresh_rate,
            )


def _create_progress_bar(
    # pyre-ignore: Invalid type parameters [24]
    dataloader: Iterable,
    *,
    desc: str,
    num_epochs_completed: int,
    num_steps_completed: int,
    max_steps: Optional[int],
    max_steps_per_epoch: Optional[int],
) -> Optional[tqdm]:
    if not get_global_rank() == 0:
        return None

    current_epoch = num_epochs_completed
    total = _estimated_steps_in_epoch(
        dataloader,
        num_steps_completed=num_steps_completed,
        max_steps=max_steps,
        max_steps_per_epoch=max_steps_per_epoch,
    )
    return tqdm(desc=f"{desc} {current_epoch}", total=total)


def _update_progress_bar(
    progress_bar: tqdm, num_steps_completed: int, refresh_rate: int
) -> None:
    if not get_global_rank() == 0:
        return

    if num_steps_completed % refresh_rate == 0:
        progress_bar.update(refresh_rate)


def _close_progress_bar(
    progress_bar: tqdm, num_steps_completed: int, refresh_rate: int
) -> None:
    if not get_global_rank() == 0:
        return

    # complete remaining progress in bar
    progress_bar.update(num_steps_completed % refresh_rate)
    progress_bar.close()


def _estimated_steps_in_epoch(
    # pyre-ignore: Invalid type parameters [24]
    dataloader: Iterable,
    *,
    num_steps_completed: int,
    max_steps: Optional[int],
    max_steps_per_epoch: Optional[int],
) -> float:
    """estimate number of steps in current epoch for tqdm"""

    total = float("inf")
    if isinstance(dataloader, Sized):
        total = len(dataloader)

    if max_steps_per_epoch and max_steps:
        total = min(total, max_steps_per_epoch, max_steps - num_steps_completed)
    elif max_steps:
        total = min(total, max_steps - num_steps_completed)
    elif max_steps_per_epoch:
        total = min(total, max_steps_per_epoch)
    return total
