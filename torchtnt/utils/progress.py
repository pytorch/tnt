# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sized
from typing import Any, Dict, Iterable, Optional


class Progress:
    """Class to track progress during the loop. Includes state_dict/load_state_dict for convenience for checkpointing."""

    def __init__(
        self,
        num_epochs_completed: int = 0,
        num_steps_completed: int = 0,
        num_steps_completed_in_epoch: int = 0,
    ) -> None:
        self._num_epochs_completed: int = num_epochs_completed
        self._num_steps_completed: int = num_steps_completed
        self._num_steps_completed_in_epoch: int = num_steps_completed_in_epoch

    @property
    def num_epochs_completed(self) -> int:
        """Number of epochs completed thus far in loop."""
        return self._num_epochs_completed

    @property
    def num_steps_completed(self) -> int:
        """Number of steps completed thus far in loop."""
        return self._num_steps_completed

    @property
    def num_steps_completed_in_epoch(self) -> int:
        """Number of steps completed thus far in epoch."""
        return self._num_steps_completed_in_epoch

    def increment_step(self) -> None:
        """Increment the step counts completed and completed within the epoch."""
        self._num_steps_completed += 1
        self._num_steps_completed_in_epoch += 1

    def increment_epoch(self) -> None:
        """Increment the epochs completed and resets the steps completed within the epoch."""
        self._num_epochs_completed += 1
        self._num_steps_completed_in_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        """Returns a state_dict of a Progress instance in accordance with Stateful protocol."""
        return {
            "num_epochs_completed": self._num_epochs_completed,
            "num_steps_completed": self._num_steps_completed,
            "num_steps_completed_in_epoch": self._num_steps_completed_in_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores a Progress instance from a state_dict in accordance with Stateful protocol."""
        self._num_epochs_completed = state_dict["num_epochs_completed"]
        self._num_steps_completed = state_dict["num_steps_completed"]
        self._num_steps_completed_in_epoch = state_dict["num_steps_completed_in_epoch"]


def estimated_steps_in_epoch(
    dataloader: Iterable[object],
    *,
    num_steps_completed: int,
    max_steps: Optional[int],
    max_steps_per_epoch: Optional[int],
) -> float:
    """Estimate the number of remaining steps for the current epoch."""

    total = float("inf")
    if isinstance(dataloader, Sized):
        try:
            total = len(dataloader)
        except (NotImplementedError, TypeError):
            pass

    if max_steps_per_epoch and max_steps:
        total = min(total, max_steps_per_epoch, max_steps - num_steps_completed)
    elif max_steps:
        total = min(total, max_steps - num_steps_completed)
    elif max_steps_per_epoch:
        total = min(total, max_steps_per_epoch)
    return total


def estimated_steps_in_loop(
    dataloader: Iterable[object],
    *,
    max_steps: Optional[int],
    max_steps_per_epoch: Optional[int],
    epochs: Optional[int],
) -> Optional[int]:
    """
    Estimate the total number of steps for the current loop.

    A return value of None indicates that the number of steps couldn't be estimated.
    """

    if not max_steps and not epochs:
        return None

    if not epochs:
        return max_steps

    total_steps = None
    steps_per_epoch = estimated_steps_in_epoch(
        dataloader,
        num_steps_completed=0,
        max_steps=max_steps,
        max_steps_per_epoch=max_steps_per_epoch,
    )
    if steps_per_epoch != float("inf"):
        total_steps = int(steps_per_epoch) * epochs

    if total_steps and max_steps:
        return min(total_steps, max_steps)

    return total_steps or max_steps


def estimated_steps_in_fit(
    *,
    train_dataloader: Iterable[object],
    eval_dataloader: Iterable[object],
    epochs: Optional[int],
    max_steps: Optional[int],
    max_train_steps_per_epoch: Optional[int],
    max_eval_steps_per_epoch: Optional[int],
    eval_every_n_steps: Optional[int],
    eval_every_n_epochs: Optional[int],
) -> Optional[int]:
    """
    Estimate the total number of steps for fit run.

    If the number of training/eval steps couldn't be calculated, None is returned.
    """
    training_steps = estimated_steps_in_loop(
        train_dataloader,
        max_steps=max_steps,
        max_steps_per_epoch=max_train_steps_per_epoch,
        epochs=epochs,
    )
    if not training_steps:
        return None

    if not eval_every_n_steps and not eval_every_n_epochs:
        return training_steps

    number_of_eval_steps_per_eval_epoch = estimated_steps_in_loop(
        eval_dataloader,
        max_steps=None,
        max_steps_per_epoch=max_eval_steps_per_epoch,
        epochs=1,
    )
    if not number_of_eval_steps_per_eval_epoch:
        return None

    total_eval_epochs = 0
    if eval_every_n_epochs and epochs:
        total_eval_epochs += epochs // eval_every_n_epochs

    if eval_every_n_steps:
        total_eval_epochs += training_steps // eval_every_n_steps

    return training_steps + total_eval_epochs * number_of_eval_steps_per_eval_epoch
