# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.progress import Progress

logger: logging.Logger = logging.getLogger(__name__)


class EmptyDataloaderDetectorCallback(Callback):
    """
    A callback that detects consecutive empty epochs and raises an error or warning when a threshold is reached.

    This callback helps identify issues where dataloaders return empty batches, which can cause confusing
    downstream problems that are hard to debug. It implements a fail-fast strategy to surface these issues early.
    """

    def __init__(
        self,
        threshold: int = 2,
        raise_exception: bool = True,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be a positive integer")

        self._threshold = threshold
        self._raise_exception = raise_exception
        self._consecutive_empty_train_epochs = 0
        self._consecutive_empty_eval_epochs = 0
        self._consecutive_empty_predict_epochs = 0

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        self._handle_epoch_end("train", unit.train_progress)

    def on_eval_epoch_end(self, state: State, unit: TEvalUnit) -> None:
        self._handle_epoch_end("eval", unit.eval_progress)

    def on_predict_epoch_end(self, state: State, unit: TPredictUnit) -> None:
        self._handle_epoch_end("predict", unit.predict_progress)

    def _handle_epoch_end(self, phase: str, progress: Progress) -> None:
        num_steps = progress.num_steps_completed_in_prev_epoch

        if phase == "train":
            consecutive_count = self._consecutive_empty_train_epochs
        elif phase == "eval":
            consecutive_count = self._consecutive_empty_eval_epochs
        else:
            consecutive_count = self._consecutive_empty_predict_epochs

        self._check_empty_epoch(
            num_steps,
            phase,
            progress.num_epochs_completed,
            consecutive_count,
        )

        if num_steps == 0:
            if phase == "train":
                self._consecutive_empty_train_epochs += 1
            elif phase == "eval":
                self._consecutive_empty_eval_epochs += 1
            else:
                self._consecutive_empty_predict_epochs += 1
        else:
            if phase == "train":
                self._consecutive_empty_train_epochs = 0
            elif phase == "eval":
                self._consecutive_empty_eval_epochs = 0
            else:
                self._consecutive_empty_predict_epochs = 0

    def _check_empty_epoch(
        self,
        num_steps: int,
        phase: str,
        epoch_num: int,
        consecutive_count: int,
    ) -> None:
        if num_steps == 0:
            logger.warning(
                f"Empty {phase} epoch detected! Epoch {epoch_num} completed 0 steps. "
                f"Consecutive empty {phase} epochs: {consecutive_count + 1}"
            )

            if consecutive_count + 1 >= self._threshold:
                error_msg = (
                    f"Detected {consecutive_count + 1} consecutive empty {phase} epochs, "
                    f"which exceeds the threshold of {self._threshold}. This indicates that the "
                    f"dataloader is returning empty batches, which could be due to an empty "
                    f"training table or infrastructure issues with the dataloader."
                )

                if self._raise_exception:
                    raise RuntimeError(error_msg)
                else:
                    logger.warning(error_msg)
