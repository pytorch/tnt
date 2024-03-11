# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import re
import time
from datetime import timedelta
from typing import Literal, Union

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit
from torchtnt.utils.distributed import get_global_rank, sync_bool
from torchtnt.utils.rank_zero_log import rank_zero_info


class TimeLimitInterrupter(Callback):
    """
    This callback tracks the time spent in training and stops the training loop when it exceeds the specified duration.

    Args:
        duration: The maximum amount of time to spend in training. Can be specified as a string in the form of DD:HH:MM (days, hours, minutes) or as a timedelta.
            For example, to specify 20 hours is "00:20:00".
        interval: Can be either "epoch" or "step". Determines whether to check for time limit exceeding on every epoch or step.
        interval_freq: How often to check for time limit exceeding. For example, if interval is "epoch" and interval_freq is 2, then the callback will check every two epochs.

    Note:
        This callback uses the global process group to communicate between ranks.

    """

    def __init__(
        self,
        duration: Union[str, timedelta],
        interval: Literal["epoch", "step"] = "epoch",
        interval_freq: int = 1,
    ) -> None:
        if isinstance(duration, str):
            # checks if string matches DD:HH:MM format and is within valid range
            # 00 <= DD <= 99
            # 00 <= HH <= 23
            # 00 <= MM <= 59
            pattern = r"^\d{2}:(0[0-9]|1[0-9]|2[0-3]):([0-5][0-9])$"
            if not re.match(pattern, duration):
                raise ValueError(
                    f"Invalid duration format '{duration}'. Expected format is DD:HH:MM"
                )
            duration_format = duration.strip().split(":")
            duration_format = list(map(int, duration_format))
            duration = timedelta(
                days=duration_format[0],
                hours=duration_format[1],
                minutes=duration_format[2],
            )

        self._duration: float = duration.total_seconds()
        self._interval = interval
        self._interval_freq = interval_freq

        self._rank: int = get_global_rank()
        self._start_time: float = 0

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        if self._rank == 0:
            self._start_time = time.monotonic()

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        if self._interval == "step":
            if unit.train_progress.num_steps_completed % self._interval_freq == 0:
                self._should_stop(state)

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        if self._interval == "epoch":
            if unit.train_progress.num_epochs_completed % self._interval_freq == 0:
                self._should_stop(state)

    def _should_stop(self, state: State) -> None:
        """
        All ranks sync with rank 0 determine if time limit has exceeded.
        If so, indicates the training loop to stop.
        """

        if self._rank == 0:
            time_elapsed = time.monotonic() - self._start_time
            should_stop = time_elapsed >= self._duration
        else:
            should_stop = False

        should_stop = sync_bool(should_stop, coherence_mode="rank_zero")
        if should_stop:
            rank_zero_info(
                f"Training duration of {self._duration} seconds has exceeded. Time elapsed is {time.monotonic() - self._start_time} seconds. Stopping training."
            )
            state.stop()
