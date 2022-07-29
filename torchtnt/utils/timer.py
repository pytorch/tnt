#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from contextlib import contextmanager
from time import perf_counter
from typing import Dict, Generator

import torch


class Timer:
    """
    A timer which records intervals between starts and stops, as well as cumulative time in seconds.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset timer state."""
        self._paused: bool = True
        self._interval_start_time: float = 0.0
        self._interval_stop_time: float = 0.0
        self._total_time_seconds: float = 0.0

    def start(self) -> None:
        """Start timer interval."""
        if not self.paused:
            warnings.warn("Cannot start timer while timer is running.")
            return
        self._paused = False
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._interval_start_time = perf_counter()

    def stop(self) -> None:
        """Stop timer interval. Interval time will be added to the total."""
        if self.paused:
            warnings.warn("Cannot stop timer while timer is paused.")
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._interval_stop_time = perf_counter()
        self._paused = True
        self._total_time_seconds += self.interval_time_seconds

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def interval_time_seconds(self) -> float:
        """
        Interval between most recent stop and start in seconds.
        If timer is still running, return interval between most recent start and now.
        """
        if self._interval_start_time == 0.0:
            return 0.0
        interval_stop_time = self._interval_stop_time if self.paused else perf_counter()
        return interval_stop_time - self._interval_start_time

    @property
    def total_time_seconds(self) -> float:
        """Sum of all interval times in seconds since the last reset.
        If timer is still running, include the current interval time in the total.
        """
        running_interval = 0 if self.paused else self.interval_time_seconds
        return self._total_time_seconds + running_interval

    def state_dict(self) -> Dict[str, float]:
        """
        Pause timer and export state_dict for checkpointing.

        Raises:
            Exception:
                If state_dict is called while timer is still running.
        """
        if not self.paused:
            raise Exception("Timer must be paused before creating state_dict.")
        return {
            "interval_start_time": self._interval_start_time,
            "interval_stop_time": self._interval_stop_time,
            "total_time_seconds": self._total_time_seconds,
        }

    def load_state_dict(self, state_dict: Dict[str, float]) -> None:
        """Load timer state from state dict."""
        self._interval_start_time = state_dict["interval_start_time"]
        self._interval_stop_time = state_dict["interval_stop_time"]
        self._total_time_seconds = state_dict["total_time_seconds"]

    @contextmanager
    def time(self) -> Generator[None, None, None]:
        """Yields a context manager to encapsulate the scope of a timed action."""
        try:
            self.start()
            yield
        finally:
            self.stop()
