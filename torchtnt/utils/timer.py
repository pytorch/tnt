#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import warnings
from contextlib import contextmanager
from time import perf_counter
from typing import Dict, Generator, Optional, TypeVar

import torch
import torch.distributed as dist

AsyncOperator = TypeVar("AsyncOperator")


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


class FullSyncPeriodicTimer:
    """
    Measures time (resets if given interval elapses) on rank 0
    and propagates result to other ranks.
    Propagation is done asynchronously from previous step
    in order to avoid blocking of a training process.
    """

    def __init__(self, interval: datetime.timedelta, cpu_pg: dist.ProcessGroup) -> None:
        self._interval = interval
        self._cpu_pg = cpu_pg
        self._prev_time: float = perf_counter()
        self._timeout_tensor: torch.Tensor = torch.zeros(1, dtype=torch.int)
        # pyre-fixme[34]: `Variable[AsyncOperator]` isn't present in the function's parameters.
        self._prev_work: Optional[AsyncOperator] = None

    def check(self) -> bool:
        ret = False
        curr_time = perf_counter()

        if self._prev_work is not None:
            # pyre-fixme[16]: `Variable[AsyncOperator]` has no attribute wait.
            self._prev_work.wait()
            ret = self._timeout_tensor[0].item() == 1
            if ret:
                self._prev_time = curr_time

        self._timeout_tensor[0] = (
            1 if (curr_time - self._prev_time) >= self._interval.total_seconds() else 0
        )
        self._prev_work = dist.broadcast(
            self._timeout_tensor, 0, group=self._cpu_pg, async_op=True
        )

        return ret
