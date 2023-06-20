#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable, Optional

import torch
from torch.profiler import record_function
from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.timer import get_timer_summary, Timer

_log: logging.Logger = logging.getLogger(__name__)


def profile_dataloader(
    dataloader: Iterable[object],
    *,
    max_steps: Optional[int] = None,
    profiler: Optional[torch.profiler.profile] = None,
    timer: Optional[Timer] = None,
) -> Timer:
    """
    A helper function that profiles the dataloader iterations.

    Args:
        dataloader: dataloader to be profiled.
        max_steps (optional): maximum number of steps to run for. If not set, the dataloader will run until its iterator is exhausted.
        profiler (optional): torch profiler to be used.
        timer (optional): timer to be used to track duration.
    """
    timer = timer if timer is not None else Timer()
    with timer.time("iter(dataloader)"), record_function("iter(dataloader)"):
        data_iter = iter(dataloader)

    # If max_steps is not set, run until the dataloader is exhausted
    steps_completed = 0

    if profiler:
        profiler.start()

    while max_steps is None or (steps_completed < max_steps):
        try:
            with timer.time("next(iter)"), record_function("next(iter)"):
                next(data_iter)
            steps_completed += 1
            if profiler:
                profiler.step()
        except StopIteration:
            break

    if profiler:
        profiler.stop()

    rank = get_global_rank()
    _log.info(f"Timer summary for rank {rank}:\n{get_timer_summary(timer)}")
    return timer
