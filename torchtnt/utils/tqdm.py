#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable, Optional

from torchtnt.utils.progress import estimated_steps_in_epoch
from tqdm.auto import tqdm

logger: logging.Logger = logging.getLogger(__name__)


def create_progress_bar(
    dataloader: Iterable[object],
    *,
    desc: str,
    num_epochs_completed: int,
    num_steps_completed: int,
    max_steps: Optional[int],
    max_steps_per_epoch: Optional[int],
) -> tqdm:
    current_epoch = num_epochs_completed
    total = estimated_steps_in_epoch(
        dataloader,
        num_steps_completed=num_steps_completed,
        max_steps=max_steps,
        max_steps_per_epoch=max_steps_per_epoch,
    )
    return tqdm(
        desc=f"{desc} {current_epoch}",
        total=total,
        initial=num_steps_completed,
        bar_format="{l_bar}{bar}{r_bar}\n",
    )


def update_progress_bar(
    progress_bar: tqdm, num_steps_completed: int, refresh_rate: int
) -> None:
    if num_steps_completed % refresh_rate == 0:
        progress_bar.update(refresh_rate)


def close_progress_bar(
    progress_bar: tqdm, num_steps_completed: int, refresh_rate: int
) -> None:
    # complete remaining progress in bar
    progress_bar.update(num_steps_completed % refresh_rate)
    progress_bar.close()
