# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.memory_snapshot_profiler import (
    MemorySnapshotParams,
    MemorySnapshotProfiler,
)

logger: logging.Logger = logging.getLogger(__name__)


class MemorySnapshot(Callback):

    """
    A callback for memory snapshot collection during training, saving pickle files to the user-specified directory.
    Uses `Memory Snapshots <https://zdevito.github.io/2022/08/16/memory-snapshots.html>`.

    Args:
        output_dir: Directory where to save the memory snapshots.
        memory_snapshot_params: Instance of MemorySnapshotParams which will be passed to MemorySnapshotProfiler.

    Note: It is recommended to instantiate this callback **as early as possible** in your training/eval/prediction script,
        ideally before model initialization, to make sure all memory allocation is captured.

    """

    def __init__(
        self,
        *,
        output_dir: str,
        memory_snapshot_params: Optional[MemorySnapshotParams] = None,
    ) -> None:
        self.memory_snapshot_profiler = MemorySnapshotProfiler(
            output_dir=output_dir, memory_snapshot_params=memory_snapshot_params
        )

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        self.memory_snapshot_profiler.step()

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        self.memory_snapshot_profiler.step()

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        self.memory_snapshot_profiler.step()
