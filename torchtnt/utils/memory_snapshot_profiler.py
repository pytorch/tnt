# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from types import TracebackType
from typing import Optional, Type

import torch
from torchtnt.utils.oom import attach_oom_observer, log_memory_snapshot
from torchtnt.utils.version import is_torch_version_geq_2_0

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshotParams:
    """
    Memory snapshot parameters.

    Args:
        stop_step: Number of steps after which to dump memory snapshot, and stop recording memory history.
        max_entries: Maximum number of events to keep in memory.
        enable_oom_observer: Whether to attach an observer to record OOM events. If stop_step is set, the
            OOM observer will only be active until stop_step is reached.
    """

    stop_step: Optional[int] = 2
    max_entries: int = 100000
    enable_oom_observer: bool = True


class MemorySnapshotProfiler:
    """
    Records a history of memory allocation and free events, and dumps to a
    file which can be visualized offline. It by default keeps track of
    100000 events, and dumps at user specified step as well as on OOM.
    The profiler is ideally started before model is instantiated in memory
    so that allocations that stay constant in training are accounted for.

    Args:
        output_dir: Directory where to save the memory snapshots.
        memory_snapshot_params: Instance of MemorySnapshotParams.
    """

    def __init__(
        self,
        output_dir: str,
        memory_snapshot_params: Optional[MemorySnapshotParams] = None,
    ) -> None:
        self.output_dir: str = output_dir
        self.params: MemorySnapshotParams = (
            memory_snapshot_params or MemorySnapshotParams()
        )
        self.step_num: int = 0

        if not is_torch_version_geq_2_0():
            raise RuntimeError("CUDA memory snapshot requires torch>=2.0")
        if self.params.enable_oom_observer:
            attach_oom_observer(
                output_dir=output_dir, trace_max_entries=self.params.max_entries
            )

        logger.info(
            f"Created MemorySnapshotProfiler with MemorySnapshotParams={self.params}."
        )

    def __enter__(self) -> None:
        self.start()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> Optional[bool]:
        self.stop()

    def start(self) -> None:
        if not torch.cuda.is_available():
            logger.warn("CUDA unavailable. Not recording memory history.")
            return

        logger.info("Starting to record memory history.")
        torch.cuda.memory._record_memory_history(max_entries=self.params.max_entries)

    def stop(self) -> None:
        if not torch.cuda.is_available():
            logger.warn("CUDA unavailable. Not recording memory history.")
            return

        logger.info("Stopping recording memory history.")
        torch.cuda.memory._record_memory_history(enabled=None)

    def step(self) -> None:
        self.step_num += 1
        if self.params.stop_step is not None and self.step_num == self.params.stop_step:
            log_memory_snapshot(output_dir=self.output_dir)
            self.stop()
