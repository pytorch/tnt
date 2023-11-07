#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import torch
from torchtnt.utils.device import get_device_from_env
from torchtnt.utils.memory_snapshot_profiler import (
    MemorySnapshotParams,
    MemorySnapshotProfiler,
)
from torchtnt.utils.version import is_torch_version_geq_2_0


class MemorySnapshotProfilerTest(unittest.TestCase):

    cuda_available: bool = torch.cuda.is_available()
    torch_version_geq_2_0: bool = is_torch_version_geq_2_0()

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    @unittest.skipUnless(
        condition=torch_version_geq_2_0,
        reason="This test needs changes from PyTorch 2.0 to run.",
    )
    def test_stop_step(self) -> None:
        """Test that a memory snapshot is saved when stop_step is reached."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_snapshot_profiler = MemorySnapshotProfiler(
                output_dir=temp_dir,
                memory_snapshot_params=MemorySnapshotParams(start_step=0, stop_step=2),
            )

            # initialize device & allocate memory for tensors
            device = get_device_from_env()
            a = torch.rand((1024, 1024), device=device)
            b = torch.rand((1024, 1024), device=device)
            _ = (a + b) * (a - b)

            memory_snapshot_profiler.step()

            # Check if the corresponding files exist
            save_dir = os.path.join(temp_dir, "memory_snapshot", "oom_rank0")

            pickle_dump_path = os.path.join(save_dir, "snapshot.pickle")
            trace_path = os.path.join(save_dir, "trace_plot.html")
            segment_plot_path = os.path.join(save_dir, "segment_plot.html")

            # after first step files do not exist
            self.assertFalse(os.path.exists(pickle_dump_path))
            self.assertFalse(os.path.exists(trace_path))
            self.assertFalse(os.path.exists(segment_plot_path))

            # after second step stop_step is reached and files should exist
            memory_snapshot_profiler.step()
            self.assertTrue(os.path.exists(pickle_dump_path))
            self.assertTrue(os.path.exists(trace_path))
            self.assertTrue(os.path.exists(segment_plot_path))

    @unittest.skipUnless(
        condition=torch_version_geq_2_0,
        reason="This test needs changes from PyTorch 2.0 to run.",
    )
    def test_validation(self) -> None:
        """Test parameter validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "start_step must be nonnegative."):
                _ = MemorySnapshotProfiler(
                    output_dir=temp_dir,
                    memory_snapshot_params=MemorySnapshotParams(
                        start_step=-1, stop_step=0
                    ),
                )
            with self.assertRaisesRegex(
                ValueError, "stop_step must be specified when start_step is set."
            ):
                _ = MemorySnapshotProfiler(
                    output_dir=temp_dir,
                    memory_snapshot_params=MemorySnapshotParams(
                        start_step=2, stop_step=None
                    ),
                )
            with self.assertRaisesRegex(ValueError, "start_step must be < stop_step."):
                _ = MemorySnapshotProfiler(
                    output_dir=temp_dir,
                    memory_snapshot_params=MemorySnapshotParams(
                        start_step=2, stop_step=0
                    ),
                )
            with self.assertRaisesRegex(ValueError, "stop_step must be positive."):
                _ = MemorySnapshotProfiler(
                    output_dir=temp_dir,
                    memory_snapshot_params=MemorySnapshotParams(stop_step=0),
                )
            with self.assertRaisesRegex(
                ValueError,
                "stop_step must be enabled with either start_step or enable_oom_observer.",
            ):
                _ = MemorySnapshotProfiler(
                    output_dir=temp_dir,
                    memory_snapshot_params=MemorySnapshotParams(
                        stop_step=2, enable_oom_observer=False
                    ),
                )
            with self.assertRaisesRegex(
                ValueError,
                "At least one of start_step/stop_step or enable_oom_observer must be set.",
            ):
                _ = MemorySnapshotProfiler(
                    output_dir=temp_dir,
                    memory_snapshot_params=MemorySnapshotParams(
                        start_step=None, stop_step=None, enable_oom_observer=False
                    ),
                )
