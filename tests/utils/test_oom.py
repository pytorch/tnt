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
from torchtnt.utils.oom import (
    _bytes_to_mb_gb,
    is_out_of_cpu_memory,
    is_out_of_cuda_memory,
    is_out_of_memory_error,
    log_memory_snapshot,
)
from torchtnt.utils.version import is_torch_version_geq_2_0


class OomTest(unittest.TestCase):

    # pyre-fixme[4]: Attribute must be annotated.
    cuda_available = torch.cuda.is_available()

    def test_is_out_of_cpu_memory(self) -> None:
        """Test CPU OOM error detection."""
        cpu_oom_error = RuntimeError("DefaultCPUAllocator: can't allocate memory")
        self.assertTrue(is_out_of_cpu_memory(cpu_oom_error))
        not_cpu_oom_error = RuntimeError("RuntimeError: blah")
        self.assertFalse(is_out_of_cpu_memory(not_cpu_oom_error))

    def test_is_out_of_cuda_memory(self) -> None:
        """Test cuda OOM error detection."""
        cuda_oom_error_1 = RuntimeError("CUDA out of memory. Tried to allocate ...")
        self.assertTrue(is_out_of_cuda_memory(cuda_oom_error_1))
        cuda_oom_error_2 = RuntimeError(
            "RuntimeError: cuda runtime error (2) : out of memory"
        )
        self.assertTrue(is_out_of_cuda_memory(cuda_oom_error_2))
        not_cuda_oom_error = RuntimeError("RuntimeError: blah")
        self.assertFalse(is_out_of_cuda_memory(not_cuda_oom_error))

    def test_is_out_of_memory_error(self) -> None:
        """Test general OOM error detection."""
        cpu_oom_error = RuntimeError("DefaultCPUAllocator: can't allocate memory")
        self.assertTrue(is_out_of_memory_error(cpu_oom_error))
        cuda_oom_error_1 = RuntimeError("CUDA out of memory. Tried to allocate ...")
        self.assertTrue(is_out_of_memory_error(cuda_oom_error_1))
        cuda_oom_error_2 = RuntimeError(
            "RuntimeError: cuda runtime error (2) : out of memory"
        )
        self.assertTrue(is_out_of_memory_error(cuda_oom_error_2))
        not_oom_error = RuntimeError("RuntimeError: blah")
        self.assertFalse(is_out_of_memory_error(not_oom_error))

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torchtnt.utils.version.is_torch_version_geq_2_0()` to decorator factory
    #  `unittest.skipUnless`.
    @unittest.skipUnless(
        condition=is_torch_version_geq_2_0(),
        reason="This test needs changes from PyTorch 2.0 to run.",
    )
    def test_log_memory_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Record history
            torch.cuda.memory._record_memory_history(enabled="all", max_entries=10000)

            # initialize device & allocate memory for tensors
            device = get_device_from_env()
            a = torch.rand((1024, 1024), device=device)
            b = torch.rand((1024, 1024), device=device)
            _ = (a + b) * (a - b)

            # save a snapshot
            log_memory_snapshot(temp_dir, "foo")

            # Check if the corresponding files exist
            save_dir = os.path.join(temp_dir, "foo_rank0")

            pickle_dump_path = os.path.join(save_dir, "snapshot.pickle")
            self.assertTrue(os.path.exists(pickle_dump_path))

            trace_path = os.path.join(save_dir, "trace_plot.html")
            self.assertTrue(os.path.exists(trace_path))

            segment_plot_path = os.path.join(save_dir, "segment_plot.html")
            self.assertTrue(os.path.exists(segment_plot_path))

    def test_bytes_to_mb_gb(self) -> None:
        bytes_to_mb_test_cases = [
            (0, "0.0 MB"),
            (100000, "0.1 MB"),
            (1000000, "0.95 MB"),
            (1000000000, "0.93 GB"),
            (1000000000000, "931.32 GB"),
        ]
        for inp, expected in bytes_to_mb_test_cases:
            self.assertEqual(expected, _bytes_to_mb_gb(inp))
