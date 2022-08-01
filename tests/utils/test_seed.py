#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from torchtnt.utils.seed import seed


class SeedTest(unittest.TestCase):
    def test_seed_range(self) -> None:
        """
        Verify that exceptions are raised on input values
        """
        with self.assertRaises(ValueError, msg="Invalid seed value provided"):
            seed(-1)

        invalid_max = np.iinfo(np.uint64).max
        with self.assertRaises(ValueError, msg="Invalid seed value provided"):
            seed(invalid_max)

        # should not raise any exceptions
        seed(42)

    def test_deterministic_true(self) -> None:
        for det_debug_mode, det_debug_mode_str in [(1, "warn"), (2, "error")]:
            warn_only = det_debug_mode == 1
            for deterministic in (det_debug_mode, det_debug_mode_str):
                with self.subTest(deterministic=deterministic):
                    seed(42, deterministic=deterministic)
                    self.assertTrue(torch.backends.cudnn.deterministic)
                    self.assertFalse(torch.backends.cudnn.benchmark)
                    self.assertEqual(
                        det_debug_mode, torch.get_deterministic_debug_mode()
                    )
                    self.assertTrue(torch.are_deterministic_algorithms_enabled())
                    self.assertEqual(
                        warn_only, torch.is_deterministic_algorithms_warn_only_enabled()
                    )

    def test_deterministic_false(self) -> None:
        for deterministic in ("default", 0):
            with self.subTest(deterministic=deterministic):
                seed(42, deterministic=deterministic)
                self.assertFalse(torch.backends.cudnn.deterministic)
                self.assertTrue(torch.backends.cudnn.benchmark)
                self.assertEquals(0, torch.get_deterministic_debug_mode())
                self.assertFalse(torch.are_deterministic_algorithms_enabled())
                self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_deterministic_unset(self) -> None:
        det = torch.backends.cudnn.deterministic
        benchmark = torch.backends.cudnn.benchmark
        det_debug_mode = torch.get_deterministic_debug_mode()
        det_algo_enabled = torch.are_deterministic_algorithms_enabled()
        det_algo_warn_only_enabled = (
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        seed(42, deterministic=None)
        self.assertEqual(det, torch.backends.cudnn.deterministic)
        self.assertEqual(benchmark, torch.backends.cudnn.benchmark)
        self.assertEqual(det_debug_mode, torch.get_deterministic_debug_mode())
        self.assertEqual(det_algo_enabled, torch.are_deterministic_algorithms_enabled())
        self.assertEqual(
            det_algo_warn_only_enabled,
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
