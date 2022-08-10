#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.launcher as launcher
from torchtnt.utils.early_stop_checker import EarlyStopChecker
from torchtnt.utils.full_sync_early_stop_checker import FullSyncEarlyStopChecker
from torchtnt.utils.test_utils import get_pet_launch_config


class FullSyncEarlyStopCheckerTest(unittest.TestCase):
    @classmethod
    def _full_sync_worker(cls, coherence_mode: Optional[str]) -> bool:
        dist.init_process_group("gloo")
        full_sync_es = FullSyncEarlyStopChecker(
            EarlyStopChecker("min", 3), coherence_mode=coherence_mode
        )
        if dist.get_rank() == 0:  # rank 0
            losses = [0.25, 0.25, 0.26, 0.25]
        else:  # rank 1
            losses = [0.4, 0.3, 0.28, 0.25]
        for loss in losses:
            should_stop = full_sync_es.check(loss)
        return should_stop

    def test_full_sync_early_stop_single_process(self) -> None:
        # Initialize an early stop checker and a full sync early stop checker
        losses = [0.4, 0.3, 0.28, 0.25, 0.26, 0.25]
        es = EarlyStopChecker("min", 3)
        full_sync_es = FullSyncEarlyStopChecker(EarlyStopChecker("min", 3))

        for loss in losses:
            should_stop = es.check(torch.tensor(loss))
            self.assertFalse(should_stop)
            should_stop = full_sync_es.check(torch.tensor(loss))
            self.assertFalse(should_stop)

        # Patience should run out for both es and full_async_es
        should_stop = es.check(torch.tensor(0.25))
        self.assertTrue(should_stop)
        should_stop = full_sync_es.check(torch.tensor(0.25))
        self.assertTrue(should_stop)

    def test_full_sync_early_stop_multi_process_coherence_mode_rank_zero(self) -> None:
        config = get_pet_launch_config(2)
        # Launch 2 worker processes. Each will check for early stopping
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(
            "rank_zero"
        )
        # Both processes should return True using full sync checker with 'zero' coherence_mode
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    def test_full_sync_early_stop_multi_process_coherence_mode_any(self) -> None:
        config = get_pet_launch_config(2)
        # Launch 2 worker processes. Each will check for early stopping
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(
            "any"
        )
        # Both processes should return True using full sync checker with 'any' coherence_mode
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    def test_full_sync_early_stop_multi_process_coherence_mode_all(self) -> None:
        config = get_pet_launch_config(2)
        # Launch 2 worker processes. Each will check for early stopping
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(
            "all"
        )
        # Both processes should return False using full sync checker with 'all' coherence_mode
        self.assertFalse(result[0])
        self.assertFalse(result[1])

    def test_full_sync_early_stop_multi_process_coherence_mode_none(self) -> None:
        config = get_pet_launch_config(2)
        # Launch 2 worker processes. Each will check for early stopping
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(
            None
        )
        # Both processes should return False using full sync checker with 'all' coherence_mode
        self.assertTrue(result[0])  # return value from rank 0 process
        self.assertFalse(result[1])  # return value from rank 1 process
