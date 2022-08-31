#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed.launcher as pet
from torchtnt.utils import (
    get_device_from_env,
    get_pet_launch_config,
    get_process_group_backend_from_device,
)
from torchtnt.utils.env import init_from_env


class EnvTest(unittest.TestCase):
    def test_init_from_env(self) -> None:
        """Integration test to confirm consistency across device initialization utilities."""
        device = init_from_env()
        self.assertEqual(device, get_device_from_env())
        self.assertFalse(torch.distributed.is_initialized())

    @staticmethod
    def _test_worker_fn(init_pg_explicit: bool) -> None:
        """
        Integration test to confirm distributed initialization and consistency with process group backend utilities.
        """
        if init_pg_explicit:
            torch.distributed.init_process_group(backend="gloo")
        device = init_from_env()
        if not torch.distributed.is_initialized():
            raise AssertionError("Expected torch.distributed to be initialized")
        device_from_env = get_device_from_env()
        if device != device_from_env:
            raise AssertionError(
                f"Expected different device: received {device}, expected {device_from_env}"
            )
        pg_backend = torch.distributed.get_backend()
        expected_pg_backend = (
            get_process_group_backend_from_device(device)
            if not init_pg_explicit
            else "gloo"
        )
        if pg_backend != expected_pg_backend:
            raise AssertionError(
                f"Expected different process group backend: received {pg_backend}, expected {expected_pg_backend}"
            )
        return device

    def _test_launch_worker(
        self,
        num_processes: int,
        init_pg_explicit: bool,
    ) -> None:
        lc = get_pet_launch_config(num_processes)
        pet.elastic_launch(lc, entrypoint=self._test_worker_fn)(init_pg_explicit)

    def test_init_from_env_no_dup(self) -> None:
        self._test_launch_worker(2, init_pg_explicit=False)
        # trivial test case to ensure test passes with no exceptions
        self.assertTrue(True)

    def test_init_from_env_dup(self) -> None:
        self._test_launch_worker(2, init_pg_explicit=True)
        # trivial test case to ensure test passes with no exceptions
        self.assertTrue(True)
