#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
import unittest
from typing import Optional
from unittest.mock import patch

import torch
from torchtnt.utils.distributed import (
    all_gather_tensors,
    get_process_group_backend_from_device,
    rank_zero_fn,
    revert_sync_batchnorm,
)


NUM_MP_TESTS = 2  # number of tests needing multiprocessing sockets


class DistributedTest(unittest.TestCase):
    def setUp(self):
        """
        Preparation: Pre-aggregate all free socket ports
        """
        self._free_ports = []
        for _ in range(NUM_MP_TESTS):
            sock = socket.socket()
            sock.bind(("localhost", 0))
            self._free_ports.append(str(sock.getsockname()[1]))

    def test_get_process_group_backend_cpu(self) -> None:
        device = torch.device("cpu")
        pg_backend = get_process_group_backend_from_device(device)
        self.assertEqual(pg_backend, "gloo")

    def test_get_process_group_backend_gpu(self) -> None:
        device = torch.device("cuda:0")
        pg_backend = get_process_group_backend_from_device(device)
        self.assertEqual(pg_backend, "nccl")

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_gather_uneven(self, world_size: Optional[int] = 4) -> None:
        torch.multiprocessing.spawn(
            self._test_ddp_gather_uneven_tensors,
            args=(world_size, self._free_ports[0]),
            nprocs=world_size,
        )

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_gather_uneven_multidim(self, world_size: Optional[int] = 4) -> None:
        torch.multiprocessing.spawn(
            self._test_ddp_gather_uneven_tensors_multidim,
            args=(world_size, self._free_ports[1]),
            nprocs=world_size,
        )

    @staticmethod
    def _setup_ddp(rank: int, world_size: int, free_port: str) -> None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = free_port

        torch.distributed.init_process_group(
            "gloo" if not torch.cuda.is_available() else "nccl",
            rank=rank,
            world_size=world_size,
        )

    @staticmethod
    def _test_ddp_gather_uneven_tensors(
        rank: int, worldsize: int, free_port: str
    ) -> None:
        DistributedTest._setup_ddp(rank, worldsize, free_port)
        tensor = torch.ones(rank)
        result = all_gather_tensors(tensor)
        assert len(result) == worldsize
        for idx in range(worldsize):
            assert len(result[idx]) == idx
            assert (result[idx] == torch.ones_like(result[idx])).all()

    @staticmethod
    def _test_ddp_gather_uneven_tensors_multidim(
        rank: int, worldsize: int, free_port: str
    ) -> None:
        DistributedTest._setup_ddp(rank, worldsize, free_port)
        tensor = torch.ones(rank + 1, 4 - rank)
        result = all_gather_tensors(tensor)
        assert len(result) == worldsize
        for idx in range(worldsize):
            val = result[idx]
            assert val.shape == (idx + 1, 4 - idx)
            assert (val == torch.ones_like(val)).all()

    def test_rank_zero_fn_rank_zero(self):
        @rank_zero_fn
        def foo():
            return 1

        x = foo()
        assert x == 1

    @patch("torchtnt.utils.distributed.get_global_rank")
    def test_rank_zero_fn_rank_non_zero(self, get_global_rank) -> None:
        get_global_rank.return_value = 1

        @rank_zero_fn
        def foo():
            return 1

        x = foo()
        assert x is None

    def test_revert_sync_batchnorm(self) -> None:
        original_batchnorm = torch.nn.modules.batchnorm.BatchNorm1d(4)
        original_batchnorm.running_mean.random_(-1, 1)
        original_batchnorm.running_var.random_(0, 1)
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            original_batchnorm,
        )

        sync_bn_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        reverted_model = revert_sync_batchnorm(sync_bn_model)

        _, batch_norm = reverted_model.children()
        self.assertIsInstance(batch_norm, torch.nn.modules.batchnorm._BatchNorm)
        self.assertNotIsInstance(batch_norm, torch.nn.SyncBatchNorm)
        self.assertTrue(
            torch.equal(batch_norm.running_mean, original_batchnorm.running_mean)
        )
        self.assertTrue(
            torch.equal(batch_norm.running_var, original_batchnorm.running_var)
        )
