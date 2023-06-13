#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from typing import Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.launcher as launcher
from torchtnt.utils.device import get_device_from_env
from torchtnt.utils.distributed import (
    all_gather_tensors,
    destroy_process_group,
    get_global_rank,
    get_local_rank,
    get_process_group_backend_from_device,
    get_world_size,
    rank_zero_fn,
    revert_sync_batchnorm,
    sync_bool,
)
from torchtnt.utils.test_utils import get_pet_launch_config


class DistributedTest(unittest.TestCase):
    def test_get_process_group_backend_cpu(self) -> None:
        device = torch.device("cpu")
        pg_backend = get_process_group_backend_from_device(device)
        self.assertEqual(pg_backend, "gloo")

    def test_get_process_group_backend_gpu(self) -> None:
        device = torch.device("cuda:0")
        pg_backend = get_process_group_backend_from_device(device)
        self.assertEqual(pg_backend, "nccl")

    def test_get_world_size_single(self) -> None:
        self.assertEqual(get_world_size(), 1)

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_get_world_size(self) -> None:
        world_size = 4
        config = get_pet_launch_config(world_size)
        launcher.elastic_launch(config, entrypoint=self._test_get_world_size)(
            world_size
        )

    @staticmethod
    def _test_get_world_size(world_size: int) -> None:
        assert get_world_size() == world_size

        dist.init_process_group("gloo")
        assert get_world_size() == dist.get_world_size()

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_get_global_rank(self) -> None:
        config = get_pet_launch_config(4)
        launcher.elastic_launch(config, entrypoint=self._test_get_global_rank)()

    @staticmethod
    def _test_get_global_rank() -> None:
        dist.init_process_group("gloo")
        assert get_global_rank() == dist.get_rank()

    def test_get_global_rank_single(self) -> None:
        self.assertEqual(get_global_rank(), 0)

    def test_get_local_rank_single(self) -> None:
        self.assertEqual(get_local_rank(), 0)

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_get_local_rank(self) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(config, entrypoint=self._test_get_local_rank)()

    @staticmethod
    def _test_get_local_rank() -> None:
        # when launched on a single node, these should be equal
        assert get_local_rank() == get_global_rank()

    @staticmethod
    def _destroy_process_group() -> None:
        dist.init_process_group("gloo")
        destroy_process_group()
        assert not torch.distributed.is_initialized()

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_destroy_process_group(self) -> None:
        # should be a no-op if dist is not initialized
        destroy_process_group()
        config = get_pet_launch_config(2)
        launcher.elastic_launch(config, entrypoint=self._destroy_process_group)()

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_gather_uneven(self, world_size: Optional[int] = 4) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(
            config, entrypoint=self._test_ddp_gather_uneven_tensors
        )()

    @staticmethod
    def _test_ddp_gather_uneven_tensors() -> None:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        tensor = torch.ones(rank)
        result = all_gather_tensors(tensor)
        assert len(result) == world_size
        for idx in range(world_size):
            assert len(result[idx]) == idx
            assert (result[idx] == torch.ones_like(result[idx])).all()

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_gather_uneven_multidim(self) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(
            config, entrypoint=self._test_ddp_gather_uneven_tensors_multidim
        )()

    @staticmethod
    def _test_ddp_gather_uneven_tensors_multidim() -> None:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        tensor = torch.ones(rank + 1, 4 - rank)
        result = all_gather_tensors(tensor)
        assert len(result) == world_size
        for idx in range(world_size):
            val = result[idx]
            assert val.shape == (idx + 1, 4 - idx)
            assert (val == torch.ones_like(val)).all()

    @unittest.skipUnless(
        condition=torch.cuda.is_available(),
        reason="This test should only run on a GPU host.",
    )
    def test_gather_uneven_multidim_nccl(self) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(
            config, entrypoint=self._test_ddp_gather_uneven_tensors_multidim_nccl
        )()

    @staticmethod
    def _test_ddp_gather_uneven_tensors_multidim_nccl() -> None:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        tensor = torch.ones(rank + 1, 4 - rank, device=get_device_from_env())
        result = all_gather_tensors(tensor)
        assert len(result) == world_size
        for idx in range(world_size):
            val = result[idx]
            assert val.shape == (idx + 1, 4 - idx)
            assert (val == 1).all()

    def test_rank_zero_fn_rank_zero(self) -> None:
        @rank_zero_fn
        def foo() -> int:
            return 1

        x = foo()
        assert x == 1

    @patch("torchtnt.utils.distributed.get_global_rank")
    def test_rank_zero_fn_rank_non_zero(self, get_global_rank) -> None:
        get_global_rank.return_value = 1

        @rank_zero_fn
        def foo() -> int:
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

    @classmethod
    def _full_sync_worker(cls, coherence_mode: Optional[str]) -> bool:
        dist.init_process_group("gloo")
        if dist.get_rank() == 0:
            val = True
        else:
            val = False
        return sync_bool(val, coherence_mode=coherence_mode)

    def test_sync_bool_single_process(self) -> None:
        val = True
        new_val = sync_bool(val)
        # these should be the same in a single process case
        self.assertEqual(val, new_val)

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_sync_bool_multi_process_coherence_mode_rank_zero(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(
            "rank_zero"
        )
        # Both processes should return True since rank 0 inputs True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_sync_bool_multi_process_coherence_mode_any(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(
            "any"
        )
        # Both processes should return True since one of the processes inputs True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_sync_bool_multi_process_coherence_mode_all(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(
            "all"
        )
        # Both processes should return False since not all processes input False
        self.assertFalse(result[0])
        self.assertFalse(result[1])

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_sync_bool_multi_process_coherence_mode_int_false(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(2)
        # Both processes should return False since 2 processes don't input True
        self.assertFalse(result[0])
        self.assertFalse(result[1])

    def test_sync_bool_multi_process_coherence_mode_int_true(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(1)
        # Both processes should return True since 1 processes inputs True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_sync_bool_multi_process_coherence_mode_float_true(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(0.4)
        # Both processes should return True since 40% or one of the process inputs True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    def test_sync_bool_multi_process_coherence_mode_float_false(self) -> None:
        config = get_pet_launch_config(2)
        result = launcher.elastic_launch(config, entrypoint=self._full_sync_worker)(1.0)
        # Both processes should return False since 100% of processes don't input True
        self.assertFalse(result[0])
        self.assertFalse(result[1])
