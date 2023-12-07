#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
from torchtnt.utils.device import get_device_from_env
from torchtnt.utils.distributed import all_gather_tensors, get_local_rank, PGWrapper
from torchtnt.utils.env import init_from_env
from torchtnt.utils.test_utils import spawn_multi_process


class DistributedGPUTest(unittest.TestCase):
    dist_available: bool = torch.distributed.is_available()
    cuda_available: bool = torch.cuda.is_available()

    @unittest.skipUnless(
        condition=cuda_available,
        reason="This test should only run on a GPU host.",
    )
    @unittest.skipUnless(dist_available, reason="Torch distributed is needed to run")
    def test_gather_uneven_multidim_nccl(self) -> None:
        spawn_multi_process(
            2,
            "nccl",
            self._test_ddp_gather_uneven_tensors_multidim_nccl,
        )

    @staticmethod
    def _test_ddp_gather_uneven_tensors_multidim_nccl() -> None:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        tensor = torch.ones(rank + 1, 4 - rank, device=get_device_from_env())
        result = all_gather_tensors(tensor)
        assert len(result) == world_size
        for idx in range(world_size):
            val = result[idx]
            assert val.shape == (idx + 1, 4 - idx)
            assert (val == 1).all()

    def test_pg_wrapper_scatter_object_list_nccl(self) -> None:
        spawn_multi_process(
            2,
            "nccl",
            self._test_pg_wrapper_scatter_object_list,
        )

    @classmethod
    def _test_pg_wrapper_scatter_object_list(
        cls,
    ) -> None:
        init_from_env()
        pg_wrapper = PGWrapper(dist.group.WORLD)
        output_list = [None] * 2
        pg_wrapper.scatter_object_list(
            output_list=output_list,
            input_list=[1, 2] if get_local_rank() == 0 else [None] * 2,
            src=0,
        )
        tc = unittest.TestCase()
        tc.assertEqual(output_list[0], get_local_rank() + 1)
