#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
import tempfile
import unittest
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import distributed as dist, multiprocessing as mp

from torchtnt.loggers.tensorboard import TensorBoardLogger


class TensorBoardLoggerTest(unittest.TestCase):
    def setUp(self) -> None:
        sock = socket.socket()
        sock.bind(("localhost", 0))
        self._free_port = str(sock.getsockname()[1])

    def test_log(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = TensorBoardLogger(path=log_dir)
            for i in range(5):
                logger.log("test_log", float(i) ** 2, i)
            logger.close()

            acc = EventAccumulator(log_dir)
            acc.Reload()
            for i, event in enumerate(acc.Tensors("test_log")):
                self.assertAlmostEquals(event.tensor_proto.float_val[0], float(i) ** 2)
                self.assertEquals(event.step, i)

    def test_log_dict(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = TensorBoardLogger(path=log_dir)
            metric_dict = {f"log_dict_{i}": float(i) ** 2 for i in range(5)}
            logger.log_dict(metric_dict, 1)
            logger.close()

            acc = EventAccumulator(log_dir)
            acc.Reload()
            for i in range(5):
                tensor_tag = acc.Tensors(f"log_dict_{i}")[0]
                self.assertAlmostEquals(
                    tensor_tag.tensor_proto.float_val[0], float(i) ** 2
                )
                self.assertEquals(tensor_tag.step, 1)

    def test_log_rank_zero(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            with patch.dict("os.environ", {"RANK": "1"}):
                logger = TensorBoardLogger(path=log_dir)
                self.assertEquals(logger.writer, None)

    @staticmethod
    def _setup_worker(rank: int, world_size: int, free_port: int) -> None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = free_port

        dist.init_process_group(
            "gloo" if not torch.cuda.is_available() else "nccl",
            rank=rank,
            world_size=world_size,
        )

    @staticmethod
    def _test_distributed(rank: int, world_size: int, free_port: int) -> None:
        TensorBoardLoggerTest._setup_worker(rank, world_size, free_port)

        with tempfile.TemporaryDirectory() as log_dir:
            test_path = "correct"
            invalid_path = "invalid"
            if rank == 0:
                logger = TensorBoardLogger(os.path.join(log_dir, test_path))
            else:
                logger = TensorBoardLogger(os.path.join(log_dir, invalid_path))

            assert test_path in logger.path
            assert invalid_path not in logger.path

    @unittest.skipUnless(
        dist.is_available(), reason="Torch distributed is needed to run"
    )
    def test_multiple_workers(self) -> None:
        world_size = 4
        mp.spawn(
            self._test_distributed,
            args=(world_size, self._free_port),
            nprocs=world_size,
        )
