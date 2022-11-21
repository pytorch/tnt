#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
from unittest.mock import patch

import torch.distributed.launcher as launcher
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import distributed as dist

from torchtnt.utils.loggers.tensorboard import TensorBoardLogger
from torchtnt.utils.test_utils import get_pet_launch_config


class TensorBoardLoggerTest(unittest.TestCase):
    def test_log(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = TensorBoardLogger(path=log_dir)
            for i in range(5):
                logger.log("test_log", float(i) ** 2, i)
            logger.close()

            acc = EventAccumulator(log_dir)
            acc.Reload()
            for i, event in enumerate(acc.Tensors("test_log")):
                self.assertAlmostEqual(event.tensor_proto.float_val[0], float(i) ** 2)
                self.assertEqual(event.step, i)

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
                self.assertAlmostEqual(
                    tensor_tag.tensor_proto.float_val[0], float(i) ** 2
                )
                self.assertEqual(tensor_tag.step, 1)

    def test_log_text(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = TensorBoardLogger(path=log_dir)
            for i in range(5):
                logger.log_text("test_text", f"iter:{i}", i)
            logger.close()

            acc = EventAccumulator(log_dir)
            acc.Reload()
            for i, test_text_event in enumerate(acc.Tensors("test_text/text_summary")):
                self.assertEqual(
                    test_text_event.tensor_proto.string_val[0].decode("ASCII"),
                    f"iter:{i}",
                )
                self.assertEqual(test_text_event.step, i)

    def test_log_rank_zero(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            with patch.dict("os.environ", {"RANK": "1"}):
                logger = TensorBoardLogger(path=log_dir)
                self.assertEqual(logger.writer, None)

    @staticmethod
    def _test_distributed() -> None:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
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
        config = get_pet_launch_config(2)
        launcher.elastic_launch(config, entrypoint=self._test_distributed)()
