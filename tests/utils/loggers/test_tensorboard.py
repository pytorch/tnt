#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torchtnt.utils.anomaly_evaluation import ThresholdEvaluator
from torchtnt.utils.loggers.anomaly_logger import TrackedMetric

from torchtnt.utils.loggers.tensorboard import TensorBoardLogger


class TensorBoardLoggerTest(unittest.TestCase):

    @patch(
        "torchtnt.utils.loggers.anomaly_logger.AnomalyLogger.on_anomaly_detected",
    )
    def test_log(
        self: TensorBoardLoggerTest, mock_on_anomaly_detected: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = TensorBoardLogger(
                path=log_dir,
                tracked_metrics=[
                    TrackedMetric(
                        name="test_log",
                        anomaly_evaluators=[
                            ThresholdEvaluator(min_val=25),
                        ],
                        evaluate_every_n_steps=2,
                        warmup_steps=2,
                    )
                ],
            )
            warning_container = []
            with patch(
                "torchtnt.utils.loggers.anomaly_logger.logging.Logger.warning",
                side_effect=warning_container.append,
            ):
                for i in range(5):
                    logger.log("test_log", float(i) ** 2, i)
                logger.close()

            self.assertEqual(
                warning_container,
                [
                    "Found anomaly in metric: test_log, with value: 16.0, using evaluator: ThresholdEvaluator"
                ],
            )
            mock_on_anomaly_detected.assert_called_with("test_log", 16.0, 4)

            acc = EventAccumulator(log_dir)
            acc.Reload()
            for i, event in enumerate(acc.Tensors("test_log")):
                self.assertAlmostEqual(event.tensor_proto.float_val[0], float(i) ** 2)
                self.assertEqual(event.step, i)

    def test_log_dict(self: TensorBoardLoggerTest) -> None:
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

    def test_log_text(self: TensorBoardLoggerTest) -> None:
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

    def test_log_rank_zero(self: TensorBoardLoggerTest) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            with patch.dict("os.environ", {"RANK": "1"}):
                logger = TensorBoardLogger(path=log_dir)
                self.assertEqual(logger.writer, None)

    def test_add_scalars_call_is_correctly_passed_to_summary_writer(
        self: TensorBoardLoggerTest,
    ) -> None:
        with patch(
            "torchtnt.utils.loggers.tensorboard.SummaryWriter"
        ) as mock_summary_writer_class:
            mock_summary_writer = Mock()
            mock_summary_writer_class.return_value = mock_summary_writer
            logger = TensorBoardLogger(path="/tmp")
            logger.log_scalars(
                "tnt_metrics",
                {
                    "x": 0,
                    "y": 1,
                },
                1,
                2,
            )
            mock_summary_writer.add_scalars.assert_called_with(
                main_tag="tnt_metrics",
                tag_scalar_dict={
                    "x": 0,
                    "y": 1,
                },
                global_step=1,
                walltime=2,
            )
