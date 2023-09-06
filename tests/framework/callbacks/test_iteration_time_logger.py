#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

from torch.utils.tensorboard import SummaryWriter
from torchtnt.framework._test_utils import DummyTrainUnit, generate_random_dataloader
from torchtnt.framework.callbacks.iteration_time_logger import IterationTimeLogger

from torchtnt.framework.state import State
from torchtnt.framework.train import train
from torchtnt.utils.loggers import TensorBoardLogger


class IterationTimeLoggerTest(unittest.TestCase):
    def test_iteration_time_logger_test_on_train_step_end(self) -> None:
        logger = MagicMock(spec=TensorBoardLogger)
        logger.writer = MagicMock(spec=SummaryWriter)
        state = MagicMock(spec=State)
        state.train_state.iteration_timer.recorded_durations = {
            "train_iteration_time": [1, 3, 5, 7, 9]
        }

        my_unit = DummyTrainUnit(input_dim=2)
        my_unit.train_progress.increment_step()
        my_unit.train_progress.increment_step()
        callback = IterationTimeLogger(logger=logger, moving_avg_window=4)
        callback.on_train_step_end(state, my_unit)

        logger.writer.add_scalar.assert_called_with(
            "Train Iteration Time (seconds)",
            6,  # the average of the last 4 numbers is 6
            2,  # after incrementing twice, step should be 2
        )

    def test_with_train_epoch(self) -> None:
        """
        Test IterationTimeLogger callback with train entry point
        """

        my_unit = DummyTrainUnit(input_dim=2)
        logger = MagicMock(spec=TensorBoardLogger)
        logger.writer = MagicMock(spec=SummaryWriter)
        callback = IterationTimeLogger(logger, moving_avg_window=1, log_every_n_steps=3)
        dataloader = generate_random_dataloader(
            num_samples=12, input_dim=2, batch_size=2
        )
        train(my_unit, dataloader, max_epochs=2, callbacks=[callback])
        # 2 epochs, 6 iterations each, logging every third step
        self.assertEqual(logger.writer.add_scalar.call_count, 4)
