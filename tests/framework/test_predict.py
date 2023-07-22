#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Iterator, Tuple
from unittest.mock import MagicMock

import torch
from torch import nn

from torchtnt.framework._test_utils import DummyPredictUnit, generate_random_dataloader

from torchtnt.framework.predict import predict
from torchtnt.framework.state import State
from torchtnt.framework.unit import PredictUnit
from torchtnt.utils.timer import Timer


class PredictTest(unittest.TestCase):
    def test_predict(self) -> None:
        """
        Test predict entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_steps = dataset_len / batch_size

        my_unit = DummyPredictUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        predict(my_unit, dataloader)

        self.assertEqual(my_unit.predict_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.predict_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(my_unit.predict_progress.num_steps_completed, expected_steps)

        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_predict_max_steps_per_epoch(self) -> None:
        """
        Test predict entry point with max_steps_per_epoch
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 3

        my_unit = DummyPredictUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        predict(my_unit, dataloader, max_steps_per_epoch=max_steps_per_epoch)

        self.assertEqual(my_unit.predict_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.predict_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.predict_progress.num_steps_completed, max_steps_per_epoch
        )

        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_predict_stop(self) -> None:
        """
        Test predict entry point with setting state's `stop()` flag
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 4
        steps_before_stopping = 2

        my_unit = StopPredictUnit(
            input_dim=input_dim, steps_before_stopping=steps_before_stopping
        )
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        predict(my_unit, dataloader, max_steps_per_epoch=max_steps_per_epoch)

        self.assertEqual(my_unit.predict_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.predict_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.steps_processed, my_unit.predict_progress.num_steps_completed
        )
        self.assertEqual(my_unit.steps_processed, steps_before_stopping)

    def test_predict_with_callback(self) -> None:
        """
        Test predict entry point with a callback
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 6
        expected_num_steps = dataset_len / batch_size

        my_unit = DummyPredictUnit(2)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        callback_mock = MagicMock()
        predict(
            my_unit,
            dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
            callbacks=[callback_mock],
        )

        self.assertEqual(callback_mock.on_predict_start.call_count, 1)
        self.assertEqual(callback_mock.on_predict_epoch_start.call_count, 1)
        self.assertEqual(
            callback_mock.on_predict_step_start.call_count, expected_num_steps
        )
        self.assertEqual(
            callback_mock.on_predict_step_end.call_count, expected_num_steps
        )
        self.assertEqual(callback_mock.on_predict_epoch_end.call_count, 1)
        self.assertEqual(callback_mock.on_predict_end.call_count, 1)

    def test_predict_data_iter_step(self) -> None:
        class PredictIteratorUnit(
            PredictUnit[Iterator[Tuple[torch.Tensor, torch.Tensor]]]
        ):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                self.module = nn.Linear(input_dim, 2)
                self.loss_fn = nn.CrossEntropyLoss()

            def predict_step(
                self, state: State, data: Iterator[Tuple[torch.Tensor, torch.Tensor]]
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                batch = next(data)
                inputs, targets = batch

                outputs = self.module(inputs)
                loss = self.loss_fn(outputs, targets)
                return loss, outputs

        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_steps = dataset_len / batch_size

        my_unit = PredictIteratorUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        predict(my_unit, dataloader)

        self.assertEqual(my_unit.predict_progress.num_epochs_completed, 1)
        self.assertEqual(my_unit.predict_progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(my_unit.predict_progress.num_steps_completed, expected_steps)
        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_predict_timing(self) -> None:
        """
        Test timing in predict
        """

        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 2

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        timer = Timer()
        predict(
            DummyPredictUnit(input_dim=input_dim),
            dataloader,
            max_steps_per_epoch=max_steps_per_epoch,
            timer=timer,
        )
        self.assertIn("predict.next(data_iter)", timer.recorded_durations.keys())


Batch = Tuple[torch.Tensor, torch.Tensor]


class StopPredictUnit(PredictUnit[Batch]):
    def __init__(self, input_dim: int, steps_before_stopping: int) -> None:
        super().__init__()
        # initialize module
        self.module = nn.Linear(input_dim, 2)
        self.steps_processed = 0
        self.steps_before_stopping = steps_before_stopping

    def predict_step(self, state: State, data: Batch) -> torch.Tensor:
        inputs, targets = data

        outputs = self.module(inputs)
        assert state.predict_state
        if (
            self.predict_progress.num_steps_completed_in_epoch + 1
            == self.steps_before_stopping
        ):
            state.stop()

        self.steps_processed += 1
        return outputs
