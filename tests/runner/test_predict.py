#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Iterator, Tuple

import torch
from torch import nn

from torchtnt.runner._test_utils import DummyPredictUnit, generate_random_dataloader

from torchtnt.runner.predict import predict
from torchtnt.runner.state import State
from torchtnt.runner.unit import PredictUnit


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

        state = predict(my_unit, dataloader)

        self.assertEqual(state.predict_state.progress.num_epochs_completed, 1)
        self.assertEqual(state.predict_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.predict_state.progress.num_steps_completed, expected_steps
        )

        # step_output should be reset to None
        self.assertEqual(state.predict_state.step_output, None)

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

        state = predict(my_unit, dataloader, max_steps_per_epoch=max_steps_per_epoch)

        self.assertEqual(state.predict_state.progress.num_epochs_completed, 1)
        self.assertEqual(state.predict_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.predict_state.progress.num_steps_completed, max_steps_per_epoch
        )

        # step_output should be reset to None
        self.assertEqual(state.predict_state.step_output, None)

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
        state = predict(my_unit, dataloader, max_steps_per_epoch=max_steps_per_epoch)

        self.assertEqual(state.predict_state.progress.num_epochs_completed, 1)
        self.assertEqual(state.predict_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.steps_processed, state.predict_state.progress.num_steps_completed
        )
        self.assertEqual(my_unit.steps_processed, steps_before_stopping)


    def test_predict_data_iter_step(self) -> None:
        class PredictIteratorUnit(PredictUnit[Iterator[Tuple[torch.Tensor, torch.Tensor]]]):
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

        state = predict(my_unit, dataloader)

        self.assertEqual(state.predict_state.progress.num_epochs_completed, 1)
        self.assertEqual(state.predict_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(state.predict_state.progress.num_steps_completed, expected_steps)

        # step_output should be reset to None
        self.assertEqual(state.predict_state.step_output, None)

        self.assertEqual(my_unit.module.training, initial_training_mode)

class StopPredictUnit(PredictUnit[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, input_dim: int, steps_before_stopping: int) -> None:
        super().__init__()
        # initialize module
        self.module = nn.Linear(input_dim, 2)
        self.steps_processed = 0
        self.steps_before_stopping = steps_before_stopping

    def predict_step(
        self, state: State, data: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        inputs, targets = data

        outputs = self.module(inputs)
        assert state.predict_state
        if (
            state.predict_state.progress.num_steps_completed_in_epoch + 1
            == self.steps_before_stopping
        ):
            state.stop()

        self.steps_processed += 1
        return outputs
