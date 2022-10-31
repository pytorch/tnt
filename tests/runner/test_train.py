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

from torchtnt.runner._test_utils import DummyTrainUnit, generate_random_dataloader
from torchtnt.runner.state import EntryPoint, State
from torchtnt.runner.train import init_train_state, train, train_epoch
from torchtnt.runner.unit import TrainUnit


class TrainTest(unittest.TestCase):
    def test_train(self) -> None:
        """
        Test train entry point
        """
        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3
        expected_steps_per_epoch = dataset_len / batch_size

        my_unit = DummyTrainUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=dataloader, max_epochs=max_epochs)
        train(state, my_unit)

        self.assertEqual(state.train_state.progress.num_epochs_completed, max_epochs)
        self.assertEqual(state.train_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.train_state.progress.num_steps_completed,
            max_epochs * expected_steps_per_epoch,
        )

        # step_output should be reset to None
        self.assertEqual(state.train_state.step_output, None)
        # is_last_batch should be set to True
        self.assertEqual(state.train_state.is_last_batch, True)

        self.assertEqual(my_unit.module.training, initial_training_mode)
        self.assertEqual(state.entry_point, EntryPoint.TRAIN)

    def test_train_max_steps_per_epoch(self) -> None:
        """
        Test train entry point with max_steps_per_epoch
        """
        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3
        max_steps_per_epoch = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        state = init_train_state(
            dataloader=dataloader,
            max_epochs=max_epochs,
            max_steps_per_epoch=max_steps_per_epoch,
        )

        train(state, my_unit)

        self.assertEqual(state.train_state.progress.num_epochs_completed, max_epochs)
        self.assertEqual(state.train_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.train_state.progress.num_steps_completed,
            max_epochs * max_steps_per_epoch,
        )
        self.assertEqual(state.entry_point, EntryPoint.TRAIN)

        # step_output should be reset to None
        self.assertEqual(state.train_state.step_output, None)

        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_train_epoch(self) -> None:
        """
        Test train_epoch entry point
        """
        input_dim = 2
        dataset_len = 8
        batch_size = 2
        expected_steps_per_epoch = dataset_len / batch_size

        my_unit = DummyTrainUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        state = init_train_state(dataloader=dataloader, max_epochs=1)

        train_epoch(state, my_unit)

        self.assertEqual(state.train_state.progress.num_epochs_completed, 1)
        self.assertEqual(state.train_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.train_state.progress.num_steps_completed,
            expected_steps_per_epoch,
        )
        self.assertEqual(state.entry_point, EntryPoint.TRAIN)

        # step_output should be reset to None
        self.assertEqual(state.train_state.step_output, None)

        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_train_stop(self) -> None:
        """
        Test train entry point with setting state's `stop()` flag
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 3
        max_steps_per_epoch = 4
        steps_before_stopping = 2

        my_unit = StopTrainUnit(
            input_dim=input_dim, steps_before_stopping=steps_before_stopping
        )
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        state = init_train_state(
            dataloader=dataloader,
            max_epochs=max_epochs,
            max_steps_per_epoch=max_steps_per_epoch,
        )

        train(state, my_unit)

        self.assertEqual(state.train_state.progress.num_epochs_completed, 1)
        self.assertEqual(state.train_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.steps_processed, state.train_state.progress.num_steps_completed
        )
        self.assertEqual(my_unit.steps_processed, steps_before_stopping)

    def test_train_with_callback(self) -> None:
        """
        Test train entry point with a callback
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 6
        max_epochs = 3
        expected_num_total_steps = dataset_len / batch_size * max_epochs

        my_unit = MagicMock()
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(
            dataloader=dataloader,
            max_epochs=max_epochs,
            max_steps_per_epoch=max_steps_per_epoch,
        )

        callback_mock = MagicMock()
        train(state, my_unit, callbacks=[callback_mock])
        self.assertEqual(callback_mock.on_train_start.call_count, 1)
        self.assertEqual(callback_mock.on_train_epoch_start.call_count, max_epochs)
        self.assertEqual(
            callback_mock.on_train_step_start.call_count, expected_num_total_steps
        )
        self.assertEqual(
            callback_mock.on_train_step_end.call_count, expected_num_total_steps
        )
        self.assertEqual(callback_mock.on_train_epoch_end.call_count, max_epochs)
        self.assertEqual(callback_mock.on_train_end.call_count, 1)

    def test_train_data_iter_step(self) -> None:
        class TrainIteratorUnit(TrainUnit[Iterator[Tuple[torch.Tensor, torch.Tensor]]]):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                self.module = nn.Linear(input_dim, 2)
                self.loss_fn = nn.CrossEntropyLoss()

            def train_step(
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

        my_unit = TrainIteratorUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        state = init_train_state(
            dataloader=dataloader,
            max_epochs=1,
        )
        train(state, my_unit)

        self.assertEqual(state.train_state.progress.num_epochs_completed, 1)
        self.assertEqual(state.train_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(state.train_state.progress.num_steps_completed, expected_steps)

        # step_output should be reset to None
        self.assertEqual(state.train_state.step_output, None)

        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_train_max_steps(self) -> None:
        max_steps = 3
        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3
        expected_steps_per_epoch = dataset_len / batch_size

        my_unit = MagicMock()
        my_unit.modules = MagicMock(return_value={})
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        state = init_train_state(
            dataloader=dataloader, max_epochs=None, max_steps=max_steps
        )
        train(state, my_unit)

        self.assertEqual(state.train_state.progress.num_steps_completed, max_steps)
        self.assertEqual(my_unit.train_step.call_count, max_steps)

        # hit max epoch condition before max steps
        my_unit = MagicMock()
        my_unit.modules = MagicMock(return_value={})
        state = init_train_state(
            dataloader=dataloader, max_epochs=max_epochs, max_steps=100000
        )
        train(state, my_unit)
        self.assertEqual(state.train_state.progress.num_epochs_completed, max_epochs)
        self.assertEqual(state.train_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.train_state.progress.num_steps_completed,
            max_epochs * expected_steps_per_epoch,
        )
        self.assertEqual(
            my_unit.train_step.call_count, max_epochs * expected_steps_per_epoch
        )


class StopTrainUnit(TrainUnit[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, input_dim: int, steps_before_stopping: int) -> None:
        super().__init__()
        # initialize module, loss_fn, & optimizer
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)
        self.steps_processed = 0
        self.steps_before_stopping = steps_before_stopping

    def train_step(
        self, state: State, data: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        assert state.train_state
        if (
            state.train_state.progress.num_steps_completed_in_epoch + 1
            == self.steps_before_stopping
        ):
            state.stop()

        self.steps_processed += 1
        return loss, outputs
