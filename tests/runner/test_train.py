#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtnt.runner.train import train, train_epoch

from torchtnt.tests.runner.utils import DummyTrainUnit, generate_random_dataloader


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

        state = train(my_unit, dataloader, max_epochs=max_epochs)

        self.assertEqual(state.train_state.progress.num_epochs_completed, max_epochs)
        self.assertEqual(state.train_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.train_state.progress.num_steps_completed,
            max_epochs * expected_steps_per_epoch,
        )

        # step_output should be reset to None
        self.assertEqual(state.train_state.step_output, None)

        self.assertEqual(my_unit.module.training, initial_training_mode)

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

        state = train(
            my_unit,
            dataloader,
            max_epochs=max_epochs,
            max_steps_per_epoch=max_steps_per_epoch,
        )

        self.assertEqual(state.train_state.progress.num_epochs_completed, max_epochs)
        self.assertEqual(state.train_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.train_state.progress.num_steps_completed,
            max_epochs * max_steps_per_epoch,
        )

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

        state = train_epoch(my_unit, dataloader)

        self.assertEqual(state.train_state.progress.num_epochs_completed, 1)
        self.assertEqual(state.train_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.train_state.progress.num_steps_completed,
            expected_steps_per_epoch,
        )

        # step_output should be reset to None
        self.assertEqual(state.train_state.step_output, None)

        self.assertEqual(my_unit.module.training, initial_training_mode)
