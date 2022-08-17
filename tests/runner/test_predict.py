#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtnt.runner._test_utils import DummyPredictUnit, generate_random_dataloader

from torchtnt.runner.predict import predict


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
