#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtnt.runner.fit import fit

from torchtnt.tests.runner.utils import (
    DummyEvalUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)


class FitTest(unittest.TestCase):
    def test_fit_evaluate_every_n_epochs(self) -> None:
        """
        Test fit entry point with evaluate_every_n_epochs=1
        """
        input_dim = 2
        train_dataset_len = 8
        eval_dataset_len = 4
        batch_size = 2
        max_epochs = 3
        evaluate_every_n_epochs = 1
        expected_train_steps_per_epoch = train_dataset_len / batch_size
        expected_eval_steps_per_epoch = eval_dataset_len / batch_size
        expected_num_evaluate_calls = max_epochs / evaluate_every_n_epochs

        my_train_unit = DummyTrainUnit(input_dim=input_dim)
        my_eval_unit = DummyEvalUnit(input_dim=input_dim)

        train_dataloader = generate_random_dataloader(
            train_dataset_len, input_dim, batch_size
        )
        eval_dataloader = generate_random_dataloader(
            eval_dataset_len, input_dim, batch_size
        )

        state = fit(
            my_train_unit,
            my_eval_unit,
            train_dataloader,
            eval_dataloader,
            max_epochs=max_epochs,
            evaluate_every_n_epochs=evaluate_every_n_epochs,
        )

        self.assertEqual(state.train_state.progress.num_epochs_completed, max_epochs)
        self.assertEqual(state.train_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.train_state.progress.num_steps_completed,
            max_epochs * expected_train_steps_per_epoch,
        )

        self.assertEqual(
            state.eval_state.progress.num_epochs_completed,
            expected_num_evaluate_calls,
        )
        self.assertEqual(state.eval_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.eval_state.progress.num_steps_completed,
            max_epochs * expected_eval_steps_per_epoch,
        )

        # step_output should be reset to None
        self.assertEqual(state.eval_state.step_output, None)
        self.assertEqual(state.train_state.step_output, None)

    def test_fit_evaluate_every_n_steps(self) -> None:
        """
        Test fit entry point with evaluate_every_n_steps=2
        """
        input_dim = 2
        train_dataset_len = 16
        eval_dataset_len = 4
        batch_size = 2
        max_epochs = 3
        evaluate_every_n_steps = 2
        expected_train_steps_per_epoch = train_dataset_len / batch_size
        expected_eval_steps_per_epoch = eval_dataset_len / batch_size
        expected_num_evaluate_calls_per_train_epoch = (
            expected_train_steps_per_epoch / evaluate_every_n_steps
        )
        expected_num_evaluate_calls = (
            expected_num_evaluate_calls_per_train_epoch * max_epochs
        )

        my_train_unit = DummyTrainUnit(input_dim=input_dim)
        my_eval_unit = DummyEvalUnit(input_dim=input_dim)

        train_dataloader = generate_random_dataloader(
            train_dataset_len, input_dim, batch_size
        )
        eval_dataloader = generate_random_dataloader(
            eval_dataset_len, input_dim, batch_size
        )

        state = fit(
            my_train_unit,
            my_eval_unit,
            train_dataloader,
            eval_dataloader,
            max_epochs=max_epochs,
            evaluate_every_n_epochs=None,
            evaluate_every_n_steps=evaluate_every_n_steps,
        )

        self.assertEqual(state.train_state.progress.num_epochs_completed, max_epochs)
        self.assertEqual(state.train_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.train_state.progress.num_steps_completed,
            max_epochs * expected_train_steps_per_epoch,
        )

        self.assertEqual(
            state.eval_state.progress.num_epochs_completed,
            expected_num_evaluate_calls,
        )
        self.assertEqual(state.eval_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.eval_state.progress.num_steps_completed,
            expected_num_evaluate_calls * expected_eval_steps_per_epoch,
        )

        # step_output should be reset to None
        self.assertEqual(state.eval_state.step_output, None)
        self.assertEqual(state.train_state.step_output, None)
