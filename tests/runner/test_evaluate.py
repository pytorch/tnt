#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple
from unittest.mock import MagicMock

import torch
from torch import nn
from torchtnt.runner._test_utils import DummyEvalUnit, generate_random_dataloader
from torchtnt.runner.evaluate import evaluate
from torchtnt.runner.state import State
from torchtnt.runner.unit import EvalUnit


class EvaluateTest(unittest.TestCase):
    def test_evaluate(self) -> None:
        """
        Test evaluate entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_steps = dataset_len / batch_size

        my_unit = DummyEvalUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        state = evaluate(my_unit, dataloader)

        self.assertEqual(state.eval_state.progress.num_epochs_completed, 1)
        self.assertEqual(state.eval_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(state.eval_state.progress.num_steps_completed, expected_steps)

        # step_output should be reset to None
        self.assertEqual(state.eval_state.step_output, None)

        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_evaluate_max_steps_per_epoch(self) -> None:
        """
        Test evaluate entry point with max_steps_per_epoch
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 4

        my_unit = DummyEvalUnit(input_dim=input_dim)
        initial_training_mode = my_unit.module.training

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        state = evaluate(my_unit, dataloader, max_steps_per_epoch=max_steps_per_epoch)

        self.assertEqual(state.eval_state.progress.num_epochs_completed, 1)
        self.assertEqual(state.eval_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            state.eval_state.progress.num_steps_completed, max_steps_per_epoch
        )

        # step_output should be reset to None
        self.assertEqual(state.eval_state.step_output, None)

        self.assertEqual(my_unit.module.training, initial_training_mode)

    def test_evaluate_stop(self) -> None:
        """
        Test evaluate entry point with setting state's `stop()` flag
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 4
        steps_before_stopping = 2

        my_unit = StopEvalUnit(
            input_dim=input_dim, steps_before_stopping=steps_before_stopping
        )
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = evaluate(my_unit, dataloader, max_steps_per_epoch=max_steps_per_epoch)

        self.assertEqual(state.eval_state.progress.num_epochs_completed, 1)
        self.assertEqual(state.eval_state.progress.num_steps_completed_in_epoch, 0)
        self.assertEqual(
            my_unit.steps_processed, state.eval_state.progress.num_steps_completed
        )
        self.assertEqual(my_unit.steps_processed, steps_before_stopping)

    def test_evaluate_with_callback(self) -> None:
        """
        Test evaluate entry point with a callback
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_steps_per_epoch = 6
        expected_num_steps = dataset_len / batch_size

        my_unit = MagicMock()
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        callback_mock = MagicMock()
        _ = evaluate(
            my_unit,
            dataloader,
            [callback_mock],
            max_steps_per_epoch=max_steps_per_epoch,
        )
        self.assertEqual(callback_mock.on_eval_start.call_count, 1)
        self.assertEqual(callback_mock.on_eval_epoch_start.call_count, 1)
        self.assertEqual(
            callback_mock.on_eval_step_start.call_count, expected_num_steps
        )
        self.assertEqual(callback_mock.on_eval_step_end.call_count, expected_num_steps)
        self.assertEqual(callback_mock.on_eval_epoch_end.call_count, 1)
        self.assertEqual(callback_mock.on_eval_end.call_count, 1)


class StopEvalUnit(EvalUnit[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, input_dim: int, steps_before_stopping: int) -> None:
        super().__init__()
        # initialize module & loss_fn
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.steps_processed = 0
        self.steps_before_stopping = steps_before_stopping

    def eval_step(
        self, state: State, data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)

        assert state.eval_state
        if (
            state.eval_state.progress.num_steps_completed_in_epoch + 1
            == self.steps_before_stopping
        ):
            state.stop()

        self.steps_processed += 1
        return loss, outputs
