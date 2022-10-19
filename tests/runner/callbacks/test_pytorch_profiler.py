#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch
from torchtnt.runner._test_utils import (
    DummyEvalUnit,
    DummyPredictUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.runner.callbacks.pytorch_profiler import PyTorchProfiler
from torchtnt.runner.evaluate import evaluate, init_eval_state
from torchtnt.runner.predict import init_predict_state, predict
from torchtnt.runner.train import init_train_state, train


class PyTorchProfilerTest(unittest.TestCase):
    def test_profiler_train(self) -> None:
        """
        Test PytorchProfiler callback with train entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        expected_num_total_steps = dataset_len / batch_size * max_epochs

        my_unit = MagicMock(spec=DummyTrainUnit)
        profiler_mock = MagicMock(spec=torch.profiler.profile)

        profiler = PyTorchProfiler(profiler=profiler_mock)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=dataloader, max_epochs=max_epochs)
        train(state, my_unit, callbacks=[profiler])
        self.assertEqual(profiler_mock.start.call_count, 1)
        self.assertEqual(profiler_mock.step.call_count, expected_num_total_steps)
        self.assertEqual(profiler_mock.stop.call_count, 1)

    def test_profiler_evaluate(self) -> None:
        """
        Test PytorchProfiler callback with evaluate entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_num_total_steps = dataset_len / batch_size

        my_unit = MagicMock(spec=DummyEvalUnit)
        profiler_mock = MagicMock(spec=torch.profiler.profile)

        profiler = PyTorchProfiler(profiler=profiler_mock)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_eval_state(dataloader=dataloader)

        evaluate(state, my_unit, callbacks=[profiler])
        self.assertEqual(profiler_mock.start.call_count, 1)
        self.assertEqual(profiler_mock.step.call_count, expected_num_total_steps)
        self.assertEqual(profiler_mock.stop.call_count, 1)

    def test_profiler_predict(self) -> None:
        """
        Test PytorchProfiler callback with predict entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_num_total_steps = dataset_len / batch_size

        my_unit = MagicMock(spec=DummyPredictUnit)
        profiler_mock = MagicMock(spec=torch.profiler.profile)

        profiler = PyTorchProfiler(profiler=profiler_mock)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_predict_state(dataloader=dataloader)

        predict(state, my_unit, callbacks=[profiler])
        self.assertEqual(profiler_mock.start.call_count, 1)
        self.assertEqual(profiler_mock.step.call_count, expected_num_total_steps)
        self.assertEqual(profiler_mock.stop.call_count, 1)
