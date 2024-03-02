#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock

import torch
from torchtnt.framework._test_utils import (
    DummyEvalUnit,
    DummyPredictUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callbacks.pytorch_profiler import PyTorchProfiler
from torchtnt.framework.evaluate import evaluate
from torchtnt.framework.predict import predict
from torchtnt.framework.train import train


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

        my_unit = DummyTrainUnit(input_dim)
        profiler_mock = MagicMock(spec=torch.profiler.profile)

        profiler = PyTorchProfiler(profiler=profiler_mock)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[profiler])
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

        my_unit = DummyEvalUnit(2)
        profiler_mock = MagicMock(spec=torch.profiler.profile)

        profiler = PyTorchProfiler(profiler=profiler_mock)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        evaluate(my_unit, dataloader, callbacks=[profiler])
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

        my_unit = DummyPredictUnit(2)
        profiler_mock = MagicMock(spec=torch.profiler.profile)

        profiler = PyTorchProfiler(profiler=profiler_mock)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        predict(my_unit, dataloader, callbacks=[profiler])
        self.assertEqual(profiler_mock.start.call_count, 1)
        self.assertEqual(profiler_mock.step.call_count, expected_num_total_steps)
        self.assertEqual(profiler_mock.stop.call_count, 1)
