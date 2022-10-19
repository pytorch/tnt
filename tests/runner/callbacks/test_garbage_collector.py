#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import unittest
from unittest.mock import MagicMock

from torchtnt.runner._test_utils import (
    DummyEvalUnit,
    DummyPredictUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.runner.callbacks.garbage_collector import GarbageCollector
from torchtnt.runner.evaluate import evaluate, init_eval_state
from torchtnt.runner.predict import predict
from torchtnt.runner.train import init_train_state, train


class GarbageCollectorTest(unittest.TestCase):
    def test_garbage_collector_call_count_train(self) -> None:
        """
        Test GarbageCollector callback was called correct number of times (with train entry point)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        expected_num_total_steps = dataset_len / batch_size * max_epochs

        my_unit = MagicMock(spec=DummyTrainUnit)
        gc_callback_mock = MagicMock(spec=GarbageCollector)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=dataloader, max_epochs=max_epochs)

        train(state, my_unit, callbacks=[gc_callback_mock])
        self.assertEqual(gc_callback_mock.on_train_start.call_count, 1)
        self.assertEqual(
            gc_callback_mock.on_train_step_end.call_count, expected_num_total_steps
        )
        self.assertEqual(gc_callback_mock.on_train_end.call_count, 1)

    def test_garbage_collector_enabled_train(self) -> None:
        """
        Test garbage collection is enabled after runs are finished (with train entry point)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2

        my_unit = MagicMock(spec=DummyTrainUnit)
        gc_callback = GarbageCollector(2)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=dataloader, max_epochs=max_epochs)

        self.assertTrue(gc.isenabled())
        train(state, my_unit, callbacks=[gc_callback])
        self.assertTrue(gc.isenabled())

    def test_garbage_collector_call_count_evaluate(self) -> None:
        """
        Test GarbageCollector callback was called correct number of times (with evaluate entry point)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_num_total_steps = dataset_len / batch_size

        my_unit = MagicMock(spec=DummyEvalUnit)
        gc_callback_mock = MagicMock(spec=GarbageCollector)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_eval_state(dataloader=dataloader)

        evaluate(state, my_unit, callbacks=[gc_callback_mock])
        self.assertEqual(gc_callback_mock.on_eval_start.call_count, 1)
        self.assertEqual(
            gc_callback_mock.on_eval_step_end.call_count, expected_num_total_steps
        )
        self.assertEqual(gc_callback_mock.on_eval_end.call_count, 1)

    def test_garbage_collector_enabled_evaluate(self):
        """
        Test garbage collection is enabled after runs are finished (with evaluate entry point)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2

        my_unit = MagicMock(spec=DummyEvalUnit)
        gc_callback = GarbageCollector(2)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_eval_state(dataloader=dataloader)

        self.assertTrue(gc.isenabled())
        evaluate(state, my_unit, callbacks=[gc_callback])
        self.assertTrue(gc.isenabled())

    def test_garbage_collector_call_count_predict(self) -> None:
        """
        Test GarbageCollector callback was called correct number of times (with predict entry point)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        expected_num_total_steps = dataset_len / batch_size

        my_unit = MagicMock(spec=DummyPredictUnit)
        gc_callback_mock = MagicMock(spec=GarbageCollector)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        predict(my_unit, dataloader, callbacks=[gc_callback_mock])
        self.assertEqual(gc_callback_mock.on_predict_start.call_count, 1)
        self.assertEqual(
            gc_callback_mock.on_predict_step_end.call_count, expected_num_total_steps
        )
        self.assertEqual(gc_callback_mock.on_predict_end.call_count, 1)

    def test_garbage_collector_enabled_predict(self):
        """
        Test garbage collection is enabled after runs are finished (with predict entry point)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2

        my_unit = MagicMock(spec=DummyPredictUnit)
        gc_callback = GarbageCollector(2)

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        self.assertTrue(gc.isenabled())
        predict(my_unit, dataloader, callbacks=[gc_callback])
        self.assertTrue(gc.isenabled())
