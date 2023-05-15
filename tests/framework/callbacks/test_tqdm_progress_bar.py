#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

from torchtnt.framework._test_utils import (
    DummyEvalUnit,
    DummyPredictUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callbacks.tqdm_progress_bar import (
    _estimated_steps_in_epoch,
    TQDMProgressBar,
)
from torchtnt.framework.state import EntryPoint, PhaseState, State
from torchtnt.framework.train import init_train_state, train


class TQDMProgressBarTest(unittest.TestCase):
    def test_progress_bar_train(self) -> None:
        """
        Test TQDMProgressBar callback with train entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1
        expected_total = dataset_len / batch_size

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = State(
            entry_point=EntryPoint.TRAIN,
            train_state=PhaseState(
                dataloader=dataloader,
                max_epochs=max_epochs,
            ),
        )

        my_unit = MagicMock(spec=DummyTrainUnit)
        progress_bar = TQDMProgressBar()
        progress_bar.on_train_epoch_start(state, my_unit)
        self.assertEqual(progress_bar._train_progress_bar.total, expected_total)

    def test_progress_bar_train_integration(self) -> None:
        """
        Test TQDMProgressBar callback with train entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=dataloader, max_epochs=max_epochs)

        my_unit = MagicMock(spec=DummyTrainUnit)
        progress_bar = TQDMProgressBar()
        train(state, my_unit, callbacks=[progress_bar])

    def test_progress_bar_evaluate(self) -> None:
        """
        Test TQDMProgressBar callback with evaluate entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1
        expected_total = dataset_len / batch_size

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = State(
            entry_point=EntryPoint.EVALUATE,
            eval_state=PhaseState(
                dataloader=dataloader,
                max_epochs=max_epochs,
            ),
        )

        my_unit = MagicMock(spec=DummyEvalUnit)
        progress_bar = TQDMProgressBar()
        progress_bar.on_eval_epoch_start(state, my_unit)
        self.assertEqual(progress_bar._eval_progress_bar.total, expected_total)

    def test_progress_bar_predict(self) -> None:
        """
        Test TQDMProgressBar callback with predict entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1
        expected_total = dataset_len / batch_size

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = State(
            entry_point=EntryPoint.PREDICT,
            predict_state=PhaseState(
                dataloader=dataloader,
                max_epochs=max_epochs,
            ),
        )

        my_unit = MagicMock(spec=DummyPredictUnit)
        progress_bar = TQDMProgressBar()
        progress_bar.on_predict_epoch_start(state, my_unit)
        self.assertEqual(progress_bar._predict_progress_bar.total, expected_total)

    def test_progress_bar_mid_progress(self) -> None:
        """
        Test TQDMProgressBar callback when progress already has occurred (can occur when loading checkpoint)
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1
        expected_total = dataset_len / batch_size

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = State(
            entry_point=EntryPoint.PREDICT,
            predict_state=PhaseState(
                dataloader=dataloader,
                max_epochs=max_epochs,
            ),
        )
        state.predict_state.progress._num_steps_completed = 2

        my_unit = MagicMock(spec=DummyPredictUnit)
        progress_bar = TQDMProgressBar()
        progress_bar.on_predict_epoch_start(state, my_unit)
        self.assertEqual(progress_bar._predict_progress_bar.total, expected_total)
        self.assertEqual(progress_bar._predict_progress_bar.n, 2)

    def test_estimated_steps_in_epoch(self) -> None:
        """
        Test TQDMProgressBar's _estimate_steps_in_epoch function
        """

        input_dim = 2
        dataset_len = 20
        batch_size = 2
        dataloader_size = dataset_len / batch_size

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        self.assertEqual(
            _estimated_steps_in_epoch(
                dataloader, num_steps_completed=0, max_steps=5, max_steps_per_epoch=5
            ),
            5,
        )
        self.assertEqual(
            _estimated_steps_in_epoch(
                dataloader, num_steps_completed=4, max_steps=5, max_steps_per_epoch=4
            ),
            1,
        )
        self.assertEqual(
            _estimated_steps_in_epoch(
                dataloader, num_steps_completed=0, max_steps=4, max_steps_per_epoch=10
            ),
            4,
        )
        self.assertEqual(
            _estimated_steps_in_epoch(
                dataloader,
                num_steps_completed=0,
                max_steps=None,
                max_steps_per_epoch=None,
            ),
            dataloader_size,
        )
        self.assertEqual(
            _estimated_steps_in_epoch(
                dataloader,
                num_steps_completed=0,
                max_steps=None,
                max_steps_per_epoch=500,
            ),
            dataloader_size,
        )
