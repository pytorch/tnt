# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchtnt.framework._test_utils import (
    Batch,
    get_dummy_eval_state,
    get_dummy_train_state,
)
from torchtnt.framework.callback import Callback
from torchtnt.framework.callbacks.empty_dataloader_detector import (
    EmptyDataloaderDetectorCallback,
)
from torchtnt.framework.state import State
from torchtnt.framework.train import train
from torchtnt.framework.unit import EvalUnit, PredictUnit, TrainUnit


class EmptyDataloaderDetectorCallbackTest(unittest.TestCase):
    def test_init_invalid_threshold(self) -> None:
        """Test that invalid threshold values raise ValueError."""
        with self.assertRaisesRegex(ValueError, "threshold must be a positive integer"):
            EmptyDataloaderDetectorCallback(threshold=0)

        with self.assertRaisesRegex(ValueError, "threshold must be a positive integer"):
            EmptyDataloaderDetectorCallback(threshold=-1)

    def test_train_empty_epoch_detection_with_exception(self) -> None:
        """Test that consecutive empty train epochs trigger exception when threshold is reached."""
        callback = EmptyDataloaderDetectorCallback(threshold=2, raise_exception=True)
        state = get_dummy_train_state()
        unit = MockTrainUnit()

        # First empty epoch - should not raise
        unit.set_steps_completed_in_prev_epoch(0)
        callback.on_train_epoch_end(state, unit)
        self.assertEqual(callback._consecutive_empty_train_epochs, 1)

        # Second empty epoch - should raise exception
        unit.set_steps_completed_in_prev_epoch(0)
        with self.assertRaisesRegex(
            RuntimeError,
            "Detected 2 consecutive empty train epochs, which exceeds the threshold of 2",
        ):
            callback.on_train_epoch_end(state, unit)

    def test_train_reset_counter_on_non_empty_epoch(self) -> None:
        """Test that consecutive empty epoch counter resets when a non-empty epoch occurs."""
        callback = EmptyDataloaderDetectorCallback(threshold=3, raise_exception=True)
        state = get_dummy_train_state()
        unit = MockTrainUnit()

        # First empty epoch
        unit.set_steps_completed_in_prev_epoch(0)
        callback.on_train_epoch_end(state, unit)
        self.assertEqual(callback._consecutive_empty_train_epochs, 1)

        # Second empty epoch
        unit.set_steps_completed_in_prev_epoch(0)
        callback.on_train_epoch_end(state, unit)
        self.assertEqual(callback._consecutive_empty_train_epochs, 2)

        # Non-empty epoch - should reset counter
        unit.set_steps_completed_in_prev_epoch(5)
        callback.on_train_epoch_end(state, unit)
        self.assertEqual(callback._consecutive_empty_train_epochs, 0)

        # Another empty epoch - counter should start from 1 again
        unit.set_steps_completed_in_prev_epoch(0)
        callback.on_train_epoch_end(state, unit)
        self.assertEqual(callback._consecutive_empty_train_epochs, 1)

    def test_different_phases_independent_counters(self) -> None:
        """Test that different phases maintain independent empty epoch counters."""
        callback = EmptyDataloaderDetectorCallback(threshold=3, raise_exception=True)
        train_state = get_dummy_train_state()
        eval_state = get_dummy_eval_state()
        train_unit = MockTrainUnit()
        eval_unit = MockEvalUnit()

        # Empty train epochs
        train_unit.set_steps_completed_in_prev_epoch(0)
        callback.on_train_epoch_end(train_state, train_unit)
        callback.on_train_epoch_end(train_state, train_unit)
        self.assertEqual(callback._consecutive_empty_train_epochs, 2)
        self.assertEqual(callback._consecutive_empty_eval_epochs, 0)

        # Empty eval epochs
        eval_unit.set_steps_completed_in_prev_epoch(0)
        callback.on_eval_epoch_end(eval_state, eval_unit)
        self.assertEqual(callback._consecutive_empty_train_epochs, 2)
        self.assertEqual(callback._consecutive_empty_eval_epochs, 1)

        # Non-empty train epoch should only reset train counter
        train_unit.set_steps_completed_in_prev_epoch(5)
        callback.on_train_epoch_end(train_state, train_unit)
        self.assertEqual(callback._consecutive_empty_train_epochs, 0)
        self.assertEqual(callback._consecutive_empty_eval_epochs, 1)

    def test_empty_dataloader_detection_with_real_training_loop(self) -> None:
        """
        Test that simulates the real scenario from failed MAST job f762746046-pviolatingquery_cse.
        Tests EmptyDataloaderDetectorCallback with actual training loop and empty dataloaders.
        """

        class EmptyDataset(Dataset[Batch]):
            """Dataset that returns no data to simulate empty dataloader scenario."""

            def __len__(self) -> int:
                return 0

            def __getitem__(self, idx: int) -> Batch:
                raise IndexError("Empty dataset")

        class SimpleTrainUnit(TrainUnit[Batch]):
            """Simple train unit for e2e testing."""

            def __init__(self) -> None:
                super().__init__()
                self.module = nn.Linear(2, 1)
                self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)

            def train_step(self, state: State, data: Batch) -> None:
                inputs, targets = data
                outputs = self.module(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        callback_with_exception = EmptyDataloaderDetectorCallback(
            threshold=2, raise_exception=True
        )

        train_unit = SimpleTrainUnit()
        empty_dataloader = DataLoader(EmptyDataset(), batch_size=1)

        # This should raise an exception after 2 empty epochs
        with self.assertRaisesRegex(
            RuntimeError,
            "Detected 2 consecutive empty train epochs, which exceeds the threshold of 2",
        ):
            train(
                train_unit,
                empty_dataloader,
                max_epochs=50,  # Try to run 50 epochs but should fail at 2
                callbacks=[callback_with_exception],
            )

        self.assertEqual(callback_with_exception._consecutive_empty_train_epochs, 1)

    def test_e2e_empty_dataloader_detection_with_warning_mode(self) -> None:
        """
        E2E test with warning mode - should complete training but log warnings.
        """

        class EmptyDataset(Dataset[Batch]):
            """Dataset that returns no data."""

            def __len__(self) -> int:
                return 0

            def __getitem__(self, idx: int) -> Batch:
                raise IndexError("Empty dataset")

        class SimpleTrainUnit(TrainUnit[Batch]):
            """Simple train unit for e2e testing."""

            def __init__(self) -> None:
                super().__init__()
                self.module = nn.Linear(2, 1)
                self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)

            def train_step(self, state: State, data: Batch) -> None:
                inputs, targets = data
                outputs = self.module(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        callback_with_warning = EmptyDataloaderDetectorCallback(
            threshold=2, raise_exception=False
        )

        train_unit = SimpleTrainUnit()
        empty_dataloader = DataLoader(EmptyDataset(), batch_size=1)

        with patch(
            "torchtnt.framework.callbacks.empty_dataloader_detector.logger"
        ) as mock_logger:
            train(
                train_unit,
                empty_dataloader,
                max_epochs=3,
                callbacks=[callback_with_warning],
            )

            mock_logger.warning.assert_called()
            warning_calls = mock_logger.warning.call_args_list
            empty_epoch_warnings = [
                call
                for call in warning_calls
                if "Empty train epoch detected" in str(call)
            ]
            self.assertEqual(len(empty_epoch_warnings), 3)

            threshold_warnings = [
                call
                for call in warning_calls
                if "consecutive empty train epochs" in str(call)
                and "exceeds the threshold" in str(call)
            ]
            self.assertGreaterEqual(len(threshold_warnings), 1)

        self.assertEqual(callback_with_warning._consecutive_empty_train_epochs, 3)

    def test_mixed_empty_and_non_empty_epochs(self) -> None:
        """
        Test with mixed empty and non-empty epochs to verify counter reset behavior.
        This test uses a dataset that alternates between empty and non-empty epochs.
        """

        class MixedDataset(Dataset[Batch]):
            """Dataset that alternates between empty and non-empty based on epoch."""

            def __init__(self) -> None:
                self.current_epoch = 0

            def __len__(self) -> int:
                return (
                    0 if self.current_epoch % 2 == 0 else 2
                )  # Even epochs (0, 2, 4) are empty, odd epochs (1, 3, 5) have data

            def __getitem__(self, idx: int) -> Batch:
                if self.current_epoch % 2 == 0:
                    raise IndexError("Empty epoch")
                else:
                    return (torch.randn(2), torch.randn(1))

            def increment_epoch(self) -> None:
                self.current_epoch += 1

        class EpochIncrementCallback(Callback):
            """Callback to increment the dataset's epoch counter."""

            def __init__(self, dataset: MixedDataset) -> None:
                self.dataset = dataset

            def on_train_epoch_start(self, state: State, unit: TrainUnit) -> None:
                self.dataset.increment_epoch()

        class SimpleTrainUnit(TrainUnit[Batch]):
            """Simple train unit for e2e testing."""

            def __init__(self) -> None:
                super().__init__()
                self.module = nn.Linear(2, 1)
                self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)

            def train_step(self, state: State, data: Batch) -> None:
                inputs, targets = data
                outputs = self.module(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        mixed_dataset = MixedDataset()
        epoch_increment_callback = EpochIncrementCallback(mixed_dataset)
        empty_detector_callback = EmptyDataloaderDetectorCallback(
            threshold=3, raise_exception=True
        )

        train_unit = SimpleTrainUnit()
        mixed_dataloader = DataLoader(mixed_dataset, batch_size=1)

        # Should complete successfully since non-empty epochs reset the counter
        # Pattern: empty(0), non-empty(1), empty(2), non-empty(3), empty(4), non-empty(5)
        train(
            train_unit,
            mixed_dataloader,
            max_epochs=6,
            callbacks=[epoch_increment_callback, empty_detector_callback],
        )
        self.assertEqual(empty_detector_callback._consecutive_empty_train_epochs, 1)


class MockTrainUnit(TrainUnit[Batch]):
    """Mock train unit for testing."""

    def __init__(self) -> None:
        super().__init__()
        self._steps_completed_in_prev_epoch = 0

    def train_step(self, state: State, data: Batch) -> None:
        pass

    def set_steps_completed_in_prev_epoch(self, steps: int) -> None:
        """Set the number of steps completed in the previous epoch."""
        self._steps_completed_in_prev_epoch = steps
        self.train_progress._num_steps_completed_in_prev_epoch = steps


class MockEvalUnit(EvalUnit[Batch]):
    """Mock eval unit for testing."""

    def __init__(self) -> None:
        super().__init__()
        self._steps_completed_in_prev_epoch = 0

    def eval_step(self, state: State, data: Batch) -> None:
        pass

    def set_steps_completed_in_prev_epoch(self, steps: int) -> None:
        """Set the number of steps completed in the previous epoch."""
        self._steps_completed_in_prev_epoch = steps
        self.eval_progress._num_steps_completed_in_prev_epoch = steps


class MockPredictUnit(PredictUnit[Batch]):
    """Mock predict unit for testing."""

    def __init__(self) -> None:
        super().__init__()
        self._steps_completed_in_prev_epoch = 0

    def predict_step(self, state: State, data: Batch) -> None:
        pass

    def set_steps_completed_in_prev_epoch(self, steps: int) -> None:
        """Set the number of steps completed in the previous epoch."""
        self._steps_completed_in_prev_epoch = steps
        self.predict_progress._num_steps_completed_in_prev_epoch = steps
