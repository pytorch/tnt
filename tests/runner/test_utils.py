#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import pytest

from torch import nn
from torchtnt.runner.callback import TrainCallback
from torchtnt.runner.progress import Progress
from torchtnt.runner.state import EntryPoint, PhaseState, State
from torchtnt.runner.unit import TrainUnit, TTrainData
from torchtnt.runner.utils import (
    _reset_module_training_mode,
    _run_callback_fn,
    _set_module_training_mode,
)
from torchtnt.utils import Timer


class UtilsTest(unittest.TestCase):
    def test_set_module_training_mode(self) -> None:
        """
        Test _set_module_training_mode
        """
        module = nn.Linear(1, 1)
        loss_fn = nn.CrossEntropyLoss()

        tracked_modules = {"module": module, "loss_fn": loss_fn}

        # set module training mode to False
        prior_module_train_states = _set_module_training_mode(tracked_modules, False)

        self.assertFalse(module.training)
        self.assertFalse(loss_fn.training)

        self.assertTrue(prior_module_train_states["module"])
        self.assertTrue(prior_module_train_states["loss_fn"])

        # set back to True
        prior_module_train_states = _set_module_training_mode(tracked_modules, True)

        self.assertTrue(module.training)
        self.assertTrue(loss_fn.training)

        self.assertFalse(prior_module_train_states["module"])
        self.assertFalse(prior_module_train_states["loss_fn"])

    def test_reset_module_training_mode(self) -> None:
        """
        Test _reset_module_training_mode
        """
        module = nn.Linear(1, 1)
        loss_fn = nn.CrossEntropyLoss()

        tracked_modules = {"module": module, "loss_fn": loss_fn}

        # set module training mode to False
        prior_module_train_states = _set_module_training_mode(tracked_modules, False)

        self.assertFalse(module.training)
        self.assertFalse(loss_fn.training)

        # set back to True using reset
        _reset_module_training_mode(tracked_modules, prior_module_train_states)

        self.assertTrue(module.training)
        self.assertTrue(loss_fn.training)

    def test_run_callback_fn(self) -> None:
        """
        Test _run_callback_fn
        """
        callback = DummyTrainCallback("train")
        train_unit = MagicMock()
        dummy_train_state = State(
            entry_point=EntryPoint.TRAIN,
            timer=Timer(),
            train_state=PhaseState(progress=Progress(), dataloader=[1, 2, 3]),
        )

        self.assertEqual(callback.dummy_data, "train")
        _run_callback_fn([callback], "on_train_start", dummy_train_state, train_unit)
        self.assertEqual(callback.dummy_data, "on_train_start")

        with pytest.raises(ValueError, match="Invalid callback method name provided"):
            _run_callback_fn([callback], "dummy_attr", dummy_train_state, train_unit)

        with pytest.raises(
            AttributeError, match="object has no attribute 'on_train_finish'"
        ):
            _run_callback_fn(
                [callback], "on_train_finish", dummy_train_state, train_unit
            )


class DummyTrainCallback(TrainCallback):
    def __init__(self, dummy_data: str) -> None:
        self.dummy_data = dummy_data
        self.dummy_attr = 1

    def on_train_start(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        self.dummy_data = "on_train_start"

    def on_train_epoch_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        self.dummy_data = "on_train_epoch_end"
