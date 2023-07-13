#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Set, Union
from unittest.mock import MagicMock

from torchtnt.framework._callback_handler import CallbackHandler
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.timer import Timer


class DummyCallback(Callback):
    def __init__(self) -> None:
        self.called_hooks: Set[str] = set()

    def on_exception(
        self,
        state: State,
        unit: Union[TTrainUnit, TEvalUnit, TPredictUnit],
        exc: BaseException,
    ):
        self.called_hooks.add("on_exception")

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_start")

    def on_train_epoch_start(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_epoch_start")

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_step_start")

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_step_end")

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_epoch_end")

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        self.called_hooks.add("on_train_end")

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_start")

    def on_eval_epoch_start(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_epoch_start")

    def on_eval_step_start(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_step_start")

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_step_end")

    def on_eval_epoch_end(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_epoch_end")

    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        self.called_hooks.add("on_eval_end")

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_start")

    def on_predict_epoch_start(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_epoch_start")

    def on_predict_step_start(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_step_start")

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_step_end")

    def on_predict_epoch_end(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_epoch_end")

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        self.called_hooks.add("on_predict_end")


class CallbackHandlerTest(unittest.TestCase):
    def test_callback_handler(self) -> None:
        """
        Test CallbackHandler with all of the hooks on Callback
        """
        unit = MagicMock(spec=TTrainUnit)
        timer = Timer()
        state = State(
            entry_point=EntryPoint.TRAIN,
            timer=timer,
            train_state=None,
        )
        callback = DummyCallback()
        called_hooks = callback.called_hooks
        cb_handler = CallbackHandler([callback])

        cb_handler.on_exception(state, unit, ValueError("test"))
        self.assertIn("on_exception", called_hooks)
        self.assertIn("DummyCallback.on_exception", timer.recorded_durations.keys())

        cb_handler.on_train_start(state, unit)
        self.assertIn("on_train_start", called_hooks)
        self.assertIn("DummyCallback.on_train_start", timer.recorded_durations.keys())

        cb_handler.on_train_epoch_start(state, unit)
        self.assertIn("on_train_epoch_start", called_hooks)
        self.assertIn(
            "DummyCallback.on_train_epoch_start", timer.recorded_durations.keys()
        )

        cb_handler.on_train_step_start(state, unit)
        self.assertIn("on_train_step_start", called_hooks)
        self.assertIn(
            "DummyCallback.on_train_step_start", timer.recorded_durations.keys()
        )

        cb_handler.on_train_step_end(state, unit)
        self.assertIn("on_train_step_end", called_hooks)
        self.assertIn(
            "DummyCallback.on_train_step_end", timer.recorded_durations.keys()
        )

        cb_handler.on_train_step_end(state, unit)
        self.assertIn("on_train_step_end", called_hooks)
        self.assertIn(
            "DummyCallback.on_train_step_end", timer.recorded_durations.keys()
        )

        cb_handler.on_train_epoch_end(state, unit)
        self.assertIn("on_train_epoch_end", called_hooks)
        self.assertIn(
            "DummyCallback.on_train_epoch_end", timer.recorded_durations.keys()
        )

        cb_handler.on_train_end(state, unit)
        self.assertIn("on_train_end", called_hooks)
        self.assertIn("DummyCallback.on_train_end", timer.recorded_durations.keys())

        unit = MagicMock(spec=TEvalUnit)
        cb_handler.on_eval_start(state, unit)
        self.assertIn("on_eval_start", called_hooks)
        self.assertIn("DummyCallback.on_eval_start", timer.recorded_durations.keys())

        cb_handler.on_eval_epoch_start(state, unit)
        self.assertIn("on_eval_epoch_start", called_hooks)
        self.assertIn(
            "DummyCallback.on_eval_epoch_start", timer.recorded_durations.keys()
        )

        cb_handler.on_eval_step_start(state, unit)
        self.assertIn("on_eval_step_start", called_hooks)
        self.assertIn(
            "DummyCallback.on_eval_step_start", timer.recorded_durations.keys()
        )

        cb_handler.on_eval_step_end(state, unit)
        self.assertIn("on_eval_step_end", called_hooks)
        self.assertIn("DummyCallback.on_eval_step_end", timer.recorded_durations.keys())

        cb_handler.on_eval_step_end(state, unit)
        self.assertIn("on_eval_step_end", called_hooks)
        self.assertIn("DummyCallback.on_eval_step_end", timer.recorded_durations.keys())

        cb_handler.on_eval_epoch_end(state, unit)
        self.assertIn("on_eval_epoch_end", called_hooks)
        self.assertIn(
            "DummyCallback.on_eval_epoch_end", timer.recorded_durations.keys()
        )

        cb_handler.on_eval_end(state, unit)
        self.assertIn("on_eval_end", called_hooks)
        self.assertIn("DummyCallback.on_eval_end", timer.recorded_durations.keys())

        unit = MagicMock(spec=TPredictUnit)
        cb_handler.on_predict_start(state, unit)
        self.assertIn("on_predict_start", called_hooks)
        self.assertIn("DummyCallback.on_predict_start", timer.recorded_durations.keys())

        cb_handler.on_predict_epoch_start(state, unit)
        self.assertIn("on_predict_epoch_start", called_hooks)
        self.assertIn(
            "DummyCallback.on_predict_epoch_start", timer.recorded_durations.keys()
        )

        cb_handler.on_predict_step_start(state, unit)
        self.assertIn("on_predict_step_start", called_hooks)
        self.assertIn(
            "DummyCallback.on_predict_step_start", timer.recorded_durations.keys()
        )

        cb_handler.on_predict_step_end(state, unit)
        self.assertIn("on_predict_step_end", called_hooks)
        self.assertIn(
            "DummyCallback.on_predict_step_end", timer.recorded_durations.keys()
        )

        cb_handler.on_predict_step_end(state, unit)
        self.assertIn("on_predict_step_end", called_hooks)
        self.assertIn(
            "DummyCallback.on_predict_step_end", timer.recorded_durations.keys()
        )

        cb_handler.on_predict_epoch_end(state, unit)
        self.assertIn("on_predict_epoch_end", called_hooks)
        self.assertIn(
            "DummyCallback.on_predict_epoch_end", timer.recorded_durations.keys()
        )

        cb_handler.on_predict_end(state, unit)
        self.assertIn("on_predict_end", called_hooks)
        self.assertIn("DummyCallback.on_predict_end", timer.recorded_durations.keys())
