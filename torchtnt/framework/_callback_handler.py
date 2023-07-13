# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.framework.utils import _get_timing_context


class CallbackHandler:
    """
    A helper class to run and time callbacks in TorchTNT.
    """

    def __init__(self, callbacks: List[Callback]) -> None:
        self._callbacks = callbacks

    def on_exception(
        self,
        state: State,
        unit: Union[TTrainUnit, TEvalUnit, TPredictUnit],
        exc: BaseException,
    ) -> None:
        fn_name = "on_exception"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_exception(state, unit, exc)

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_start"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_train_start(state, unit)

    def on_train_epoch_start(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_epoch_start"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_train_epoch_start(state, unit)

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_step_start"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_train_step_start(state, unit)

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_step_end"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_train_step_end(state, unit)

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_epoch_end"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_train_epoch_end(state, unit)

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        fn_name = "on_train_end"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_train_end(state, unit)

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_start"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_eval_start(state, unit)

    def on_eval_epoch_start(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_epoch_start"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_eval_epoch_start(state, unit)

    def on_eval_step_start(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_step_start"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_eval_step_start(state, unit)

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_step_end"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_eval_step_end(state, unit)

    def on_eval_epoch_end(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_epoch_end"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_eval_epoch_end(state, unit)

    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        fn_name = "on_eval_end"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_eval_end(state, unit)

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_start"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_predict_start(state, unit)

    def on_predict_epoch_start(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_epoch_start"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_predict_epoch_start(state, unit)

    def on_predict_step_start(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_step_start"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_predict_step_start(state, unit)

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_step_end"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_predict_step_end(state, unit)

    def on_predict_epoch_end(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_epoch_end"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_predict_epoch_end(state, unit)

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        fn_name = "on_predict_end"
        callbacks = self._callbacks
        for cb in callbacks:
            with _get_timing_context(state, f"{cb.name}.{fn_name}"):
                cb.on_predict_end(state, unit)
