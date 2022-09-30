# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Union

from torchtnt.runner.state import State
from torchtnt.runner.unit import (
    EvalUnit,
    PredictUnit,
    TEvalData,
    TPredictData,
    TrainUnit,
    TTrainData,
)

logger: logging.Logger = logging.getLogger(__name__)


class _OnExceptionMixin:
    def on_exception(
        self,
        state: State,
        unit: Union[
            TrainUnit[TTrainData], EvalUnit[TEvalData], PredictUnit[TPredictData]
        ],
        exc: BaseException,
    ) -> None:
        pass


class _NameMixin:
    @property
    def name(self) -> str:
        """Should be a distinct name per instance. This is useful for debugging, profiling, and checkpointing purposes."""
        return self.__class__.__qualname__


class _TrainCallback:
    def on_train_start(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        pass

    def on_train_epoch_start(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        pass

    def on_train_step_start(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        pass

    def on_train_step_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        pass

    def on_train_epoch_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        pass

    def on_train_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        pass


class _EvalCallback:
    def on_eval_start(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        pass

    def on_eval_epoch_start(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        pass

    def on_eval_step_start(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        pass

    def on_eval_step_end(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        pass

    def on_eval_epoch_end(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        pass

    def on_eval_end(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        pass


class _PredictCallback:
    def on_predict_start(self, state: State, unit: PredictUnit[TPredictData]) -> None:
        pass

    def on_predict_epoch_start(
        self, state: State, unit: PredictUnit[TPredictData]
    ) -> None:
        pass

    def on_predict_step_start(
        self, state: State, unit: PredictUnit[TPredictData]
    ) -> None:
        pass

    def on_predict_step_end(
        self, state: State, unit: PredictUnit[TPredictData]
    ) -> None:
        pass

    def on_predict_epoch_end(
        self, state: State, unit: PredictUnit[TPredictData]
    ) -> None:
        pass

    def on_predict_end(self, state: State, unit: PredictUnit[TPredictData]) -> None:
        pass


class Callback(
    _TrainCallback, _EvalCallback, _PredictCallback, _OnExceptionMixin, _NameMixin
):
    pass
