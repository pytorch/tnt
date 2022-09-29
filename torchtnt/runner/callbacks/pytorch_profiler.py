# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtnt.runner.callback import Callback
from torchtnt.runner.state import EntryPoint, State
from torchtnt.runner.unit import (
    EvalUnit,
    PredictUnit,
    TEvalData,
    TPredictData,
    TrainUnit,
    TTrainData,
)


class PyTorchProfiler(Callback):
    """
    A callback which profiles user code using PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html

    Args:
        profiler: a torch.profiler.profile context manager which will be used

    """

    def __init__(
        self,
        profiler: torch.profiler.profile,
    ) -> None:
        self.profiler: torch.profiler.profile = profiler

    def on_train_start(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        self.profiler.start()

    def on_train_step_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        self.profiler.step()

    def on_train_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        self.profiler.stop()

    def on_eval_start(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        # if in fit do nothing since the profiler was already started in on_train_start
        if state.entry_point == EntryPoint.EVALUATE:
            self.profiler.start()

    def on_eval_step_end(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        self.profiler.step()

    def on_eval_end(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        # if in fit do nothing since the profiler will be stopped in on_train_end
        if state.entry_point == EntryPoint.EVALUATE:
            self.profiler.stop()

    def on_predict_start(self, state: State, unit: PredictUnit[TPredictData]) -> None:
        self.profiler.start()

    def on_predict_step_end(
        self, state: State, unit: PredictUnit[TPredictData]
    ) -> None:
        self.profiler.step()

    def on_predict_end(self, state: State, unit: PredictUnit[TPredictData]) -> None:
        self.profiler.stop()
