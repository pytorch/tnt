# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import torch

from torchtnt.runner.callback import Callback
from torchtnt.runner.state import State
from torchtnt.runner.unit import TrainUnit, TTrainData


class ProfilingCallback(Callback):
    def __init__(self, dir_name: str) -> None:
        if not dir_name:
            dir_name = tempfile.mkdtemp()
        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=dir_name),
            with_stack=True,
        )

    def on_train_start(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        self.profiler.start()

    def on_train_step_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        self.profiler.step()

    def on_train_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        self.profiler.stop()
