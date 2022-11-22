# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc

from pyre_extensions import none_throws

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit


class GarbageCollector(Callback):
    """
    A callback that performs periodic synchronous garbage collection.

    In fully-synchronous distributed training, the same program is run
    across multiple processes. These processes need to communicate with each
    other, especially to communicate gradients to update model parameters.
    The overall program execution is therefore gated by the slowest running
    process. As a result, it's important that each process takes roughly the
    same amount of time to execute its code: otherwise we run into straggler
    processes. By default, Python's automatic garbage collection can be triggered at
    different points in each of these processes, creating the possibility of
    straggler processes. This callback makes it convenient to configure
    all processes performing garbage collection at the same time in the loop.

    Synchronizing the garbage collection can lead to a performance improvement.
    The frequency of garbage collection must be tuned based on the application at hand.

    Args:
        step_interval: number of steps to run before each collection
    """

    def __init__(self, step_interval: int) -> None:
        self._step_interval = step_interval

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        gc.disable()

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        train_state = none_throws(state.train_state)
        if train_state.progress.num_steps_completed % self._step_interval == 0:
            gc.collect()

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        gc.enable()

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        gc.disable()

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        eval_state = none_throws(state.eval_state)
        if eval_state.progress.num_steps_completed % self._step_interval == 0:
            gc.collect()

    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        gc.enable()

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        gc.disable()

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        predict_state = none_throws(state.predict_state)
        if predict_state.progress.num_steps_completed % self._step_interval == 0:
            gc.collect()

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        gc.enable()
