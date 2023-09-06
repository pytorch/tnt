# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Union

from pyre_extensions import none_throws
from torch.utils.tensorboard import SummaryWriter

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import PhaseState, State
from torchtnt.framework.unit import TTrainUnit
from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.loggers.tensorboard import TensorBoardLogger


class IterationTimeLogger(Callback):
    """
    A callback which logs iteration times as scalars to TensorBoard.

    Args:
        logger: Either a :class:`torchtnt.loggers.tensorboard.TensorBoardLogger`
            or a :class:`torch.utils.tensorboard.SummaryWriter` instance.
        moving_avg_window: an optional int to control the moving average window
        log_every_n_steps: an optional int to control the log frequency
    """

    def __init__(
        self,
        logger: Union[TensorBoardLogger, SummaryWriter],
        moving_avg_window: int = 1,
        log_every_n_steps: int = 1,
    ) -> None:
        if isinstance(logger, TensorBoardLogger):
            logger = logger.writer
        self._writer: SummaryWriter = none_throws(
            logger, "TensorBoardLogger.writer should not be None"
        )
        self.moving_avg_window = moving_avg_window
        self.log_every_n_steps = log_every_n_steps

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        if get_global_rank() != 0:  # only write from the main rank
            return

        step_logging_for = unit.train_progress.num_steps_completed
        if step_logging_for % self.log_every_n_steps != 0:
            return

        train_state: PhaseState = none_throws(state.train_state)
        train_iteration_time_list = none_throws(
            train_state.iteration_timer
        ).recorded_durations.get("train_iteration_time", [])
        if len(train_iteration_time_list) == 0:
            return

        last_n_values = train_iteration_time_list[-self.moving_avg_window :]
        self._writer.add_scalar(
            "Train Iteration Time (seconds)",
            sum(last_n_values) / len(last_n_values),
            step_logging_for,
        )
