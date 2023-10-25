# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Union

from pyre_extensions import none_throws
from torch.utils.tensorboard import SummaryWriter

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.loggers.tensorboard import TensorBoardLogger
from torchtnt.utils.timer import TimerProtocol


class IterationTimeLogger(Callback):
    """
    A callback which logs iteration times as scalars to TensorBoard.

    Args:
        logger: Either a :class:`torchtnt.loggers.tensorboard.TensorBoardLogger`
            or a :class:`torch.utils.tensorboard.SummaryWriter` instance.
        moving_avg_window: an optional int to control the moving average window
        log_every_n_steps: an optional int to control the log frequency
    """

    _writer: Optional[SummaryWriter] = None

    def __init__(
        self,
        logger: Union[TensorBoardLogger, SummaryWriter],
        moving_avg_window: int = 1,
        log_every_n_steps: int = 1,
    ) -> None:
        if isinstance(logger, TensorBoardLogger):
            logger = logger.writer

        if get_global_rank() == 0:  # only write from the main rank
            self._writer = none_throws(
                logger, "TensorBoardLogger.writer should not be None"
            )
        self.moving_avg_window = moving_avg_window
        self.log_every_n_steps = log_every_n_steps

    def _log_step_metrics(
        self,
        writer: SummaryWriter,
        metric_label: str,
        iteration_timer: TimerProtocol,
        step_logging_for: int,
    ) -> None:
        """
        Helper function to write a timing log message to writer based on how the class
        was configured.

        """
        if step_logging_for % self.log_every_n_steps != 0:
            return

        human_metric_names = {
            "train_iteration_time": "Train Iteration Time (seconds)",
            "eval_iteration_time": "Eval Iteration Time (seconds)",
            "predict_iteration_time": "Prediction Iteration Time (seconds)",
        }

        time_list = iteration_timer.recorded_durations.get(metric_label, [])
        if not time_list:
            return

        last_n_values = time_list[-self.moving_avg_window :]
        writer.add_scalar(
            human_metric_names[metric_label],
            sum(last_n_values) / len(last_n_values),
            step_logging_for,
        )

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        timer = none_throws(state.train_state).iteration_timer
        if writer := self._writer:
            self._log_step_metrics(
                writer,
                "train_iteration_time",
                timer,
                unit.train_progress.num_steps_completed,
            )

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        timer = none_throws(state.eval_state).iteration_timer
        if writer := self._writer:
            self._log_step_metrics(
                writer,
                "eval_iteration_time",
                timer,
                unit.eval_progress.num_steps_completed,
            )

    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        timer = none_throws(state.predict_state).iteration_timer
        if writer := self._writer:
            self._log_step_metrics(
                writer,
                "predict_iteration_time",
                timer,
                unit.predict_progress.num_steps_completed,
            )
