# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Union

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit
from torchtnt.framework.utils import extract_lr
from torchtnt.utils.loggers.logger import MetricLogger


def _write_stats(
    writers: List[MetricLogger],
    lr_stats: Dict[str, float],
    step: int,
) -> None:

    for writer in writers:
        writer.log_dict(lr_stats, step)


class LearningRateMonitor(Callback):
    """
    A callback which logs learning rate of tracked optimizers and learning rate schedulers.
    Logs learning rate for each parameter group associated with an optimizer.

    Args:
        loggers: Either a :class:`torchtnt.loggers.logger.MetricLogger` or
            list of :class:`torchtnt.loggers.logger.MetricLogger`
    """

    def __init__(
        self,
        loggers: Union[MetricLogger, List[MetricLogger]],
        *,
        logging_interval: str = "epoch",
    ) -> None:
        if not isinstance(loggers, list):
            loggers = [loggers]

        expected_intervals = ("epoch", "step")
        if logging_interval not in expected_intervals:
            raise ValueError(
                f"Invalid value '{logging_interval}' for argument logging_interval. Accepted values are {expected_intervals}."
            )

        self._loggers: List[MetricLogger] = loggers
        self.logging_interval = logging_interval

    def on_train_epoch_start(self, state: State, unit: TTrainUnit) -> None:
        if not self._loggers:
            return

        if self.logging_interval != "epoch":
            return

        lr_stats = extract_lr(unit)

        step = unit.train_progress.num_steps_completed
        _write_stats(self._loggers, lr_stats, step)

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        if not self._loggers:
            return

        if self.logging_interval != "step":
            return

        lr_stats = extract_lr(unit)

        step = unit.train_progress.num_steps_completed
        _write_stats(self._loggers, lr_stats, step)
