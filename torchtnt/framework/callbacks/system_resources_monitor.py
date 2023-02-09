# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import psutil
import torch
from pyre_extensions import none_throws
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import PhaseState, State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.utils.device import collect_system_stats, get_device_from_env
from torchtnt.utils.loggers.logger import MetricLogger


def _write_stats(
    writers: List[MetricLogger],
    stats: Dict[str, float],
    step: int,
) -> None:

    for writer in writers:
        writer.log_dict(stats, step)


class SystemResourcesMonitor(Callback):
    """
    A callback which logs system stats, including:
    - CPU usage
    - resident set size
    - GPU usage
    - cuda memory stats

    Args:
        loggers: Logger (or iterable collection of loggers) for experiment tracking.
        Could be either a :class:`torchtnt.loggers.logger.MetricLogger` or
        list of :class:`torchtnt.loggers.logger.MetricLogger`
        logging_interval: whether to print system state every step or every epoch.
        Defaults to every epoch.
    """

    def __init__(
        self,
        loggers: Union[MetricLogger, List[MetricLogger]],
        *,
        logging_interval: Literal["epoch", "step"] = "epoch",
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
        self.process = psutil.Process()
        self.max_gpu_mem_usage_b: Optional[float] = None
        self.device: torch.device = get_device_from_env()

    def write_system_stats(
        self, logging_interval: Literal["epoch", "step"], state: PhaseState
    ) -> None:
        if not self._loggers:
            return

        if self.logging_interval != logging_interval:
            return

        system_stats = collect_system_stats(self.device)
        step = state.progress.num_steps_completed
        _write_stats(self._loggers, system_stats, step)

    def on_train_epoch_start(self, state: State, unit: TTrainUnit) -> None:
        self.write_system_stats("epoch", none_throws(state.train_state))

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        self.write_system_stats("step", none_throws(state.train_state))

    def on_eval_epoch_start(self, state: State, unit: TEvalUnit) -> None:
        self.write_system_stats("epoch", none_throws(state.eval_state))

    def on_eval_step_start(self, state: State, unit: TEvalUnit) -> None:
        self.write_system_stats("step", none_throws(state.eval_state))

    def on_predict_epoch_start(self, state: State, unit: TPredictUnit) -> None:
        self.write_system_stats("epoch", none_throws(state.predict_state))

    def on_predict_step_start(self, state: State, unit: TPredictUnit) -> None:
        self.write_system_stats("step", none_throws(state.predict_state))
