#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging

from fsspec import open as fs_open
from torchtnt.utils.loggers.file import FileLogger
from torchtnt.utils.loggers.logger import MetricLogger

logger: logging.Logger = logging.getLogger(__name__)


class CSVLogger(FileLogger, MetricLogger):
    """
    CSV file logger. CSV headers are time, step, and names passed to `log`.

    Args:
        path (str): path to write logs to
        steps_before_flushing: (int, optional): Number of steps to buffer in logger before flushing
        log_all_ranks: (bool, optional): Log all ranks if true, else log only on rank 0.
    """

    def __init__(
        self,
        path: str,
        steps_before_flushing: int = 100,
        log_all_ranks: bool = False,
    ) -> None:
        super().__init__(path, steps_before_flushing, log_all_ranks)

    def flush(self) -> None:
        data = self._log_buffer
        if not data:
            logger.debug("No logs to write.")
            return

        if self._rank == 0 or self._log_all_ranks:
            with fs_open(self.path, "w") as f:
                data_list = list(data.values())
                w = csv.DictWriter(f, data_list[0].keys())
                w.writeheader()
                w.writerows(data_list)

    def close(self) -> None:
        self.flush()
