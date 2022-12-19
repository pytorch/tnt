# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

try:
    from torcheval.tools import (
        get_module_summary,
        get_summary_table,
        ModuleSummary,
        prune_module_summary,
    )

    _TORCHEVAL_AVAILABLE = True
except Exception:
    _TORCHEVAL_AVAILABLE = False

import logging

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit

logger: logging.Logger = logging.getLogger(__name__)


class ModuleSummaryCallback(Callback):
    """
    A callback which generates and logs a summary of the modules.
    """

    def __init__(self, max_depth: Optional[int] = None) -> None:
        if not _TORCHEVAL_AVAILABLE:
            raise RuntimeError(
                "ModuleSummary support requires torcheval. "
                "Please make sure ``torcheval`` is installed. "
                "Installation: https://github.com/pytorch/torcheval#installing-torcheval"
            )
        self._max_depth = max_depth
        self._module_summaries: List[ModuleSummary] = []

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        self._retrieve_module_summaries(unit)
        self._log_module_summary_tables()

    def _retrieve_module_summaries(self, unit: TTrainUnit) -> None:
        self._module_summaries.clear()
        for module_name in unit.tracked_modules():
            module_summary = get_module_summary(unit.tracked_modules()[module_name])
            module_summary._module_name = module_name
            if self._max_depth:
                prune_module_summary(module_summary, max_depth=self._max_depth)
            self._module_summaries.append(module_summary)

    def _log_module_summary_tables(self) -> None:
        for ms in self._module_summaries:
            logging.info("\n" + get_summary_table(ms))
