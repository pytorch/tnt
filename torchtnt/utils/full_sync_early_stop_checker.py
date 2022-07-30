#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import Optional, Union

import torch
import torch.distributed as dist
from torchtnt.utils.distributed import PGWrapper
from torchtnt.utils.early_stop_checker import EarlyStopChecker

_log: logging.Logger = logging.getLogger(__name__)


class FullSyncEarlyStopChecker:
    """
    Distributed early stop checker which returns a signal from monitoring a metric. The signal is
    decided upon an agreement within the whole process group. The current implementation supports 4
    different modes to make a consensus on stopping decision.
        1. any (default): If any rank receives a stopping signal, all ranks should signal to stop.
        2. all: Only if all ranks receive a stopping signal should all ranks stop.
        3. rank_zero: Makes rank 0 process's check result as source of truth and broadcasts the
            result across all other processes.
        4. None: Return immediately without doing any synchronization.

    Args:
        es_checker (EarlyStopChecker): a single process early stop checker initialized with metric
            monitoring criteria.
        pg (ProcessGroup, optional): The process group to work on. If None, the default process
            group will be used.
        coherence_mode (str, optional): Different mode through which users can communicate the
            stopping decision.

    Raises:
        ValueError:
            If the `coherence_mode` is not supported.
    """

    def __init__(
        self,
        es_checker: EarlyStopChecker,
        pg: Optional[dist.ProcessGroup] = None,
        coherence_mode: Optional[str] = "any",
    ) -> None:
        self._es_checker = es_checker
        self._pg: Optional[dist.ProcessGroup] = pg
        allowed_modes = ("all", "any", "rank_zero", None)
        if coherence_mode not in allowed_modes:
            raise ValueError(
                f"Invalid `coherence_mode` provided: {coherence_mode}. Expected one of {allowed_modes}."
            )
        self._coherence_mode: Optional[str] = coherence_mode

    def _check_sync(self, indicator: torch.Tensor) -> bool:
        if self._coherence_mode == "rank_zero":
            return self._check_rank_zero(indicator)
        elif self._coherence_mode == "any":
            return self._check_any(indicator)
        else:  # assume "all" coherence mode
            return self._check_all(indicator)

    def _check_rank_zero(self, indicator: torch.Tensor) -> bool:
        # Broadcast from rank 0 to all other ranks
        dist.broadcast(indicator, src=0, group=self._pg)
        return bool(indicator[0].item())

    def _check_any(self, indicator: torch.Tensor) -> bool:
        # sum up the indicators across all the ranks.
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return indicator.item() > 0

    def _check_all(self, indicator: torch.Tensor) -> bool:
        # sum up the indicators across all the ranks.
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return indicator.item() == PGWrapper(self._pg).get_world_size()

    def check(self, val: Union[torch.Tensor, float]) -> bool:

        ret = self._es_checker.check(val)
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or self._coherence_mode is None
        ):
            return ret

        device = torch.device(
            torch.cuda.current_device()
            # pyre-fixme[16]: `ProcessGroup` has no attribute `get_backend`.
            if self._pg and self._pg.get_backend() == "nccl"
            else "cpu"
        )

        dtype = torch.uint8
        if PGWrapper(self._pg).get_world_size() > 256:
            dtype = torch.int

        indicator = (
            torch.ones(1, device=device, dtype=dtype)
            if ret
            else torch.zeros(1, device=device, dtype=dtype)
        )

        # Get signal from all the nodes
        rank = PGWrapper(self._pg).get_rank()
        _log.debug(f"Early stop check on rank {rank}: {ret}")

        return self._check_sync(indicator)
