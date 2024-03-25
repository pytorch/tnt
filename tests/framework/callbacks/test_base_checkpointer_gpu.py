# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from datetime import timedelta
from typing import Iterable, Optional

from torch import distributed as dist
from torchtnt.framework.callbacks.base_checkpointer import BaseCheckpointer
from torchtnt.framework.callbacks.checkpointer_types import RestoreOptions
from torchtnt.framework.state import State
from torchtnt.framework.unit import AppStateMixin, TTrainData
from torchtnt.utils.distributed import spawn_multi_process
from torchtnt.utils.test_utils import skip_if_not_distributed, skip_if_not_gpu


class BaseCheckpointerStub(BaseCheckpointer):
    """
    We only need the BaseCheckpointer __init__ function to be executed, since the process group
    is created there. The rest of the functions are stubbed out.
    """

    def __init__(
        self, dirpath: str, process_group: Optional[dist.ProcessGroup]
    ) -> None:
        super().__init__(dirpath=dirpath, process_group=process_group)

    def _checkpoint_impl(
        self,
        state: State,
        unit: AppStateMixin,
        checkpoint_path: str,
        hook: str,
    ) -> bool:
        return True

    @staticmethod
    def restore(
        path: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
    ) -> None:
        return


class BaseCheckpointerGPUTest(unittest.TestCase):
    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_process_group_plumbing(self) -> None:
        """
        Creates a new process group and verifies GLOO group is created accordingly
        """
        spawn_multi_process(
            2,
            "nccl",
            self._test_process_group_plumbing,
        )
        spawn_multi_process(
            2,
            "gloo",
            self._test_process_group_plumbing,
        )
        spawn_multi_process(
            2,
            "nccl",
            self._test_process_group_plumbing,
        )

    @staticmethod
    def _test_process_group_plumbing() -> None:
        checkpoint_cb = BaseCheckpointerStub(
            "foo",
            process_group=None,
        )
        tc = unittest.TestCase()
        tc.assertEqual(
            dist.get_backend(checkpoint_cb._process_group), dist.Backend.GLOO
        )
        if dist.get_backend(dist.group.WORLD) == dist.Backend.GLOO:
            # verify no new process group was created
            tc.assertEqual(checkpoint_cb._process_group, dist.group.WORLD)

        print("Test successfully finished here")

    @staticmethod
    def _test_with_existing_nccl_process_group() -> None:
        checkpoint_cb = BaseCheckpointerStub(
            "foo",
            process_group=dist.new_group(
                timeout=timedelta(seconds=3600), backend=dist.Backend.NCCL
            ),
        )
        tc = unittest.TestCase()
        tc.assertEqual(
            dist.get_backend(checkpoint_cb._process_group), dist.Backend.GLOO
        )
