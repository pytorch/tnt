#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchtnt.framework.callbacks.dcp_saver import _LATEST_DCP_AVAIL

if not _LATEST_DCP_AVAIL:
    raise unittest.SkipTest("Latest Pytorch is required to run DCP tests")

import math
import os
import shutil
import tempfile
from typing import Any, Dict, Iterator, List, Optional
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE, StorageMeta
from torch.utils.data import DataLoader
from torchsnapshot.test_utils import assert_state_dict_eq, check_state_dict_eq
from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callbacks.checkpointer_types import KnobOptions, RestoreOptions
from torchtnt.framework.callbacks.dcp_saver import DistributedCheckpointSaver
from torchtnt.framework.train import train
from torchtnt.utils.distributed import get_global_rank, spawn_multi_process
from torchtnt.utils.env import seed
from torchtnt.utils.test_utils import skip_if_not_distributed


class DistributedCheckpointSaverTest(unittest.TestCase):
    def test_save_restore(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        expected_steps_per_epoch = math.ceil(dataset_len / batch_size)
        save_every_n_train_steps = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        expected_paths: List[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            cumulative_steps = 0
            for epoch in range(max_epochs):
                for _ in range(
                    save_every_n_train_steps,
                    expected_steps_per_epoch + 1,
                    save_every_n_train_steps,
                ):
                    cumulative_steps += save_every_n_train_steps
                    expected_paths.append(
                        os.path.join(
                            temp_dir, f"epoch_{epoch}_train_step_{cumulative_steps}"
                        )
                    )
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
                knob_options=KnobOptions(1),
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[dcp_cb])

            end_num_steps_completed = my_unit.train_progress.num_steps_completed
            self.assertGreater(len(expected_paths), 0)
            dcp_cb.restore(expected_paths[0], my_unit)
            restored_num_steps_completed = my_unit.train_progress.num_steps_completed
            # A snapshot is saved every n steps
            # so the first snapshot's progress will be equal to save_every_n_train_steps
            self.assertNotEqual(restored_num_steps_completed, end_num_steps_completed)
            self.assertEqual(restored_num_steps_completed, save_every_n_train_steps)

    def test_save_restore_dataloader_state(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        save_every_n_train_steps = 2
        max_steps = 3

        my_unit = DummyTrainUnit(input_dim=input_dim)
        stateful_dataloader = DummyStatefulDataLoader(
            dataloader=generate_random_dataloader(dataset_len, input_dim, batch_size)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
                knob_options=KnobOptions(1),
            )
            train(
                my_unit,
                stateful_dataloader,
                max_steps=max_steps,
                callbacks=[dcp_cb],
            )
            # state_dict has been called once on dataloader
            self.assertEqual(stateful_dataloader.state_dict_call_count, 1)
            self.assertEqual(stateful_dataloader.load_state_dict_call_count, 0)

            # restoring from first checkpoint, has dataloader in manifest
            dcp_cb.restore(
                temp_dir + f"/epoch_{0}_train_step_{save_every_n_train_steps}",
                my_unit,
                train_dataloader=stateful_dataloader,
            )
            # load_state_dict has been called once on dataloader
            self.assertEqual(stateful_dataloader.load_state_dict_call_count, 1)

            # restoring from last checkpoint (on train end), does not have dataloader state in manifest

            with self.assertLogs(level="WARNING") as log:
                dcp_cb.restore(
                    temp_dir + f"/epoch_{1}_train_step_{max_steps}",
                    my_unit,
                    train_dataloader=stateful_dataloader,
                )
                # load_state_dict is not called again on dataloader because there is no dataloader in manifest
                self.assertEqual(stateful_dataloader.load_state_dict_call_count, 1)
                self.assertEqual(
                    log.output[0],
                    "WARNING:torchtnt.utils.rank_zero_log:train_dataloader was passed to `restore` but no train dataloader exists in the Snapshot",
                )

    def test_restore_from_latest(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1
        save_every_n_train_steps = 2
        expected_steps_per_epoch = math.ceil(dataset_len / batch_size)

        my_unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
                knob_options=KnobOptions(1),
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[dcp_cb])

            with mock.patch(
                "torchtnt.framework.callbacks.dcp_saver.DistributedCheckpointSaver.restore"
            ) as mock_restore:
                restored = dcp_cb.restore_from_latest(temp_dir, my_unit, no_dist=True)
                self.assertIn(
                    temp_dir
                    + f"/epoch_{max_epochs}_train_step_{expected_steps_per_epoch}",
                    mock_restore.call_args.args,
                )
                self.assertTrue(restored)

    def test_save_restore_no_train_progress(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        expected_steps_per_epoch = math.ceil(dataset_len / batch_size)
        save_every_n_train_steps = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        expected_paths: List[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            cumulative_steps = 0
            for epoch in range(max_epochs):
                for _ in range(
                    save_every_n_train_steps,
                    expected_steps_per_epoch + 1,
                    save_every_n_train_steps,
                ):
                    cumulative_steps += save_every_n_train_steps
                    expected_paths.append(
                        os.path.join(
                            temp_dir, f"epoch_{epoch}_train_step_{cumulative_steps}"
                        )
                    )
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
                knob_options=KnobOptions(1),
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[dcp_cb])

            end_num_steps_completed = my_unit.train_progress.num_steps_completed
            self.assertGreater(len(expected_paths), 0)
            dcp_cb.restore(
                expected_paths[0],
                my_unit,
                restore_options=RestoreOptions(restore_train_progress=False),
            )
            restored_num_steps_completed = my_unit.train_progress.num_steps_completed
            # no train progress was restored so the progress after restoration should be the same as the progress before restoration
            self.assertEqual(restored_num_steps_completed, end_num_steps_completed)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_save_restore_no_optimizer_restore(self, mock_dist_cp: MagicMock) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(restore_optimizers=False)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
        )
        app_state = mock_dist_cp.load.call_args.args[0]["app_state"].state_dict()
        self.assertNotIn("optimizer", app_state)
        DistributedCheckpointSaver.restore(path="path/to/snapshot", unit=my_unit)
        app_state = mock_dist_cp.load.call_args.args[0]["app_state"].state_dict()
        self.assertIn("optimizer", app_state)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_save_restore_no_lr_scheduler_restore(
        self, mock_dist_cp: MagicMock
    ) -> None:
        my_unit = DummyAutoUnit(module=nn.Linear(2, 3))
        restore_options = RestoreOptions(restore_lr_schedulers=False)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot", unit=my_unit, restore_options=restore_options
        )
        app_state = mock_dist_cp.load.call_args.args[0]["app_state"].state_dict()
        self.assertNotIn("lr_scheduler", app_state)
        DistributedCheckpointSaver.restore(path="path/to/snapshot", unit=my_unit)
        app_state = mock_dist_cp.load.call_args.args[0]["app_state"].state_dict()
        self.assertIn("lr_scheduler", app_state)

    @skip_if_not_distributed
    def test_save_restore_ddp(self) -> None:
        spawn_multi_process(
            2,
            "cpu:gloo,cuda:gloo",
            self._save_restore_ddp,
        )

    @staticmethod
    def _save_restore_ddp() -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        save_every_n_epochs = 1
        seed(0)

        my_unit = DummyAutoUnit(module=torch.nn.Linear(input_dim, 2), strategy="ddp")
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        if get_global_rank() == 0:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = ""

        dcp_cb = DistributedCheckpointSaver(
            temp_dir,
            save_every_n_epochs=save_every_n_epochs,
            knob_options=KnobOptions(1),
        )
        temp_dir = dcp_cb.dirpath
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[dcp_cb])
        tc = unittest.TestCase()
        try:
            my_new_unit = DummyAutoUnit(
                module=torch.nn.Linear(input_dim, 2), strategy="ddp"
            )
            optim_equal = check_state_dict_eq(
                my_new_unit.optimizer.state_dict(), my_unit.optimizer.state_dict()
            )
            tc.assertFalse(optim_equal)
            module_equal = check_state_dict_eq(
                my_new_unit.module.state_dict(), my_unit.module.state_dict()
            )
            tc.assertFalse(module_equal)
            # get latest checkpoint
            ckpt_path = os.path.join(temp_dir, f"epoch_{max_epochs}_train_step_10")
            dcp_cb.restore(ckpt_path, my_new_unit)

            assert_state_dict_eq(
                tc, my_new_unit.optimizer.state_dict(), my_unit.optimizer.state_dict()
            )
            assert_state_dict_eq(
                tc, my_new_unit.module.state_dict(), my_unit.module.state_dict()
            )
        finally:
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_save_default_planner_storage_components(
        self, mock_dist_cp: MagicMock
    ) -> None:
        from torch.distributed.checkpoint._fsspec_filesystem import FsspecWriter

        input_dim = 2
        save_every_n_train_steps = 1

        my_unit = DummyTrainUnit(input_dim=input_dim)

        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
                knob_options=KnobOptions(1),
            )

            dcp_cb._save(
                checkpoint_id=temp_dir,
                app_state=my_unit.module.state_dict(),
            )

            planner = mock_dist_cp.save.call_args_list[0][1]["planner"]
            storage_writer = mock_dist_cp.save.call_args_list[0][1]["storage_writer"]

            self.assertIsInstance(planner, DefaultSavePlanner)
            self.assertIsInstance(storage_writer, FsspecWriter)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_save_planner_storage_components(self, mock_dist_cp: MagicMock) -> None:
        input_dim = 2
        save_every_n_train_steps = 1

        my_unit = DummyTrainUnit(input_dim=input_dim)

        with tempfile.TemporaryDirectory() as temp_dir:
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
                knob_options=KnobOptions(1),
            )

            dcp_cb._save(
                checkpoint_id=temp_dir,
                app_state=my_unit.module.state_dict(),
                planner=DummySavePlanner(),
                storage_writer=DummyStorageWriter(path=temp_dir),
            )

            planner = mock_dist_cp.save.call_args_list[0][1]["planner"]
            storage_writer = mock_dist_cp.save.call_args_list[0][1]["storage_writer"]

            self.assertIsInstance(planner, DummySavePlanner)
            self.assertIsInstance(storage_writer, DummyStorageWriter)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_restore_default_planner_storage_components(
        self, mock_dist_cp: MagicMock
    ) -> None:
        from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader

        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(restore_optimizers=False)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
        )
        planner = mock_dist_cp.load.call_args[1]["planner"]
        storage_reader = mock_dist_cp.load.call_args[1]["storage_reader"]

        self.assertIsInstance(planner, DefaultLoadPlanner)
        self.assertIsInstance(storage_reader, FsspecReader)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_restore_planner_storage_components(self, mock_dist_cp: MagicMock) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(restore_optimizers=False)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
            planner=DummyLoadPlanner(),
            storage_reader=DummyStorageReader(path="path/to/snapshot"),
        )
        planner = mock_dist_cp.load.call_args[1]["planner"]
        storage_reader = mock_dist_cp.load.call_args[1]["storage_reader"]

        self.assertIsInstance(planner, DummyLoadPlanner)
        self.assertIsInstance(storage_reader, DummyStorageReader)

    @patch("torchtnt.framework.callbacks.dcp_saver.dcp")
    def test_restore_allow_partial_loading(self, mock_dist_cp: MagicMock) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(strict=False)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
        )

        allow_partial_load = mock_dist_cp.load.call_args[1][
            "planner"
        ].allow_partial_load
        self.assertTrue(allow_partial_load)

        restore_options = RestoreOptions(strict=True)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
        )

        allow_partial_load = mock_dist_cp.load.call_args[1][
            "planner"
        ].allow_partial_load
        self.assertFalse(allow_partial_load)


class DummyStatefulDataLoader:
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.state_dict_call_count = 0
        self.load_state_dict_call_count = 0

    def state_dict(self) -> Dict[str, Any]:
        self.state_dict_call_count += 1
        return {"some_data": 1}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.load_state_dict_call_count += 1
        return None

    def __iter__(self) -> Iterator[object]:
        return iter(self.dataloader)


class DummySavePlanner(DefaultSavePlanner):
    def __init__(self) -> None:
        super().__init__()

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        storage_meta: Optional[StorageMeta],
        is_coordinator: bool,
    ) -> None:
        super().set_up_planner(state_dict, storage_meta, is_coordinator)


class DummyLoadPlanner(DefaultLoadPlanner):
    def __init__(self) -> None:
        super().__init__()

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[Metadata],
        is_coordinator: bool,
    ) -> None:
        super().set_up_planner(state_dict, metadata, is_coordinator)


class DummyStorageWriter(FileSystemWriter):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        pass


class DummyStorageReader(FileSystemReader):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        pass
