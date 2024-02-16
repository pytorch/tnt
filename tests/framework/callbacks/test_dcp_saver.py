#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, Iterator, List
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsnapshot.test_utils import assert_state_dict_eq, check_state_dict_eq

from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callbacks.checkpointer_types import RestoreOptions
from torchtnt.framework.callbacks.dcp_saver import DistributedCheckpointSaver
from torchtnt.framework.train import train
from torchtnt.utils.distributed import get_global_rank, spawn_multi_process
from torchtnt.utils.env import seed
from torchtnt.utils.test_utils import skip_if_not_distributed, skip_if_not_gpu


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
                        os.path.join(temp_dir, f"epoch_{epoch}_step_{cumulative_steps}")
                    )
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
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
                temp_dir + f"/epoch_{0}_step_{save_every_n_train_steps}",
                my_unit,
                train_dataloader=stateful_dataloader,
            )
            # load_state_dict has been called once on dataloader
            self.assertEqual(stateful_dataloader.load_state_dict_call_count, 1)

            # restoring from last checkpoint (on train end), does not have dataloader state in manifest

            with self.assertLogs(level="WARNING") as log:
                dcp_cb.restore(
                    temp_dir + f"/epoch_{1}_step_{max_steps}",
                    my_unit,
                    train_dataloader=stateful_dataloader,
                )
                # load_state_dict is not called again on dataloader because there is no dataloader in manifest
                self.assertEqual(stateful_dataloader.load_state_dict_call_count, 1)
                self.assertEqual(
                    log.output,
                    [
                        "WARNING:torchtnt.utils.rank_zero_log:train_dataloader was passed to `restore` but no train dataloader exists in the Snapshot"
                    ],
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
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[dcp_cb])

            with mock.patch(
                "torchtnt.framework.callbacks.dcp_saver.DistributedCheckpointSaver.restore"
            ) as mock_restore:
                restored = dcp_cb.restore_from_latest(temp_dir, my_unit, no_dist=True)
                self.assertIn(
                    temp_dir + f"/epoch_{max_epochs}_step_{expected_steps_per_epoch}",
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
                        os.path.join(temp_dir, f"epoch_{epoch}_step_{cumulative_steps}")
                    )
            dcp_cb = DistributedCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
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

    @patch("torchtnt.framework.callbacks.dcp_saver.dist_cp")
    def test_save_restore_no_optimizer_restore(self, mock_dist_cp: MagicMock) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(restore_optimizers=False)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot",
            unit=my_unit,
            restore_options=restore_options,
        )
        app_state = mock_dist_cp.load_state_dict.call_args.args[0]["app_state"]
        self.assertNotIn("optimizer", app_state)
        DistributedCheckpointSaver.restore(path="path/to/snapshot", unit=my_unit)
        app_state = mock_dist_cp.load_state_dict.call_args.args[0]["app_state"]
        self.assertIn("optimizer", app_state)

    @patch("torchtnt.framework.callbacks.dcp_saver.dist_cp")
    def test_save_restore_no_lr_scheduler_restore(
        self, mock_dist_cp: MagicMock
    ) -> None:
        my_unit = DummyAutoUnit(module=nn.Linear(2, 3))
        restore_options = RestoreOptions(restore_lr_schedulers=False)
        DistributedCheckpointSaver.restore(
            path="path/to/snapshot", unit=my_unit, restore_options=restore_options
        )
        app_state = mock_dist_cp.load_state_dict.call_args.args[0]["app_state"]
        self.assertNotIn("lr_scheduler", app_state)
        DistributedCheckpointSaver.restore(path="path/to/snapshot", unit=my_unit)
        app_state = mock_dist_cp.load_state_dict.call_args.args[0]["app_state"]
        self.assertIn("lr_scheduler", app_state)

    @skip_if_not_distributed
    @skip_if_not_gpu
    def test_save_restore_fsdp(self) -> None:
        spawn_multi_process(
            2,
            "nccl",
            self._save_restore_fsdp,
        )

    @staticmethod
    def _save_restore_fsdp() -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        save_every_n_epochs = 1

        my_unit = DummyAutoUnit(module=torch.nn.Linear(input_dim, 2), strategy="fsdp")
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        if get_global_rank() == 0:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = ""

        dcp_cb = DistributedCheckpointSaver(
            temp_dir,
            save_every_n_epochs=save_every_n_epochs,
        )
        temp_dir = dcp_cb.dirpath
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[dcp_cb])

        tc = unittest.TestCase()
        try:
            my_new_unit = DummyAutoUnit(
                module=torch.nn.Linear(input_dim, 2), strategy="fsdp"
            )
            tc.assertNotEqual(
                my_new_unit.optimizer.state_dict(), my_unit.optimizer.state_dict()
            )
            # get latest checkpoint
            ckpt_path = os.path.join(temp_dir, f"epoch_{max_epochs}_step_10")
            dcp_cb.restore(ckpt_path, my_new_unit)
            tc.assertEqual(
                my_new_unit.optimizer.state_dict(), my_unit.optimizer.state_dict()
            )
        finally:
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    @skip_if_not_distributed
    def test_save_restore_ddp(self) -> None:
        spawn_multi_process(
            2,
            "gloo",
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
            ckpt_path = os.path.join(temp_dir, f"epoch_{max_epochs}_step_10")
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


class DummyStatefulDataLoader:
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.state_dict_call_count = 0
        self.load_state_dict_call_count = 0

    def state_dict(self) -> Dict[str, Any]:
        self.state_dict_call_count += 1
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.load_state_dict_call_count += 1
        return None

    def __iter__(self) -> Iterator[object]:
        return iter(self.dataloader)
