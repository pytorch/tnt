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
import time
import unittest
from typing import Any, List, Tuple

import torch
import torch.distributed as dist
from torch.distributed import launcher
from torch.optim.lr_scheduler import ExponentialLR

from torchtnt.framework._test_utils import DummyTrainUnit, generate_random_dataloader
from torchtnt.framework.auto_unit import AutoUnit
from torchtnt.framework.callbacks import Lambda, TorchSnapshotSaver
from torchtnt.framework.state import State
from torchtnt.framework.train import train
from torchtnt.utils.distributed import get_global_rank, PGWrapper
from torchtnt.utils.env import init_from_env
from torchtnt.utils.lr_scheduler import TLRScheduler
from torchtnt.utils.test_utils import get_pet_launch_config


class TorchSnapshotSaverTest(unittest.TestCase):
    cuda_available = torch.cuda.is_available()

    def test_save_every_n_train_steps(self) -> None:
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
            snapshot = TorchSnapshotSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
                replicated=["**"],
            )
            # Artificially increase the step duration, otherwise torchsnapshot
            # doesn't have the time to save all snapshots and will skip some.
            slowdown = Lambda(on_train_step_end=lambda *_: time.sleep(0.1))
            train(
                my_unit,
                dataloader,
                max_epochs=max_epochs,
                callbacks=[snapshot, slowdown],
            )
            for path in expected_paths:
                self.assertTrue(os.path.exists(path) and os.path.isdir(path))

    def test_save_every_n_train_epochs(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 3
        expected_steps_per_epoch = math.ceil(dataset_len / batch_size)
        save_every_n_train_epochs = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with tempfile.TemporaryDirectory() as temp_dir:
            expected_path = os.path.join(
                temp_dir,
                f"epoch_{save_every_n_train_epochs}_step_{expected_steps_per_epoch * (save_every_n_train_epochs)}",
            )
            snapshot = TorchSnapshotSaver(
                temp_dir,
                save_every_n_epochs=save_every_n_train_epochs,
                replicated=["**"],
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot])
            self.assertTrue(
                os.path.exists(expected_path) and os.path.isdir(expected_path)
            )

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
            snapshot_cb = TorchSnapshotSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
                replicated=["**"],
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])

            end_num_steps_completed = my_unit.train_progress.num_steps_completed
            self.assertGreater(len(expected_paths), 0)
            snapshot_cb.restore(expected_paths[0], my_unit)
            restored_num_steps_completed = my_unit.train_progress.num_steps_completed
            # A snapshot is saved every n steps
            # so the first snapshot's progress will be equal to save_every_n_train_steps
            self.assertNotEqual(restored_num_steps_completed, end_num_steps_completed)
            self.assertEqual(restored_num_steps_completed, save_every_n_train_steps)

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
            snapshot_cb = TorchSnapshotSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
                replicated=["**"],
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])

            end_num_steps_completed = my_unit.train_progress.num_steps_completed
            self.assertGreater(len(expected_paths), 0)
            snapshot_cb.restore(
                expected_paths[0], my_unit, restore_train_progress=False
            )
            restored_num_steps_completed = my_unit.train_progress.num_steps_completed
            # no train progress was restored so the progress after restoration should be the same as the progress before restoration
            self.assertEqual(restored_num_steps_completed, end_num_steps_completed)

    def test_save_on_train_end(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2

        expected_path = (
            f"epoch_{max_epochs}_step_{max_epochs * (dataset_len // batch_size)}"
        )

        my_unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(os.path.exists(os.path.join(temp_dir, expected_path)))
            snapshot_cb = TorchSnapshotSaver(
                temp_dir,
                replicated=["**"],
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])

            expected_path = (
                f"epoch_{max_epochs}_step_{max_epochs * (dataset_len // batch_size)}"
            )
            self.assertTrue(os.path.exists(os.path.join(temp_dir, expected_path)))

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_directory_sync_collective(self) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(config, entrypoint=self._directory_sync_collective)()

    @staticmethod
    def _directory_sync_collective() -> None:
        init_from_env()
        try:
            if get_global_rank() == 0:
                temp_dir = tempfile.mkdtemp()
            else:
                temp_dir = "foo"

            snapshot_cb = TorchSnapshotSaver(temp_dir)
            dirpath = snapshot_cb.dirpath
            tc = unittest.TestCase()
            tc.assertTrue("tmp" in dirpath)
            tc.assertFalse("foo" in dirpath)
        finally:
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_save_restore_fsdp(self) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(config, entrypoint=self._save_restore_fsdp)()

    @staticmethod
    def _save_restore_fsdp() -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2
        save_every_n_epochs = 1

        my_unit = DummyAutoUnit(input_dim=input_dim, strategy="fsdp")
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        if get_global_rank() == 0:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = ""

        snapshot_cb = TorchSnapshotSaver(
            temp_dir,
            save_every_n_epochs=save_every_n_epochs,
        )
        temp_dir = snapshot_cb.dirpath
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])

        tc = unittest.TestCase()
        try:
            my_new_unit = DummyAutoUnit(input_dim=input_dim, strategy="fsdp")
            tc.assertNotEqual(
                my_new_unit.optimizer.state_dict(), my_unit.optimizer.state_dict()
            )
            # get latest checkpoint
            ckpt_path = os.path.join(temp_dir, f"epoch_{max_epochs}_step_10")
            snapshot_cb.restore(ckpt_path, my_new_unit)
            tc.assertEqual(
                my_new_unit.optimizer.state_dict(), my_unit.optimizer.state_dict()
            )
        finally:
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    def test_saver_invalid_args(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                ValueError, "Invalid value passed for save_every_n_train_steps.*"
            ):
                TorchSnapshotSaver(temp_dir, save_every_n_train_steps=-2)
            with self.assertRaisesRegex(
                ValueError, "Invalid value passed for save_every_n_train_steps.*"
            ):
                TorchSnapshotSaver(temp_dir, save_every_n_train_steps=0)
            with self.assertRaisesRegex(
                ValueError, "Invalid value passed for save_every_n_epochs.*"
            ):
                TorchSnapshotSaver(temp_dir, save_every_n_epochs=-2)
            with self.assertRaisesRegex(
                ValueError, "Invalid value passed for save_every_n_epochs.*"
            ):
                TorchSnapshotSaver(temp_dir, save_every_n_epochs=0)

    def test_latest_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(TorchSnapshotSaver.get_latest_checkpoint_path(temp_dir))

        with tempfile.TemporaryDirectory() as temp_dir:
            latest_path = os.path.join(temp_dir, "epoch_0_step_0")
            os.mkdir(latest_path)
            self.assertEqual(
                TorchSnapshotSaver.get_latest_checkpoint_path(temp_dir), latest_path
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            path_1 = os.path.join(temp_dir, "epoch_0_step_0")
            os.mkdir(path_1)
            path_2 = os.path.join(temp_dir, "epoch_0_step_100")
            os.mkdir(path_2)
            path_3 = os.path.join(temp_dir, "epoch_1_step_100")
            os.mkdir(path_3)
            path_4 = os.path.join(temp_dir, "epoch_700")
            os.mkdir(path_4)
            self.assertEqual(
                TorchSnapshotSaver.get_latest_checkpoint_path(temp_dir), path_3
            )

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_latest_checkpoint_path_distributed(self) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(
            config, entrypoint=self._latest_checkpoint_path_distributed
        )()

    @staticmethod
    def _latest_checkpoint_path_distributed() -> None:
        tc = unittest.TestCase()
        is_rank0 = get_global_rank() == 0

        if is_rank0:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = ""
        tc.assertIsNone(TorchSnapshotSaver.get_latest_checkpoint_path(temp_dir))
        if is_rank0:
            shutil.rmtree(temp_dir)  # delete temp directory

        if is_rank0:
            temp_dir = tempfile.mkdtemp()
            path_1 = os.path.join(temp_dir, "epoch_0_step_0")
            os.mkdir(path_1)
            path_2 = os.path.join(temp_dir, "epoch_0_step_100")
            os.mkdir(path_2)
            path_3 = os.path.join(temp_dir, "epoch_1_step_100")
            os.mkdir(path_3)
            path_4 = os.path.join(temp_dir, "epoch_700")
            os.mkdir(path_4)
        else:
            temp_dir = ""
            path_3 = ""

        pg = PGWrapper(dist.group.WORLD)
        path_container = [path_3] if is_rank0 else [None]
        pg.broadcast_object_list(path_container, 0)
        expected_path = path_container[0]
        tc.assertEqual(
            TorchSnapshotSaver.get_latest_checkpoint_path(temp_dir), expected_path
        )

        if is_rank0:
            shutil.rmtree(temp_dir)  # delete temp directory


Batch = Tuple[torch.tensor, torch.tensor]


class DummyAutoUnit(AutoUnit[Batch]):
    def __init__(self, input_dim: int, *args, **kwargs):
        super().__init__(module=torch.nn.Linear(input_dim, 2), *args, **kwargs)

    def compute_loss(self, state: State, data: Batch) -> Tuple[torch.Tensor, Any]:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        my_optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)
        my_lr_scheduler = ExponentialLR(my_optimizer, gamma=0.9)
        return my_optimizer, my_lr_scheduler
