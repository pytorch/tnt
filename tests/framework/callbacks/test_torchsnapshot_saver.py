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
from typing import Any, Dict, Iterator, List
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import launcher
from torch.utils.data import DataLoader
from torchsnapshot import Snapshot
from torchsnapshot.snapshot import SNAPSHOT_METADATA_FNAME
from torchsnapshot.test_utils import assert_state_dict_eq, check_state_dict_eq

from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    DummyFitUnit,
    DummyTrainUnit,
    generate_random_dataloader,
    get_dummy_fit_state,
    get_dummy_train_state,
)
from torchtnt.framework.callbacks.lambda_callback import Lambda
from torchtnt.framework.callbacks.torchsnapshot_saver import (
    _delete_snapshot,
    _get_app_state,
    _override_knobs,
    _retrieve_checkpoint_dirpaths,
    get_latest_checkpoint_path,
    KnobOptions,
    RestoreOptions,
    TorchSnapshotSaver,
)
from torchtnt.framework.train import train
from torchtnt.utils.distributed import get_global_rank, PGWrapper
from torchtnt.utils.env import init_from_env, seed
from torchtnt.utils.test_utils import get_pet_launch_config, spawn_multi_process


class TorchSnapshotSaverTest(unittest.TestCase):
    cuda_available: bool = torch.cuda.is_available()
    distributed_available: bool = torch.distributed.is_available()

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
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot])
            self.assertTrue(
                os.path.exists(expected_path) and os.path.isdir(expected_path)
            )

    @patch.object(TorchSnapshotSaver, "_async_snapshot", autospec=True)
    def test_save_fit_entrypoint(self, mock_async_snapshot: Mock) -> None:
        input_dim = 2

        my_unit = DummyFitUnit(input_dim=input_dim)
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = TorchSnapshotSaver(
                temp_dir, save_every_n_train_steps=1, save_every_n_epochs=1
            )
            train_state = get_dummy_train_state()
            fit_state = get_dummy_fit_state()
            my_unit.train_progress._num_steps_completed = 15
            my_unit.eval_progress._num_steps_completed = 10

            snapshot.on_train_step_end(train_state, my_unit)
            snapshot_path = mock_async_snapshot.call_args.args[1]
            self.assertIn(f"epoch_0_step_{15}", snapshot_path)

            snapshot.on_train_step_end(fit_state, my_unit)
            snapshot_path = mock_async_snapshot.call_args.args[1]
            self.assertIn(f"epoch_0_step_{15 + 10}", snapshot_path)

            snapshot.on_train_epoch_end(train_state, my_unit)
            snapshot_path = mock_async_snapshot.call_args.args[1]
            self.assertIn(f"epoch_0_step_{15}", snapshot_path)

            snapshot.on_train_epoch_end(fit_state, my_unit)
            snapshot_path = mock_async_snapshot.call_args.args[1]
            self.assertIn(f"epoch_0_step_{15 + 10}", snapshot_path)

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
            snapshot_cb = TorchSnapshotSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )
            train(
                my_unit,
                stateful_dataloader,
                max_steps=max_steps,
                callbacks=[snapshot_cb],
            )
            # state_dict has been called once on dataloader
            self.assertEqual(stateful_dataloader.state_dict_call_count, 1)
            self.assertEqual(stateful_dataloader.load_state_dict_call_count, 0)

            # restoring from first checkpoint, has dataloader in manifest
            snapshot_cb.restore(
                temp_dir + f"/epoch_{0}_step_{save_every_n_train_steps}",
                my_unit,
                train_dataloader=stateful_dataloader,
            )
            # load_state_dict has been called once on dataloader
            self.assertEqual(stateful_dataloader.load_state_dict_call_count, 1)

            # restoring from last checkpoint (on train end), does not have dataloader state in manifest

            with self.assertLogs(level="WARNING") as log:
                snapshot_cb.restore(
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
            snapshot_cb = TorchSnapshotSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])

            # Include a directory that does not have snapshot metadata saved
            # The restore function should skip this
            os.mkdir(os.path.join(temp_dir, "epoch_100_step_200"))

            with mock.patch(
                "torchtnt.framework.callbacks.torchsnapshot_saver.TorchSnapshotSaver.restore"
            ) as mock_restore:
                restored = snapshot_cb.restore_from_latest(temp_dir, my_unit)
                self.assertIn(
                    temp_dir + f"/epoch_{max_epochs}_step_{expected_steps_per_epoch}",
                    mock_restore.call_args.args,
                )
                self.assertTrue(restored)

    def test_restore_from_latest_empty_dir(self) -> None:
        input_dim = 2
        save_every_n_train_steps = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_cb = TorchSnapshotSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )

            with self.assertLogs(level="WARNING") as log:
                restored = snapshot_cb.restore_from_latest(temp_dir, my_unit)
                self.assertEqual(
                    log.output,
                    [
                        f"WARNING:torchtnt.framework.callbacks.torchsnapshot_saver:Input dirpath doesn't contain any subdirectories: {temp_dir}"
                    ],
                )
                self.assertFalse(restored)

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
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])

            end_num_steps_completed = my_unit.train_progress.num_steps_completed
            self.assertGreater(len(expected_paths), 0)
            snapshot_cb.restore(
                expected_paths[0],
                my_unit,
                restore_options=RestoreOptions(restore_train_progress=False),
            )
            restored_num_steps_completed = my_unit.train_progress.num_steps_completed
            # no train progress was restored so the progress after restoration should be the same as the progress before restoration
            self.assertEqual(restored_num_steps_completed, end_num_steps_completed)

    @patch("torchtnt.framework.callbacks.torchsnapshot_saver.torchsnapshot")
    def test_save_restore_no_optimizer_restore(
        self, mock_torchsnapshot: MagicMock
    ) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        restore_options = RestoreOptions(restore_optimizers=False)
        TorchSnapshotSaver.restore(
            path="path/to/snapshot", unit=my_unit, restore_options=restore_options
        )
        app_state = mock_torchsnapshot.Snapshot().restore.call_args.args[0]
        self.assertNotIn("optimizer", app_state)
        TorchSnapshotSaver.restore(path="path/to/snapshot", unit=my_unit)
        app_state = mock_torchsnapshot.Snapshot().restore.call_args.args[0]
        self.assertIn("optimizer", app_state)

    @patch("torchtnt.framework.callbacks.torchsnapshot_saver.torchsnapshot")
    def test_save_restore_no_lr_scheduler_restore(
        self, mock_torchsnapshot: MagicMock
    ) -> None:
        my_unit = DummyAutoUnit(module=nn.Linear(2, 3))
        restore_options = RestoreOptions(restore_lr_schedulers=False)
        TorchSnapshotSaver.restore(
            path="path/to/snapshot", unit=my_unit, restore_options=restore_options
        )
        app_state = mock_torchsnapshot.Snapshot().restore.call_args.args[0]
        self.assertNotIn("lr_scheduler", app_state)
        TorchSnapshotSaver.restore(path="path/to/snapshot", unit=my_unit)
        app_state = mock_torchsnapshot.Snapshot().restore.call_args.args[0]
        self.assertIn("lr_scheduler", app_state)

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
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])

            expected_path = (
                f"epoch_{max_epochs}_step_{max_epochs * (dataset_len // batch_size)}"
            )
            self.assertTrue(os.path.exists(os.path.join(temp_dir, expected_path)))

            with self.assertLogs(level="WARNING") as log:
                # train again without resetting progress
                train(
                    my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb]
                )
                self.assertEqual(
                    log.output,
                    [
                        "WARNING:torchtnt.framework.callbacks.torchsnapshot_saver:Final checkpoint already exists, skipping."
                    ],
                )

    @unittest.skipUnless(
        condition=distributed_available, reason="Torch distributed is needed to run"
    )
    def test_directory_sync_collective(self) -> None:
        spawn_multi_process(
            2,
            "gloo",
            self._directory_sync_collective,
        )

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
        condition=distributed_available, reason="Torch distributed is needed to run"
    )
    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
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

        snapshot_cb = TorchSnapshotSaver(
            temp_dir,
            save_every_n_epochs=save_every_n_epochs,
            replicated=["**"],
        )
        temp_dir = snapshot_cb.dirpath
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])

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
            self.assertIsNone(get_latest_checkpoint_path(temp_dir))

        with tempfile.TemporaryDirectory() as temp_dir:
            latest_path = os.path.join(temp_dir, "epoch_0_step_0")
            os.mkdir(latest_path)
            self._create_snapshot_metadata(latest_path)
            self.assertEqual(get_latest_checkpoint_path(temp_dir), latest_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            path_1 = os.path.join(temp_dir, "epoch_0_step_0")
            os.mkdir(path_1)
            self._create_snapshot_metadata(path_1)
            path_2 = os.path.join(temp_dir, "epoch_0_step_100")
            os.mkdir(path_2)
            self._create_snapshot_metadata(path_2)

            # Missing metadata file
            path_3 = os.path.join(temp_dir, "epoch_1_step_100")
            os.mkdir(path_3)

            # Ill-formatted name
            path_4 = os.path.join(temp_dir, "epoch_700")
            os.mkdir(path_4)
            self.assertEqual(get_latest_checkpoint_path(temp_dir), path_2)

    @staticmethod
    def _create_snapshot_metadata(output_dir: str) -> None:
        path = os.path.join(output_dir, SNAPSHOT_METADATA_FNAME)
        with open(path, "w"):
            pass

    @unittest.skipUnless(
        condition=distributed_available, reason="Torch distributed is needed to run"
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
        tc.assertIsNone(get_latest_checkpoint_path(temp_dir))
        if is_rank0:
            shutil.rmtree(temp_dir)  # delete temp directory

        if is_rank0:
            temp_dir = tempfile.mkdtemp()
            path_1 = os.path.join(temp_dir, "epoch_0_step_0")
            os.mkdir(path_1)
            TorchSnapshotSaverTest._create_snapshot_metadata(path_1)
            path_2 = os.path.join(temp_dir, "epoch_0_step_100")
            os.mkdir(path_2)
            TorchSnapshotSaverTest._create_snapshot_metadata(path_2)

            # Missing metadata file
            path_3 = os.path.join(temp_dir, "epoch_1_step_100")
            os.mkdir(path_3)

            # Ill-formatted name
            path_4 = os.path.join(temp_dir, "epoch_700")
            os.mkdir(path_4)
        else:
            temp_dir = ""
            path_2 = ""

        pg = PGWrapper(dist.group.WORLD)
        path_container = [path_2] if is_rank0 else [None]
        pg.broadcast_object_list(path_container, 0)
        expected_path = path_container[0]
        tc.assertEqual(get_latest_checkpoint_path(temp_dir), expected_path)

        if is_rank0:
            shutil.rmtree(temp_dir)  # delete temp directory

    @unittest.skipUnless(
        condition=distributed_available, reason="Torch distributed is needed to run"
    )
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

        snapshot_cb = TorchSnapshotSaver(
            temp_dir,
            save_every_n_epochs=save_every_n_epochs,
            replicated=["**"],
        )
        temp_dir = snapshot_cb.dirpath
        train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])
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
            snapshot_cb.restore(ckpt_path, my_new_unit)

            assert_state_dict_eq(
                tc, my_new_unit.optimizer.state_dict(), my_unit.optimizer.state_dict()
            )
            assert_state_dict_eq(
                tc, my_new_unit.module.state_dict(), my_unit.module.state_dict()
            )
        finally:
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    @unittest.skipUnless(
        condition=distributed_available, reason="Torch distributed is needed to run"
    )
    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_process_group_plumbing(self) -> None:
        """
        Creates a new process group and verifies that it's passed through correctly
        """
        spawn_multi_process(
            2,
            "nccl",
            self._test_process_group_plumbing,
        )

    @staticmethod
    def _test_process_group_plumbing() -> None:
        new_pg = dist.new_group(backend="gloo")

        if get_global_rank() == 0:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = ""

        snapshot_cb = TorchSnapshotSaver(
            temp_dir,
            process_group=new_pg,
        )
        tc = unittest.TestCase()
        try:
            tc.assertEqual(snapshot_cb._process_group, new_pg)
        finally:
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    def test_knob_override(self) -> None:
        env_var = "TORCHSNAPSHOT_MAX_PER_RANK_IO_CONCURRENCY_OVERRIDE"
        knob_options = KnobOptions(max_per_rank_io_concurrency=1)
        with _override_knobs(knob_options):
            self.assertEqual(os.environ[env_var], str(1))
        self.assertNotIn(env_var, os.environ)

        with _override_knobs(KnobOptions(max_per_rank_io_concurrency=None)):
            self.assertNotIn(env_var, os.environ)

    @patch("torchtnt.framework.callbacks.torchsnapshot_saver.get_filesystem")
    def test_retrieve_checkpoint_dirpaths(self, mock_get_filesystem: MagicMock) -> None:
        """
        Tests retrieving checkpoint directories from a given root directory
        """
        paths = [
            {"name": "tmp/epoch_0_step_10", "type": "directory"},
            {"name": "tmp/epoch_1_step_10", "type": "directory"},
            {"name": "tmp/epoch_2_step_10", "type": "directory"},
            {"name": "tmp/epoch_0_step_5", "type": "directory"},
            {"name": "tmp/epoch_0_step_3", "type": "file"},
        ]

        mock_get_filesystem.return_value.ls.return_value = paths
        returned_paths = _retrieve_checkpoint_dirpaths("foo")
        self.assertEqual(
            returned_paths,
            [
                "tmp/epoch_0_step_5",
                "tmp/epoch_0_step_10",
                "tmp/epoch_1_step_10",
                "tmp/epoch_2_step_10",
            ],
        )

    def test_delete_snapshot(self) -> None:
        """
        Tests removing checkpoint directories
        """
        app_state = {"module": nn.Linear(2, 2)}
        with tempfile.TemporaryDirectory() as temp_dir:
            dirpath = os.path.join(temp_dir, "checkpoint")
            Snapshot.take(dirpath, app_state=app_state)
            self.assertTrue(os.path.exists(dirpath))
            # check that error is thrown if .snapshot_metadata is not found in the directory when deleting
            with self.assertRaisesRegex(
                RuntimeError, f"{temp_dir} does not contain .snapshot_metadata"
            ):
                _delete_snapshot(temp_dir)
            _delete_snapshot(dirpath)
            self.assertFalse(os.path.exists(dirpath))

    def test_should_remove_snapshot(self) -> None:
        """
        Tests the helper function that checks if snapshot should be removed or not
        """
        tss = TorchSnapshotSaver("temp")

        # keep_last_n_checkpoints is toggled off
        self.assertFalse(tss._should_remove_snapshot())

        # not enough checkpoints are saved yet to be removed
        tss._keep_last_n_checkpoints = 2
        tss._ckpt_dirpaths = ["bar"]
        self.assertFalse(tss._should_remove_snapshot())

        # enough checkpoints are there to remove
        tss._keep_last_n_checkpoints = 2
        tss._ckpt_dirpaths = ["foo", "bar"]
        self.assertTrue(tss._should_remove_snapshot())

    @patch("torchtnt.framework.callbacks.torchsnapshot_saver._delete_snapshot")
    def test_remove_snapshot(self, mock_delete_snapshot: MagicMock) -> None:
        """
        Tests the helper function that removes snapshots and updates the checkpoint paths
        """
        state = get_dummy_train_state()
        tss = TorchSnapshotSaver("temp")
        tss._ckpt_dirpaths = ["foo", "bar"]
        tss._remove_snapshot(state)

        mock_delete_snapshot.assert_called_once()
        self.assertEqual(len(tss._ckpt_dirpaths), 1)
        self.assertEqual(tss._ckpt_dirpaths[0], "bar")

    @patch("torchtnt.framework.callbacks.torchsnapshot_saver._delete_snapshot")
    def test_cleanup_surplus(self, mock_delete_snapshot: MagicMock) -> None:
        """
        Tests surplus of checkpoints being cleaned up
        """
        state = get_dummy_train_state()
        unit = DummyTrainUnit(input_dim=2)
        warning_messages = []
        with tempfile.TemporaryDirectory() as temp_dir:
            tss = TorchSnapshotSaver(temp_dir, keep_last_n_checkpoints=1)
            tss._ckpt_dirpaths = ["foo", "bar", "baz"]

            expected_warning_msg = " ".join(
                [
                    f"3 checkpoints found in {temp_dir}.",
                    f"Deleting {2} oldest",
                    "checkpoints to enforce ``keep_last_n_checkpoints`` argument.",
                ]
            )

            with patch(
                "torchtnt.framework.callbacks.torchsnapshot_saver.logging.Logger.warning",
                warning_messages.append,
            ):
                tss.on_train_start(state, unit)
            self.assertEqual(tss._ckpt_dirpaths, ["baz"])
            self.assertEqual(warning_messages[0], expected_warning_msg)

            tss = TorchSnapshotSaver(temp_dir)
            tss._ckpt_dirpaths = ["foo", "bar", "baz"]

            tss.on_train_start(state, unit)
            self.assertEqual(tss._ckpt_dirpaths, ["foo", "bar", "baz"])

    def test_keep_last_n_checkpoints(self) -> None:
        """
        Tests removing checkpoint directories
        """
        unit = DummyTrainUnit(input_dim=2)
        state = get_dummy_train_state()
        with tempfile.TemporaryDirectory() as temp_dir:
            tss = TorchSnapshotSaver(
                temp_dir,
                save_every_n_train_steps=1,
                keep_last_n_checkpoints=2,
            )

            # take 10 steps
            for _ in range(10):
                unit.train_progress.increment_step()
                tss.on_train_step_end(state, unit)
                # TODO remove time.sleep to avoid potential flaky test
                time.sleep(0.1)  # sleep to ensure enough time to checkpoint

            dirs = os.listdir(temp_dir)
            self.assertEqual(len(dirs), 2)
            self.assertIn("epoch_0_step_9", dirs)
            self.assertIn("epoch_0_step_10", dirs)

    def test_keep_last_n_checkpoints_e2e(self) -> None:
        """
        Tests removing checkpoint directories e2e
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_cb = TorchSnapshotSaver(
                temp_dir,
                save_every_n_train_steps=2,
                keep_last_n_checkpoints=1,
            )
            # Artificially increase the step duration, otherwise torchsnapshot
            # doesn't have the time to save all snapshots and will skip some.
            slowdown = Lambda(on_train_step_end=lambda *_: time.sleep(0.1))

            train(
                my_unit,
                dataloader,
                max_epochs=max_epochs,
                callbacks=[snapshot_cb, slowdown],
            )
            dirs = os.listdir(temp_dir)
            self.assertEqual(len(dirs), 1)
            self.assertIn(
                f"epoch_{max_epochs}_step_{dataset_len // batch_size * max_epochs}",
                os.listdir(temp_dir),
            )

    def test_get_app_state(self) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        state = get_dummy_train_state()

        app_state = _get_app_state(state, my_unit, intra_epoch=False)
        self.assertCountEqual(
            app_state.keys(),
            ["module", "optimizer", "loss_fn", "rng_state", "train_progress"],
        )


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
