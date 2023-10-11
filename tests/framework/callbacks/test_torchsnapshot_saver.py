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
from typing import Any, Dict, Iterable, List
from unittest import mock

import torch
import torch.distributed as dist
from torch.distributed import launcher
from torchsnapshot.snapshot import SNAPSHOT_METADATA_FNAME
from torchsnapshot.test_utils import assert_state_dict_eq, check_state_dict_eq

from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.framework.callbacks.lambda_callback import Lambda
from torchtnt.framework.callbacks.torchsnapshot_saver import (
    get_latest_checkpoint_path,
    TorchSnapshotSaver,
)
from torchtnt.framework.train import train
from torchtnt.utils.distributed import get_global_rank, PGWrapper
from torchtnt.utils.env import init_from_env, seed
from torchtnt.utils.test_utils import get_pet_launch_config, spawn_multi_process


class TorchSnapshotSaverTest(unittest.TestCase):
    # pyre-fixme[4]: Attribute must be annotated.
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
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[snapshot_cb])

            expected_path = (
                f"epoch_{max_epochs}_step_{max_epochs * (dataset_len // batch_size)}"
            )
            self.assertTrue(os.path.exists(os.path.join(temp_dir, expected_path)))

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
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

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
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

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
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

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
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

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
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


class DummyStatefulDataLoader:
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[24]: Generic type `Iterable` expects 1 type parameter.
    def __init__(self, dataloader: Iterable):
        self.dataloader = dataloader
        self.state_dict_call_count = 0
        self.load_state_dict_call_count = 0

    def state_dict(self) -> Dict[str, Any]:
        self.state_dict_call_count += 1
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.load_state_dict_call_count += 1
        return None

    # pyre-fixme[3]: Return type must be annotated.
    def __iter__(self):
        return iter(self.dataloader)
