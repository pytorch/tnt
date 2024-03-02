#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import math
import os
import shutil
import tempfile
import time
import unittest
from typing import Iterable, List, Optional
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from torchtnt.framework._test_utils import (
    Batch,
    DummyFitUnit,
    DummyTrainUnit,
    generate_random_dataloader,
    get_dummy_fit_state,
    get_dummy_train_state,
)
from torchtnt.framework.callbacks.base_checkpointer import BaseCheckpointer
from torchtnt.framework.callbacks.checkpointer_types import (
    BestCheckpointConfig,
    RestoreOptions,
)
from torchtnt.framework.callbacks.lambda_callback import Lambda
from torchtnt.framework.fit import fit
from torchtnt.framework.state import State

from torchtnt.framework.train import train
from torchtnt.framework.unit import AppStateMixin, TrainUnit, TTrainData
from torchtnt.utils.distributed import get_global_rank, spawn_multi_process
from torchtnt.utils.env import init_from_env
from torchtnt.utils.test_utils import skip_if_not_distributed, skip_if_not_gpu


class BaseCheckpointSaver(BaseCheckpointer):
    """
    A basic checkpointer class that generates an empty directory upon checkpoint
    """

    def __init__(
        self,
        dirpath: str,
        *,
        save_every_n_train_steps: Optional[int] = None,
        save_every_n_epochs: Optional[int] = None,
        save_every_n_eval_epochs: Optional[int] = None,
        keep_last_n_checkpoints: Optional[int] = None,
        best_checkpoint_config: Optional[BestCheckpointConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__(
            dirpath,
            save_every_n_train_steps=save_every_n_train_steps,
            save_every_n_epochs=save_every_n_epochs,
            save_every_n_eval_epochs=save_every_n_eval_epochs,
            keep_last_n_checkpoints=keep_last_n_checkpoints,
            best_checkpoint_config=best_checkpoint_config,
            process_group=process_group,
        )
        self._latest_checkpoint_path: str = ""

    def _checkpoint_impl(
        self, state: State, unit: AppStateMixin, checkpoint_path: str, hook: str
    ) -> bool:
        self._latest_checkpoint_path = checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        return True

    @staticmethod
    def restore(
        path: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        msg: str = "",
    ) -> None:
        print(f"Checkpoint restored with message: {msg}")
        return


class BaseCheckpointerTest(unittest.TestCase):
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
            checkpointer = BaseCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )
            train(
                my_unit,
                dataloader,
                max_epochs=max_epochs,
                callbacks=[checkpointer],
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
            checkpointer = BaseCheckpointSaver(
                temp_dir,
                save_every_n_epochs=save_every_n_train_epochs,
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[checkpointer])
            self.assertTrue(
                os.path.exists(expected_path) and os.path.isdir(expected_path)
            )

    def test_save_every_n_eval_epochs(self) -> None:
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 3
        expected_steps_per_epoch = math.ceil(dataset_len / batch_size)
        save_every_n_eval_epochs = 2

        my_unit = DummyFitUnit(input_dim=input_dim)
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        with tempfile.TemporaryDirectory() as temp_dir:
            expected_path = os.path.join(
                temp_dir,
                f"epoch_2_step_{expected_steps_per_epoch * 4}",  # 3 train epochs + 1 eval epoch = 4
            )
            checkpointer = BaseCheckpointSaver(
                temp_dir,
                save_every_n_eval_epochs=save_every_n_eval_epochs,
            )
            fit(
                my_unit,
                dataloader,
                eval_dataloader=dataloader,
                evaluate_every_n_epochs=1,
                max_epochs=max_epochs,
                callbacks=[checkpointer],
            )
            self.assertTrue(
                os.path.exists(expected_path) and os.path.isdir(expected_path)
            )

    def test_save_fit_entrypoint(self) -> None:
        input_dim = 2

        my_unit = DummyFitUnit(input_dim=input_dim)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpointer = BaseCheckpointSaver(
                temp_dir, save_every_n_train_steps=1, save_every_n_epochs=1
            )
            train_state = get_dummy_train_state()
            fit_state = get_dummy_fit_state()
            my_unit.train_progress._num_steps_completed = 15
            my_unit.eval_progress._num_steps_completed = 10

            checkpointer.on_train_step_end(train_state, my_unit)
            self.assertIn(f"epoch_0_step_{15}", checkpointer._latest_checkpoint_path)

            checkpointer.on_train_step_end(fit_state, my_unit)
            self.assertIn(
                f"epoch_0_step_{15 + 10}", checkpointer._latest_checkpoint_path
            )

            checkpointer.on_train_epoch_end(train_state, my_unit)
            self.assertIn(f"epoch_0_step_{15}", checkpointer._latest_checkpoint_path)

            checkpointer.on_train_epoch_end(fit_state, my_unit)
            self.assertIn(
                f"epoch_0_step_{15 + 10}", checkpointer._latest_checkpoint_path
            )

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_restore_from_latest(self, mock_stdout: MagicMock) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            # create a dummy directory structure
            os.makedirs(os.path.join(temp_dir, "epoch_0_step_0"))

            BaseCheckpointSaver.restore_from_latest(temp_dir, my_unit, msg="foo")

            # ensure **kwargs are plumbed through correctly
            self.assertEqual(
                mock_stdout.getvalue(), "Checkpoint restored with message: foo\n"
            )

    def test_restore_from_latest_empty_dir(self) -> None:
        input_dim = 2
        save_every_n_train_steps = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        with tempfile.TemporaryDirectory() as temp_dir:
            bcs_cb = BaseCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=save_every_n_train_steps,
            )

            with self.assertLogs(level="WARNING") as log:
                restored = bcs_cb.restore_from_latest(temp_dir, my_unit)
                self.assertEqual(
                    log.output,
                    [
                        f"WARNING:torchtnt.framework.callbacks._checkpoint_utils:Input dirpath doesn't contain any subdirectories: {temp_dir}"
                    ],
                )
                self.assertFalse(restored)

    def test_restore_from_best(self) -> None:
        input_dim = 2
        state = get_dummy_train_state()

        with tempfile.TemporaryDirectory() as temp_dir:
            bcs_cb = BaseCheckpointSaver(temp_dir)

            my_unit = DummyTrainUnit(input_dim=input_dim)
            bcs_cb._generate_checkpoint_and_upkeep(state, my_unit, hook="foo")
            os.rename(
                os.path.join(temp_dir, "epoch_0_step_0"),
                os.path.join(temp_dir, "epoch_0_step_0_val_loss=0.01"),
            )

            my_unit.train_progress._num_steps_completed = 10
            bcs_cb._generate_checkpoint_and_upkeep(state, my_unit, hook="foo")
            os.rename(
                os.path.join(temp_dir, "epoch_0_step_10"),
                os.path.join(temp_dir, "epoch_0_step_10_val_loss=-0.1"),
            )

            my_unit.train_progress._num_steps_completed = 20
            bcs_cb._generate_checkpoint_and_upkeep(state, my_unit, hook="foo")
            os.rename(
                os.path.join(temp_dir, "epoch_0_step_20"),
                os.path.join(temp_dir, "epoch_0_step_20_val_loss=0.1"),
            )

            my_unit = DummyTrainUnit(input_dim=input_dim)
            with self.assertLogs(level="INFO") as log:
                restored = bcs_cb.restore_from_best(
                    temp_dir, my_unit, "val_loss", "min"
                )
                self.assertTrue(restored)
                self.assertIn(
                    f"INFO:torchtnt.utils.rank_zero_log:Loading checkpoint from {os.path.join(temp_dir, 'epoch_0_step_10_val_loss=-0.1')}",
                    log.output,
                )

            my_unit = DummyTrainUnit(input_dim=input_dim)
            with self.assertLogs(level="INFO") as log:
                restored = bcs_cb.restore_from_best(
                    temp_dir, my_unit, "val_loss", "max"
                )
                self.assertTrue(restored)
                self.assertIn(
                    f"INFO:torchtnt.utils.rank_zero_log:Loading checkpoint from {os.path.join(temp_dir, 'epoch_0_step_20_val_loss=0.1')}",
                    log.output,
                )

    def test_restore_from_best_empty_dir(self) -> None:
        input_dim = 2

        my_unit = DummyTrainUnit(input_dim=input_dim)
        with tempfile.TemporaryDirectory() as temp_dir:
            bcs_cb = BaseCheckpointSaver(
                temp_dir,
            )

            with self.assertLogs(level="WARNING") as log:
                restored = bcs_cb.restore_from_best(
                    temp_dir, my_unit, "val_loss", "min"
                )
                self.assertIn(
                    f"WARNING:torchtnt.framework.callbacks.base_checkpointer:No checkpoints with metric name val_loss were found in {temp_dir}. Not loading any checkpoint.",
                    log.output,
                )
                self.assertFalse(restored)

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
            checkpoint_cb = BaseCheckpointSaver(
                temp_dir,
            )
            train(my_unit, dataloader, max_epochs=max_epochs, callbacks=[checkpoint_cb])

            expected_path = (
                f"epoch_{max_epochs}_step_{max_epochs * (dataset_len // batch_size)}"
            )
            self.assertTrue(os.path.exists(os.path.join(temp_dir, expected_path)))

            with self.assertLogs(level="WARNING") as log:
                checkpoint_cb.metadata_fname = ".metadata"
                # create metadata file
                with open(os.path.join(temp_dir, expected_path, ".metadata"), "w"):
                    pass

                # train again without resetting progress
                train(
                    my_unit,
                    dataloader,
                    max_epochs=max_epochs,
                    callbacks=[checkpoint_cb],
                )
                self.assertEqual(
                    log.output,
                    [
                        "WARNING:torchtnt.framework.callbacks.base_checkpointer:Final checkpoint already exists, skipping."
                    ],
                )

    @skip_if_not_distributed
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

            bcs = BaseCheckpointSaver(temp_dir)
            dirpath = bcs.dirpath
            tc = unittest.TestCase()
            tc.assertTrue("tmp" in dirpath)
            tc.assertFalse("foo" in dirpath)
        finally:
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    def test_invalid_args(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                ValueError, "Invalid value passed for save_every_n_train_steps.*"
            ):
                BaseCheckpointSaver(temp_dir, save_every_n_train_steps=-2)
            with self.assertRaisesRegex(
                ValueError, "Invalid value passed for save_every_n_train_steps.*"
            ):
                BaseCheckpointSaver(temp_dir, save_every_n_train_steps=0)
            with self.assertRaisesRegex(
                ValueError, "Invalid value passed for save_every_n_epochs.*"
            ):
                BaseCheckpointSaver(temp_dir, save_every_n_epochs=-2)
            with self.assertRaisesRegex(
                ValueError, "Invalid value passed for save_every_n_epochs.*"
            ):
                BaseCheckpointSaver(temp_dir, save_every_n_epochs=0)

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

    @staticmethod
    def _test_process_group_plumbing() -> None:
        checkpoint_cb = BaseCheckpointSaver(
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

    @patch(
        "torchtnt.framework.callbacks.base_checkpointer.get_checkpoint_dirpaths",
        return_value=["epoch_1_step_10", "epoch_2_step_20"],
    )
    def test_ckpt_dirpaths(self, _: MagicMock) -> None:
        """
        Tests that ckpt_dirpaths is populated correctly
        based on if ``keep_last_n_checkpoints`` is set.
        """
        bc = BaseCheckpointSaver("foo")
        self.assertEqual(bc._ckpt_dirpaths, [])

        bc = BaseCheckpointSaver("foo", keep_last_n_checkpoints=10)
        self.assertEqual(bc._ckpt_dirpaths, ["epoch_1_step_10", "epoch_2_step_20"])

    def test_should_remove_checkpoint(self) -> None:
        """
        Tests the helper function that checks if checkpoint should be removed or not
        """
        bc = BaseCheckpointSaver("temp")

        # keep_last_n_checkpoints is toggled off
        self.assertFalse(bc._should_remove_checkpoint())

        # not enough checkpoints are saved yet to be removed
        bc._keep_last_n_checkpoints = 2
        bc._ckpt_dirpaths = ["bar"]
        self.assertFalse(bc._should_remove_checkpoint())

        # enough checkpoints are there to remove
        bc._keep_last_n_checkpoints = 2
        bc._ckpt_dirpaths = ["foo", "bar"]
        self.assertTrue(bc._should_remove_checkpoint())

    @patch("torchtnt.framework.callbacks.base_checkpointer._delete_checkpoint")
    def test_cleanup_surplus(self, mock_delete_checkpoint: MagicMock) -> None:
        """
        Tests surplus of checkpoints being cleaned up
        """
        state = get_dummy_train_state()
        unit = DummyTrainUnit(input_dim=2)
        warning_messages = []
        with tempfile.TemporaryDirectory() as temp_dir:
            bc = BaseCheckpointSaver(temp_dir, keep_last_n_checkpoints=1)
            bc._ckpt_dirpaths = ["foo", "bar", "baz"]

            expected_warning_msg = " ".join(
                [
                    f"3 checkpoints found in {temp_dir}.",
                    f"Deleting {2} oldest",
                    "checkpoints to enforce ``keep_last_n_checkpoints`` argument.",
                ]
            )

            with patch(
                "torchtnt.framework.callbacks.base_checkpointer.logging.Logger.warning",
                warning_messages.append,
            ):
                bc.on_train_start(state, unit)
            self.assertEqual(bc._ckpt_dirpaths, ["baz"])
            self.assertEqual(warning_messages[0], expected_warning_msg)

            bc = BaseCheckpointSaver(temp_dir)
            bc._ckpt_dirpaths = ["foo", "bar", "baz"]

            bc.on_train_start(state, unit)
            self.assertEqual(bc._ckpt_dirpaths, ["foo", "bar", "baz"])

    def test_keep_last_n_checkpoints(self) -> None:
        """
        Tests removing checkpoint directories
        """
        unit = DummyTrainUnit(input_dim=2)
        state = get_dummy_train_state()
        with tempfile.TemporaryDirectory() as temp_dir:
            bc = BaseCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=1,
                keep_last_n_checkpoints=2,
            )

            # take 10 steps
            for _ in range(10):
                unit.train_progress.increment_step()
                bc.on_train_step_end(state, unit)
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
            bc = BaseCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=2,
                keep_last_n_checkpoints=1,
            )
            # Artificially increase the step duration, otherwise torchcheckpoint
            # doesn't have the time to save all checkpoints and will skip some.
            slowdown = Lambda(on_train_step_end=lambda *_: time.sleep(0.1))

            train(
                my_unit,
                dataloader,
                max_epochs=max_epochs,
                callbacks=[bc, slowdown],
            )
            dirs = os.listdir(temp_dir)
            self.assertEqual(len(dirs), 1)
            self.assertIn(
                f"epoch_{max_epochs}_step_{dataset_len // batch_size * max_epochs}",
                os.listdir(temp_dir),
            )

    def test_does_checkpoint_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, ".metadata"), "w"):
                pass
            bc = BaseCheckpointSaver(
                temp_dir,
                save_every_n_train_steps=2,
                keep_last_n_checkpoints=1,
            )
            # checkpointer doesn't have a metadata_fname
            does_checkpoint_exist = bc._does_checkpoint_exist(temp_dir)
            self.assertFalse(does_checkpoint_exist)

            # checkpointer has metadata_fname and the file exists
            bc.metadata_fname = ".metadata"
            does_checkpoint_exist = bc._does_checkpoint_exist(temp_dir)
            self.assertTrue(does_checkpoint_exist)

            # checkpointer has metadata_fname but the file doesn't exist
            os.remove(os.path.join(temp_dir, ".metadata"))
            does_checkpoint_exist = bc._does_checkpoint_exist(temp_dir)
            self.assertFalse(does_checkpoint_exist)

    def test_should_save_checkpoint(self) -> None:
        """
        Tests basic functionality of should_save_checkpoint
        """
        bc = BaseCheckpointSaver("foo")

        # test default behavior
        self.assertTrue(bc._should_save_checkpoint())

        bc._ckpt_dirpaths = ["foo/epoch_0_step_1"]
        self.assertTrue(bc._should_save_checkpoint())
        bc._keep_last_n_checkpoints = 1
        self.assertTrue(bc._should_save_checkpoint())

        bc._ckpt_dirpaths = ["foo/epoch_0_step_1_val_loss=0.01"]
        bc._best_checkpoint_config = BestCheckpointConfig(
            monitored_metric="val_loss",
            mode="min",
        )
        bc._keep_last_n_checkpoints = None
        self.assertTrue(bc._should_save_checkpoint(0.02))
        bc._keep_last_n_checkpoints = 1
        self.assertFalse(bc._should_save_checkpoint(0.02))
        self.assertTrue(bc._should_save_checkpoint(0.001))
        bc._keep_last_n_checkpoints = 2
        self.assertTrue(bc._should_save_checkpoint(0.02))

        bc._best_checkpoint_config = BestCheckpointConfig(
            monitored_metric="val_loss",
            mode="max",
        )
        bc._keep_last_n_checkpoints = 1
        self.assertTrue(bc._should_save_checkpoint(0.02))
        self.assertFalse(bc._should_save_checkpoint(0.001))
        bc._keep_last_n_checkpoints = 2
        self.assertTrue(bc._should_save_checkpoint(0.001))

    def test_best_checkpoint_attr_missing(self) -> None:
        bcs = BaseCheckpointSaver(
            "foo",
            save_every_n_epochs=1,
            best_checkpoint_config=BestCheckpointConfig(
                monitored_metric="train_loss",
                mode="min",
            ),
        )

        state = get_dummy_train_state()
        my_val_unit = MyValLossUnit()

        with self.assertRaisesRegex(
            RuntimeError,
            "Unit does not have attribute train_loss, unable to retrieve metric to checkpoint.",
        ):
            bcs.on_train_epoch_end(state, my_val_unit)

    def test_best_checkpoint_no_top_k(self) -> None:
        """
        Tests basic functionality of best checkpoint saving

        - Checks that the best checkpoint is saved when the monitored metric
        - top_k is not configured, so no checkpoints should be deleted
        """

        for mode in ("min", "max"):
            with tempfile.TemporaryDirectory() as temp_dir:
                bcs = BaseCheckpointSaver(
                    temp_dir,
                    save_every_n_epochs=1,
                    best_checkpoint_config=BestCheckpointConfig(
                        monitored_metric="train_loss",
                        # pyre-fixme: Incompatible parameter type [6]
                        mode=mode,
                    ),
                )

                state = get_dummy_train_state()
                my_train_unit = MyTrainLossUnit()

                my_train_unit.train_loss = None
                bcs.on_train_epoch_end(state, my_train_unit)
                # none metric-value will not be updated in checkpoint dirpaths
                self.assertEqual(bcs._ckpt_dirpaths, [])
                self.assertEqual(os.listdir(temp_dir), ["epoch_0_step_0"])

                my_train_unit.train_loss = 0.01
                bcs.on_train_epoch_end(state, my_train_unit)
                self.assertEqual(
                    bcs._ckpt_dirpaths,
                    [os.path.join(temp_dir, "epoch_0_step_0_train_loss=0.01")],
                )

                my_train_unit.train_loss = 0.02
                my_train_unit.train_progress.increment_epoch()
                bcs.on_train_epoch_end(state, my_train_unit)
                self.assertEqual(
                    bcs._ckpt_dirpaths,
                    [
                        os.path.join(temp_dir, "epoch_1_step_0_train_loss=0.02"),
                        os.path.join(temp_dir, "epoch_0_step_0_train_loss=0.01"),
                    ]
                    if mode == "min"
                    else [
                        os.path.join(temp_dir, "epoch_0_step_0_train_loss=0.01"),
                        os.path.join(temp_dir, "epoch_1_step_0_train_loss=0.02"),
                    ],
                )

                my_train_unit.train_loss = 0.015
                my_train_unit.train_progress.increment_epoch()
                bcs.on_train_epoch_end(state, my_train_unit)
                self.assertEqual(
                    bcs._ckpt_dirpaths,
                    [
                        os.path.join(temp_dir, "epoch_1_step_0_train_loss=0.02"),
                        os.path.join(temp_dir, "epoch_2_step_0_train_loss=0.015"),
                        os.path.join(temp_dir, "epoch_0_step_0_train_loss=0.01"),
                    ]
                    if mode == "min"
                    else [
                        os.path.join(temp_dir, "epoch_0_step_0_train_loss=0.01"),
                        os.path.join(temp_dir, "epoch_2_step_0_train_loss=0.015"),
                        os.path.join(temp_dir, "epoch_1_step_0_train_loss=0.02"),
                    ],
                )

    def test_best_checkpoint_top_k(self) -> None:
        # test top_k = 1
        with tempfile.TemporaryDirectory() as temp_dir:
            bcs = BaseCheckpointSaver(
                temp_dir,
                save_every_n_epochs=1,
                best_checkpoint_config=BestCheckpointConfig(
                    monitored_metric="train_loss",
                    mode="min",
                ),
                keep_last_n_checkpoints=1,
            )

            state = get_dummy_train_state()
            my_train_unit = MyTrainLossUnit()

            bcs.on_train_epoch_end(state, my_train_unit)
            self.assertEqual(
                bcs._ckpt_dirpaths,
                [os.path.join(temp_dir, "epoch_0_step_0_train_loss=0.01")],
            )

            my_train_unit.train_loss = 0.02
            my_train_unit.train_progress.increment_epoch()
            bcs.on_train_epoch_end(state, my_train_unit)
            self.assertEqual(
                bcs._ckpt_dirpaths,
                [
                    os.path.join(temp_dir, "epoch_0_step_0_train_loss=0.01"),
                ],
            )

            my_train_unit.train_loss = 0.001
            my_train_unit.train_progress.increment_epoch()
            bcs.on_train_epoch_end(state, my_train_unit)
            self.assertEqual(
                bcs._ckpt_dirpaths,
                [
                    os.path.join(temp_dir, "epoch_2_step_0_train_loss=0.001"),
                ],
            )

        # test top_k = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            bcs = BaseCheckpointSaver(
                temp_dir,
                save_every_n_epochs=1,
                best_checkpoint_config=BestCheckpointConfig(
                    monitored_metric="train_loss",
                    mode="min",
                ),
                keep_last_n_checkpoints=2,
            )

            state = get_dummy_train_state()
            my_train_unit = MyTrainLossUnit()

            bcs.on_train_epoch_end(state, my_train_unit)
            self.assertEqual(
                bcs._ckpt_dirpaths,
                [os.path.join(temp_dir, "epoch_0_step_0_train_loss=0.01")],
            )

            my_train_unit.train_loss = 0.02
            my_train_unit.train_progress.increment_epoch()
            bcs.on_train_epoch_end(state, my_train_unit)
            self.assertEqual(
                bcs._ckpt_dirpaths,
                [
                    os.path.join(temp_dir, "epoch_1_step_0_train_loss=0.02"),
                    os.path.join(temp_dir, "epoch_0_step_0_train_loss=0.01"),
                ],
            )

            my_train_unit.train_loss = 0.001
            my_train_unit.train_progress.increment_epoch()
            bcs.on_train_epoch_end(state, my_train_unit)
            self.assertEqual(
                bcs._ckpt_dirpaths,
                [
                    os.path.join(temp_dir, "epoch_0_step_0_train_loss=0.01"),
                    os.path.join(temp_dir, "epoch_2_step_0_train_loss=0.001"),
                ],
            )


class MyValLossUnit(TrainUnit[Batch]):
    def __init__(self) -> None:
        super().__init__()
        self.val_loss = 0.01

    def train_step(self, state: State, data: Batch) -> None:
        return None


class MyTrainLossUnit(TrainUnit[Batch]):
    def __init__(self) -> None:
        super().__init__()
        self.train_loss: Optional[float] = 0.01

    def train_step(self, state: State, data: Batch) -> None:
        return None
