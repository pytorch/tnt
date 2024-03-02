# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import shutil
import tempfile
import unittest

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import launcher
from torchsnapshot import Snapshot
from torchsnapshot.snapshot import SNAPSHOT_METADATA_FNAME
from torchtnt.framework._test_utils import DummyTrainUnit, get_dummy_train_state

from torchtnt.framework.callbacks._checkpoint_utils import (
    _delete_checkpoint,
    _metadata_exists,
    _prepare_app_state_for_checkpoint,
    _retrieve_checkpoint_dirpaths,
    _sort_by_metric_value,
    _sort_by_recency,
    get_best_checkpoint_path,
    get_checkpoint_dirpaths,
    get_latest_checkpoint_path,
    rank_zero_read_and_broadcast,
)
from torchtnt.utils.distributed import get_global_rank, PGWrapper, spawn_multi_process
from torchtnt.utils.env import init_from_env
from torchtnt.utils.fsspec import get_filesystem
from torchtnt.utils.test_utils import get_pet_launch_config, skip_if_not_distributed

METADATA_FNAME: str = ".metadata"


class CheckpointUtilsTest(unittest.TestCase):
    @staticmethod
    def _create_snapshot_metadata(output_dir: str) -> None:
        path = os.path.join(output_dir, METADATA_FNAME)
        with open(path, "w"):
            pass

    def test_latest_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(get_latest_checkpoint_path(temp_dir))

        with tempfile.TemporaryDirectory() as temp_dir:
            latest_path = os.path.join(temp_dir, "epoch_0_step_0")
            os.mkdir(latest_path)
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir),
                latest_path,
            )
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME),
                None,
            )
            self._create_snapshot_metadata(latest_path)
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME),
                latest_path,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            path_1 = os.path.join(temp_dir, "epoch_0_step_0")
            os.mkdir(path_1)
            self._create_snapshot_metadata(path_1)
            path_2 = os.path.join(temp_dir, "epoch_0_step_100_val_loss=0.002")
            os.mkdir(path_2)
            self._create_snapshot_metadata(path_2)

            # Missing metadata file
            path_3 = os.path.join(temp_dir, "epoch_1_step_100")
            os.mkdir(path_3)

            # Ill-formatted name
            path_4 = os.path.join(temp_dir, "epoch_700")
            os.mkdir(path_4)
            self.assertEqual(
                get_latest_checkpoint_path(temp_dir, METADATA_FNAME), path_2
            )

    @skip_if_not_distributed
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
            CheckpointUtilsTest._create_snapshot_metadata(path_1)
            path_2 = os.path.join(temp_dir, "epoch_0_step_100")
            os.mkdir(path_2)
            CheckpointUtilsTest._create_snapshot_metadata(path_2)

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
        tc.assertEqual(
            get_latest_checkpoint_path(temp_dir, METADATA_FNAME), expected_path
        )

        if is_rank0:
            shutil.rmtree(temp_dir)  # delete temp directory

    def test_best_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(get_best_checkpoint_path(temp_dir, "val_loss", "min"))

            # no checkpoint w/ metric value
            path = os.path.join(temp_dir, "epoch_0_step_0")
            os.mkdir(path)
            self.assertIsNone(get_best_checkpoint_path(temp_dir, "val_loss", "min"))

        with tempfile.TemporaryDirectory() as temp_dir:
            best_path = os.path.join(temp_dir, "epoch_0_step_0_val_loss=0.01")
            os.mkdir(best_path)
            self.assertEqual(
                get_best_checkpoint_path(temp_dir, "val_loss", "min"),
                best_path,
            )
            self.assertIsNone(
                get_best_checkpoint_path(temp_dir, "val_loss", "min", METADATA_FNAME),
                None,
            )
            self._create_snapshot_metadata(best_path)
            self.assertEqual(
                get_best_checkpoint_path(temp_dir, "val_loss", "min", METADATA_FNAME),
                best_path,
            )

            # handle negative values
            best_path_2 = os.path.join(temp_dir, "epoch_0_step_0_val_loss=-0.01")
            os.mkdir(best_path_2)
            self.assertEqual(
                get_best_checkpoint_path(temp_dir, "val_loss", "min"),
                best_path_2,
            )

            # handle "max" mode correctly
            best_path_3 = os.path.join(temp_dir, "epoch_0_step_100_val_loss=0.1")
            os.mkdir(best_path_3)
            self.assertEqual(
                get_best_checkpoint_path(temp_dir, metric_name="val_loss", mode="max"),
                best_path_3,
            )

            # handle different metric correctly
            best_path_4 = os.path.join(temp_dir, "epoch_0_step_100_train_loss=0.2")
            os.mkdir(best_path_4)
            self.assertEqual(
                get_best_checkpoint_path(temp_dir, metric_name="val_loss", mode="max"),
                best_path_3,
            )
            self.assertEqual(
                get_best_checkpoint_path(
                    temp_dir, metric_name="train_loss", mode="max"
                ),
                best_path_4,
            )

    def test_retrieve_checkpoint_dirpaths(self) -> None:
        """
        Tests retrieving checkpoint directories from a given root directory
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [
                "epoch_0_step_10",
                "epoch_1_step_10",
                "epoch_2_step_10",
                "epoch_0_step_5",
                "epoch_0_step_6",
                "epoch_0_step_3",
            ]
            for path in paths[:-1]:
                os.mkdir(os.path.join(temp_dir, path))
            # make last path a file instead of a directory
            with open(os.path.join(temp_dir, paths[-1]), "w"):
                pass

            # compares set equality since order of returned dirpaths is not guaranteed
            # in _retrieve_checkpoint_dirpaths
            self.assertEqual(
                set(_retrieve_checkpoint_dirpaths(temp_dir, metadata_fname=None)),
                {os.path.join(temp_dir, path) for path in paths[:-1]},
            )
            self.assertEqual(
                _retrieve_checkpoint_dirpaths(temp_dir, metadata_fname=".metadata"),
                [],
            )

            # check metadata file is correct filtered for
            # by creating metadata for 3rd path in list
            with open(os.path.join(temp_dir, paths[2], ".metadata"), "w"):
                pass

            self.assertEqual(
                set(
                    _retrieve_checkpoint_dirpaths(temp_dir, metadata_fname=".metadata")
                ),
                {os.path.join(temp_dir, paths[2])},
            )

    def test_retrieve_checkpoint_dirpaths_with_metrics(self) -> None:
        """
        Tests retrieving checkpoint (w/ metrics) directories from a given root directory
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [
                "epoch_0_step_10_val_loss=10",
                "epoch_1_step_10_val_loss=5",
                "epoch_2_step_10",
                "epoch_0_step_5",
                "epoch_0_step_6_train_loss=13",
            ]
            for path in paths:
                os.mkdir(os.path.join(temp_dir, path))
            # make last path a file instead of a directory
            with open(os.path.join(temp_dir, "epoch_0_step_3_val_loss=3"), "w"):
                pass

            # compares set equality since order of returned dirpaths is not guaranteed
            # in _retrieve_checkpoint_dirpaths
            self.assertEqual(
                set(_retrieve_checkpoint_dirpaths(temp_dir, metadata_fname=None)),
                {os.path.join(temp_dir, path) for path in paths},
            )
            self.assertEqual(
                set(
                    _retrieve_checkpoint_dirpaths(
                        temp_dir, metadata_fname=None, metric_name="val_loss"
                    )
                ),
                {
                    os.path.join(temp_dir, path) for path in paths[:2]
                },  # since last path is a file
            )
            self.assertEqual(
                _retrieve_checkpoint_dirpaths(temp_dir, metadata_fname=".metadata"),
                [],
            )

            # check metadata file is correct filtered for
            # by creating metadata for 3rd path in list
            with open(os.path.join(temp_dir, paths[1], ".metadata"), "w"):
                pass

            self.assertEqual(
                set(
                    _retrieve_checkpoint_dirpaths(
                        temp_dir, metadata_fname=".metadata", metric_name="val_loss"
                    )
                ),
                {os.path.join(temp_dir, paths[1])},
            )

    @skip_if_not_distributed
    def test_distributed_get_checkpoint_dirpaths(self) -> None:
        spawn_multi_process(2, "gloo", self._distributed_get_checkpoint_dirpaths)

    @staticmethod
    def _distributed_get_checkpoint_dirpaths() -> None:
        """
        Tests that existing checkpoint directories are read and
        properly registered on all ranks
        """

        @rank_zero_read_and_broadcast
        def create_tmp_dir() -> str:
            return tempfile.mkdtemp()

        init_from_env()

        temp_dir = create_tmp_dir()
        try:
            path1 = os.path.join(temp_dir, "epoch_0_step_10")
            path2 = os.path.join(temp_dir, "epoch_1_step_20")
            if get_global_rank() == 0:
                os.mkdir(path1)
                os.mkdir(path2)
            torch.distributed.barrier()

            ckpt_dirpaths = get_checkpoint_dirpaths(temp_dir)
            tc = unittest.TestCase()
            tc.assertEqual(set(ckpt_dirpaths), {path1, path2})

            tc.assertEqual(
                get_checkpoint_dirpaths(temp_dir, metadata_fname=".metadata"), []
            )
        finally:
            if get_global_rank() == 0:
                shutil.rmtree(temp_dir)  # delete temp directory

    def test_get_checkpoint_dirpaths(self) -> None:
        """
        Tests that `get_checkpoint_dirpaths` returns
        the sorted checkpoint directories correctly
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path1 = os.path.join(temp_dir, "epoch_1_step_20")
            path2 = os.path.join(temp_dir, "epoch_4_step_130")
            path3 = os.path.join(temp_dir, "epoch_0_step_10")
            os.mkdir(path1)
            os.mkdir(path2)
            os.mkdir(path3)

            self.assertEqual(
                set(get_checkpoint_dirpaths(temp_dir)),
                {path1, path2, path3},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            path1 = os.path.join(temp_dir, "epoch_1_step_20_val_loss=0.01")
            path2 = os.path.join(temp_dir, "epoch_4_step_130_val_loss=-0.2")
            path3 = os.path.join(temp_dir, "epoch_0_step_10_val_loss=0.12")
            os.mkdir(path1)
            os.mkdir(path2)
            os.mkdir(path3)

            self.assertEqual(
                set(get_checkpoint_dirpaths(temp_dir, metric_name="val_loss")),
                {path1, path2, path3},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                get_checkpoint_dirpaths(temp_dir),
                [],
            )

    def test_checkpoint_sorting_utils(self) -> None:
        """
        Tests the sort utilities
        """
        paths = ["epoch_1_step_20", "epoch_4_step_130", "epoch_0_step_10_val_loss=10"]
        self.assertEqual(_sort_by_recency(paths), [paths[2], paths[0], paths[1]])

        paths = [
            "epoch_1_step_20_val_loss=0.09",
            "epoch_4_step_130_val_loss=29",
            "epoch_0_step_10_val_loss=10",
        ]
        self.assertEqual(
            _sort_by_metric_value(paths, mode="min"), [paths[1], paths[2], paths[0]]
        )
        self.assertEqual(
            _sort_by_metric_value(paths, mode="max"), [paths[0], paths[2], paths[1]]
        )

    def test_delete_checkpoint(self) -> None:
        """
        Tests removing checkpoint directories
        """
        app_state = {"module": nn.Linear(2, 2)}
        with tempfile.TemporaryDirectory() as temp_dir:
            dirpath = os.path.join(temp_dir, "checkpoint")
            Snapshot.take(dirpath, app_state=app_state)
            self.assertTrue(os.path.exists(dirpath))
            # check that error is thrown if .snapshot_metadata is not found in the directory when deleting
            os.remove(os.path.join(dirpath, SNAPSHOT_METADATA_FNAME))
            with self.assertRaisesRegex(
                RuntimeError, f"{temp_dir} does not contain .snapshot_metadata"
            ):
                _delete_checkpoint(temp_dir, SNAPSHOT_METADATA_FNAME)
            _delete_checkpoint(dirpath)
            self.assertFalse(os.path.exists(dirpath))

    def test_metadata_exists(self) -> None:
        app_state = {"module": nn.Linear(2, 2)}
        with tempfile.TemporaryDirectory() as temp_dir:
            dirpath = os.path.join(temp_dir, "checkpoint")
            Snapshot.take(dirpath, app_state=app_state)

            fs = get_filesystem(dirpath)
            self.assertTrue(_metadata_exists(fs, dirpath, SNAPSHOT_METADATA_FNAME))

            os.remove(os.path.join(dirpath, SNAPSHOT_METADATA_FNAME))
            self.assertFalse(_metadata_exists(fs, dirpath, SNAPSHOT_METADATA_FNAME))

    def test_get_app_state(self) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        state = get_dummy_train_state()

        app_state = _prepare_app_state_for_checkpoint(state, my_unit, intra_epoch=False)
        self.assertCountEqual(
            app_state.keys(),
            ["module", "optimizer", "loss_fn", "train_progress"],
        )

    @skip_if_not_distributed
    def test_rank_zero_read_and_broadcast(self) -> None:
        spawn_multi_process(2, "gloo", self._test_rank_zero_read_and_broadcast)

    @staticmethod
    def _test_rank_zero_read_and_broadcast() -> None:
        """
        Tests that rank_zero_read_and_broadcast decorator works as expected
        """

        @rank_zero_read_and_broadcast
        def _test_method_for_rank_zero() -> str:
            assert get_global_rank() == 0
            return "foo"

        init_from_env()
        val_from_test_method = _test_method_for_rank_zero()
        tc = unittest.TestCase()
        tc.assertEqual(val_from_test_method, "foo")
