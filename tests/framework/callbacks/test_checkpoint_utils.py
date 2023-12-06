# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import launcher
from torchsnapshot import Snapshot
from torchsnapshot.snapshot import SNAPSHOT_METADATA_FNAME
from torchtnt.framework._test_utils import DummyTrainUnit, get_dummy_train_state

from torchtnt.framework.callbacks._checkpoint_utils import (
    _delete_checkpoint,
    _prepare_app_state_for_checkpoint,
    _retrieve_checkpoint_dirpaths,
    get_latest_checkpoint_path,
)
from torchtnt.utils.distributed import get_global_rank, PGWrapper
from torchtnt.utils.test_utils import get_pet_launch_config

METADATA_FNAME: str = ".metadata"


class CheckpointUtilsTest(unittest.TestCase):
    distributed_available: bool = torch.distributed.is_available()

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
            path_2 = os.path.join(temp_dir, "epoch_0_step_100")
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

    @patch("torchtnt.framework.callbacks._checkpoint_utils.get_filesystem")
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

    def test_get_app_state(self) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        state = get_dummy_train_state()

        app_state = _prepare_app_state_for_checkpoint(state, my_unit, intra_epoch=False)
        self.assertCountEqual(
            app_state.keys(),
            ["module", "optimizer", "loss_fn", "train_progress"],
        )
