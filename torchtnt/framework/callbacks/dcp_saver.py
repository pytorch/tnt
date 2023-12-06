# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable, Optional

import torch.distributed as dist

from torch.distributed import checkpoint as dist_cp
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter

from torchtnt.framework.callbacks._checkpoint_utils import (
    _prepare_app_state_for_checkpoint,
    _prepare_app_state_for_restore,
    _TRAIN_DL_STATE_KEY,
)

from torchtnt.framework.callbacks.base_checkpointer import BaseCheckpointer
from torchtnt.framework.callbacks.checkpointer_types import RestoreOptions
from torchtnt.framework.state import State
from torchtnt.framework.unit import AppStateMixin, TTrainData
from torchtnt.framework.utils import get_timing_context
from torchtnt.utils.rank_zero_log import rank_zero_info, rank_zero_warn
from torchtnt.utils.stateful import MultiStateful, Stateful


logger: logging.Logger = logging.getLogger(__name__)


class DistributedCheckpointSaver(BaseCheckpointer):
    """
    A callback which periodically saves the application state during training using `Distributed Checkpoint <https://pytorch.org/docs/stable/distributed.checkpoint.html/>`_.

    This callback supplements the application state provided by :class:`torchtnt.unit.AppStateMixin`
    with the train progress, and train dataloader (if applicable).

    If used with :func:`torchtnt.framework.fit`, this class will also save the evaluation progress state.

    Checkpoints will be saved under ``dirpath/epoch_{epoch}_step_{step}`` where step is the *total* number of training steps completed across all epochs.

    Args:
        dirpath: Parent directory to save snapshots to.
        save_every_n_train_steps: Frequency of steps with which to save snapshots during the train epoch. If None, no intra-epoch snapshots are generated.
        save_every_n_epochs: Frequency of epochs with which to save snapshots during training. If None, no end-of-epoch snapshots are generated.
        keep_last_n_checkpoints: Number of most recent checkpoints to keep. If None, all checkpoints are kept. If an excess of existing checkpoints are present, the oldest ones will be deleted to clean the difference.
        process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)

    Note:
        If torch.distributed is available and default process group is initialized, dcp's `no_dist <https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.load_state_dict/>_`
        argument is automatically set to False. Otherwise it's set to True.

    Note:
        If checkpointing FSDP model, you can set state_dict type calling `set_state_dict_type <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type>`_ prior to starting training.
    """

    def __init__(
        self,
        dirpath: str,
        *,
        save_every_n_train_steps: Optional[int] = None,
        save_every_n_epochs: Optional[int] = None,
        keep_last_n_checkpoints: Optional[int] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__(
            dirpath=dirpath,
            save_every_n_train_steps=save_every_n_train_steps,
            save_every_n_epochs=save_every_n_epochs,
            keep_last_n_checkpoints=keep_last_n_checkpoints,
            process_group=process_group,
        )

    def _checkpoint_impl(
        self,
        state: State,
        unit: AppStateMixin,
        *,
        checkpoint_path: str,
        hook: str,
    ) -> bool:
        intra_epoch = False
        if hook == "on_train_step_end":
            intra_epoch = True

        storage_writer = FsspecWriter(checkpoint_path)

        app_state = _prepare_app_state_for_checkpoint(state, unit, intra_epoch)
        # flag to indicate whether distributed is available
        # determines what to set ``no_dist`` arg in DCP apis
        pg_available: bool = dist.is_initialized()
        with get_timing_context(state, f"{self.__class__.__name__}.save_state_dict"):
            dist_cp.save_state_dict(
                {"app_state": MultiStateful(app_state).state_dict()},
                storage_writer=storage_writer,
                process_group=self._process_group,
                no_dist=not pg_available,
            )
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
        """Utility method to restore dcp checkpoint from a path.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            path: Path of the snapshot to restore.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
            restore_options: Controls what to  filter when restoring the state.
            no_dist: Set to true if loading in non-distributed setting
        """
        storage_reader = FsspecReader(path)

        restore_options = restore_options or RestoreOptions()
        app_state = _prepare_app_state_for_restore(unit, restore_options)

        if train_dataloader is not None:
            if not isinstance(train_dataloader, Stateful):
                rank_zero_warn(
                    "train_dataloader was passed to `restore` but the dataloader does not implement the Stateful protocol to load states"
                )
            else:
                # request to restore the dataloader state only if
                # the persisted snapshot state includes the dataloader entry
                metadata = storage_reader.read_metadata()
                for key in metadata.state_dict_metadata.keys():
                    if _TRAIN_DL_STATE_KEY in key:
                        app_state[_TRAIN_DL_STATE_KEY] = train_dataloader
                        break

                if _TRAIN_DL_STATE_KEY not in app_state:
                    rank_zero_warn(
                        "train_dataloader was passed to `restore` but no train dataloader exists in the Snapshot"
                    )

        state_dict = {"app_state": MultiStateful(app_state).state_dict()}
        no_dist = not dist.is_initialized()
        dist_cp.load_state_dict(
            state_dict,
            storage_reader=storage_reader,
            process_group=process_group,
            no_dist=no_dist,
        )
        MultiStateful(app_state).load_state_dict(state_dict["app_state"])
        rank_zero_info(f"Restored snapshot from path: {path}", logger=logger)
