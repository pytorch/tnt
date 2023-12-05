# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
import os
from typing import Any, cast, Iterable, List, Optional

import torch.distributed as dist

from torchtnt.framework.callback import Callback
from torchtnt.framework.callbacks._checkpoint_utils import (
    _delete_checkpoint,
    _retrieve_checkpoint_dirpaths,
    get_latest_checkpoint_path,
)
from torchtnt.framework.callbacks.checkpointer_types import RestoreOptions
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import AppStateMixin, TEvalUnit, TTrainData, TTrainUnit
from torchtnt.framework.utils import get_timing_context
from torchtnt.utils.distributed import PGWrapper
from torchtnt.utils.fsspec import get_filesystem
from torchtnt.utils.rank_zero_log import rank_zero_warn

logger: logging.Logger = logging.getLogger(__name__)


class BaseCheckpointer(Callback, metaclass=abc.ABCMeta):
    """
    Abstract base class for file-based state_dict checkpointing. This class can be used as the base of a checkpointing callback, and handles
    checkpointing frequency logic, checkpoint naming, checkpoint purging / upkeep, and process group synchronization. There are only two methods
    that need to be implemented by subclasses:

    1) ``_checkpoint_impl`` which implements the checkpoint saving logic, given the relevant checkpoint items and path.
    2) ``restore`` which implements restoring the checkpoint given the relevant checkpoint path.

    The subclass may override the ``metadata_fname`` attribute to specify the filename of the metadata file that will be written within the checkpoint directory.
    This will be used by this base class to ensure the integrity of the checkpoint.

    Args:
        dirpath: Parent directory to save checkpoints to.
        save_every_n_train_steps: Frequency of steps with which to save checkpoints during the train epoch. If None, no intra-epoch checkpoints are generated.
        save_every_n_epochs: Frequency of epochs with which to save checkpoints during training. If None, no end-of-epoch checkpoints are generated.
        keep_last_n_checkpoints: Number of most recent checkpoints to keep. If None, all checkpoints are kept. If an excess of existing checkpoints are present, the oldest ones will be deleted to clean the difference.
        process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)

    Note:
        If torch.distributed is available and default process group is initialized, the constructor will call a collective operation for rank 0 to broadcast the dirpath to all other ranks

     Note:
        This class assumes checkpoint items are saved in the directory provided in ``_checkpoint_impl`` and will be in the form of ``<dirpath>/<epoch>-<step>/``. Checkpoint contents
        should be stored within this directory, as deleting and retrieving latest checkpoint relies on reading the <epoch>-<step> directory name within <dirpath>
    """

    metadata_fname: Optional[str] = None

    def __init__(
        self,
        dirpath: str,
        *,
        save_every_n_train_steps: Optional[int] = None,
        save_every_n_epochs: Optional[int] = None,
        keep_last_n_checkpoints: Optional[int] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        if save_every_n_train_steps is not None and save_every_n_train_steps <= 0:
            raise ValueError(
                f"Invalid value passed for save_every_n_train_steps. Expected to receive either None or positive number, but received {save_every_n_train_steps}"
            )
        if save_every_n_epochs is not None and save_every_n_epochs <= 0:
            raise ValueError(
                f"Invalid value passed for save_every_n_epochs. Expected to receive either None or positive number, but received {save_every_n_epochs}"
            )
        if keep_last_n_checkpoints is not None and keep_last_n_checkpoints <= 0:
            raise ValueError(
                f"Invalid value passed for keep_last_n_checkpoints. Expected to receive either None or positive number, but received {keep_last_n_checkpoints}"
            )

        self._save_every_n_train_steps = save_every_n_train_steps
        self._save_every_n_epochs = save_every_n_epochs
        self._keep_last_n_checkpoints = keep_last_n_checkpoints
        self._ckpt_dirpaths: List[str] = []
        if self._keep_last_n_checkpoints:
            self._ckpt_dirpaths = _retrieve_checkpoint_dirpaths(dirpath)
        self._process_group = process_group
        self._pg_wrapper = PGWrapper(process_group)

        # sync dirpaths from rank 0
        self._sync_dirpath_to_all_ranks(dirpath)

    def _sync_dirpath_to_all_ranks(self, dirpath: str) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            self._dirpath: str = dirpath
            return

        dirpath_container = [dirpath] if self._pg_wrapper.get_rank() == 0 else [""]
        # broadcast directory from global rank 0
        dist.broadcast_object_list(dirpath_container, src=0, group=self._process_group)
        updated_dirpath = dirpath_container[0]
        if updated_dirpath != dirpath:
            logger.warning(f"Updating dirpath to match rank 0: {updated_dirpath}")

        self._dirpath: str = updated_dirpath

    @property
    def dirpath(self) -> str:
        """Returns parent directory to save to."""
        return self._dirpath

    def _generate_checkpoint_and_upkeep(
        self, state: State, unit: TTrainUnit, hook: str
    ) -> bool:
        """
        Implementation for saving checkpoint while taking care of checkpoint
        name generation and cleanup of oldest checkpoints.

        Args:
            state: Current state of the trainer.
            unit: Current training unit.
            hook: Hook at which checkpoint is being saved.

        Returns:
            True if checkpoint was successfully saved. False otherwise.
        """
        # 1) generate checkpoint name
        num_steps_completed = unit.train_progress.num_steps_completed
        if state.entry_point == EntryPoint.FIT:
            num_steps_completed += cast(
                TEvalUnit, unit
            ).eval_progress.num_steps_completed
        epoch = unit.train_progress.num_epochs_completed
        checkpoint_path = _get_save_path(self._dirpath, epoch, num_steps_completed)

        # 1.5) If metadata_fname is set, ensure metadata file doesn't exist on final checkpoint
        if hook == "on_train_end" and self.metadata_fname:
            metadata_filepath = os.path.join(checkpoint_path, self.metadata_fname)
            fs = get_filesystem(metadata_filepath)
            if fs.exists(metadata_filepath):
                rank_zero_warn(
                    "Final checkpoint already exists, skipping.", logger=logger
                )
                return False

        # 2) save checkpoint
        success = self._checkpoint_impl(
            state,
            unit,
            checkpoint_path=checkpoint_path,
            hook=hook,
        )

        # 3) book keep and remove oldest checkpoints
        if success:
            if self._should_remove_checkpoint():
                self._remove_checkpoint(state)
            self._ckpt_dirpaths.append(checkpoint_path)

        return success

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        # clean up the difference if surplus of checkpoints exist
        keep_last_n_checkpoints = self._keep_last_n_checkpoints
        if (
            keep_last_n_checkpoints
            and len(self._ckpt_dirpaths) > keep_last_n_checkpoints
        ):
            logger.warning(
                " ".join(
                    [
                        f"{len(self._ckpt_dirpaths)} checkpoints found in {self._dirpath}.",
                        f"Deleting {len(self._ckpt_dirpaths) - keep_last_n_checkpoints} oldest",
                        "checkpoints to enforce ``keep_last_n_checkpoints`` argument.",
                    ]
                )
            )
            for _ in range(len(self._ckpt_dirpaths) - keep_last_n_checkpoints):
                self._remove_checkpoint(state)

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        num_steps_completed = unit.train_progress.num_steps_completed
        save_every_n_train_steps = self._save_every_n_train_steps
        if (
            save_every_n_train_steps is None
            or num_steps_completed % save_every_n_train_steps != 0
        ):
            return

        self._generate_checkpoint_and_upkeep(state, unit, hook="on_train_step_end")

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        epoch = unit.train_progress.num_epochs_completed
        save_every_n_epochs = self._save_every_n_epochs
        if save_every_n_epochs is None or epoch % save_every_n_epochs != 0:
            return

        self._generate_checkpoint_and_upkeep(state, unit, hook="on_train_epoch_end")

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        self._generate_checkpoint_and_upkeep(state, unit, hook="on_train_end")

    @abc.abstractmethod
    def _checkpoint_impl(
        self,
        state: State,
        unit: AppStateMixin,
        *,
        checkpoint_path: str,
        hook: str,
    ) -> bool:
        """
        Implementation of saving checkpoint.

        Args:
            state: current application state
            unit: current unit
            checkpoint_path: path to save checkpoint
            hook: name of callback hook that triggered this function call

        Returns:
            Whether a new checkpoint was created.
        """
        ...

    def _should_remove_checkpoint(self) -> bool:
        keep_last_n_checkpoints = self._keep_last_n_checkpoints
        return (
            keep_last_n_checkpoints is not None
            and len(self._ckpt_dirpaths) >= keep_last_n_checkpoints
        )

    def _remove_checkpoint(self, state: State) -> None:
        # remove oldest checkpoint directory
        oldest_ckpt_path = self._ckpt_dirpaths.pop(0)
        with get_timing_context(state, f"{self.__class__.__name__}.delete_checkpoint"):
            if self._pg_wrapper.get_rank() == 0:
                # only delete on rank 0
                _delete_checkpoint(oldest_ckpt_path)
            self._pg_wrapper.barrier()

    @staticmethod
    @abc.abstractmethod
    def restore(
        path: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
    ) -> None:
        """Method to restore checkpoint state from a path.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            path: Path of the checkpoint to restore.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
            restore_options: Controls what to filter when restoring the state.
        """
        ...

    @classmethod
    def restore_from_latest(
        cls,
        dirpath: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Given a parent directory where checkpoints are saved, restore the checkppoint state from the latest checkpoint in the directory.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            dirpath: Parent directory from which to get the latest checkpoint.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
            restore_options: Controls what to  filter when restoring the state.

        Returns:
            True if the latest checkpoint directory was found and successfully restored, otherwise False.
        """
        path = get_latest_checkpoint_path(
            dirpath, metadata_fname=cls.metadata_fname, process_group=process_group
        )
        if path is None:
            return False
        logger.info(f"Restoring from path: {path}")
        cls.restore(
            path,
            unit,
            train_dataloader=train_dataloader,
            process_group=process_group,
            restore_options=restore_options,
            **kwargs,
        )
        return True


def _get_save_path(dirpath: str, epoch: int, step: int) -> str:
    # TODO: discuss whether this path should be customized
    return os.path.join(dirpath, f"epoch_{epoch}_step_{step}")
