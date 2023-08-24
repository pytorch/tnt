# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
from typing import Any, Dict, List, Optional, Pattern, Set, Union

import torch.distributed as dist

from pyre_extensions import none_throws
from torchsnapshot.snapshot import PendingSnapshot, Snapshot

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import AppStateMixin, TEvalUnit, TPredictUnit, TTrainUnit
from torchtnt.framework.utils import _construct_tracked_optimizers, get_timing_context
from torchtnt.utils.distributed import get_global_rank, PGWrapper
from torchtnt.utils.fsspec import get_filesystem
from torchtnt.utils.rank_zero_log import rank_zero_info, rank_zero_warn
from torchtnt.utils.stateful import Stateful

try:
    import torchsnapshot

    _TStateful = torchsnapshot.Stateful
    _TORCHSNAPSHOT_AVAILABLE = True
except Exception:
    _TStateful = Stateful
    _TORCHSNAPSHOT_AVAILABLE = False

_EVAL_PROGRESS_STATE_KEY = "eval_progress"
_RNG_STATE_KEY = "rng_state"
_TRAIN_PROGRESS_STATE_KEY = "train_progress"
_TRAIN_DL_STATE_KEY = "train_dataloader"

logger: logging.Logger = logging.getLogger(__name__)


class TorchSnapshotSaver(Callback):
    """
    A callback which periodically saves the application state during training using `TorchSnapshot <https://pytorch.org/torchsnapshot/>`_.

    This callback supplements the application state provided by :class:`torchtnt.unit.AppStateMixin`
    with the train progress, train dataloader (if applicable), and random number generator state.

    If used with :func:`torchtnt.framework.fit`, this class will also save the evaluation progress state.

    Checkpoints will be saved under ``dirpath/epoch_{epoch}_step_{step}`` where step is the *total* number of training steps completed across all epochs.

    Args:
        dirpath: Parent directory to save snapshots to.
        save_every_n_train_steps: Frequency of steps with which to save snapshots during the train epoch. If None, no intra-epoch snapshots are generated.
        save_every_n_epochs: Frequency of epochs with which to save snapshots during training. If None, no end-of-epoch snapshots are generated.
        replicated: A glob-pattern of replicated key names that indicate which application state entries have the same state across all processes.
            For more information, see https://pytorch.org/torchsnapshot/main/api_reference.html#torchsnapshot.Snapshot.take .
        storage_options: storage_options: Additional keyword options for the storage plugin to use, to be passed to `torchsnapshot.Snapshot <https://pytorch.org/torchsnapshot/stable/api_reference.html#torchsnapshot.Snapshot>`_.
            See each storage plugin's documentation for customizations.

    Note: If torch.distributed is available and default process group is initialized, the constructor will call a collective operation for rank 0 to broadcast the dirpath to all other ranks

    Note:
        If checkpointing FSDP model, you can set state_dict type calling `set_state_dict_type <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type>`_ prior to starting training.
    """

    def __init__(
        self,
        dirpath: str,
        *,
        save_every_n_train_steps: Optional[int] = None,
        save_every_n_epochs: Optional[int] = None,
        replicated: Optional[List[str]] = None,
        storage_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        _validate_snapshot_available()
        if save_every_n_train_steps is not None and save_every_n_train_steps <= 0:
            raise ValueError(
                f"Invalid value passed for save_every_n_train_steps. Expected to receive either None or positive number, but received {save_every_n_train_steps}"
            )
        if save_every_n_epochs is not None and save_every_n_epochs <= 0:
            raise ValueError(
                f"Invalid value passed for save_every_n_epochs. Expected to receive either None or positive number, but received {save_every_n_epochs}"
            )
        self._save_every_n_epochs = save_every_n_epochs
        self._save_every_n_train_steps = save_every_n_train_steps
        self._sync_dirpath_to_all_ranks(dirpath)
        self._replicated: Set[str] = set(replicated or [])

        self._prev_snapshot: Optional[PendingSnapshot] = None
        self._storage_options = storage_options

    def _sync_dirpath_to_all_ranks(self, dirpath: str) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            self._dirpath: str = dirpath
            return

        dirpath_container = [dirpath] if get_global_rank() == 0 else [""]
        # broadcast directory from global rank 0
        dist.broadcast_object_list(dirpath_container, src=0)
        updated_dirpath = dirpath_container[0]
        if updated_dirpath != dirpath:
            logger.warning(f"Updating dirpath to match rank 0: {updated_dirpath}")

        self._dirpath: str = updated_dirpath

    @property
    def dirpath(self) -> str:
        """Returns parent directory to save to."""
        return self._dirpath

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        """Validate there's no key collision for the app state."""
        app_state = _app_state(unit)
        _check_app_state_collision(app_state)

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        num_steps_completed = unit.train_progress.num_steps_completed
        save_every_n_train_steps = self._save_every_n_train_steps
        if (
            save_every_n_train_steps is None
            or num_steps_completed % save_every_n_train_steps != 0
        ):
            return

        app_state = _get_app_state(state, unit, self._replicated, intra_epoch=True)
        epoch = unit.train_progress.num_epochs_completed
        snapshot_path = _get_snapshot_save_path(
            self._dirpath, epoch, num_steps_completed
        )
        with get_timing_context(
            state, f"{self.__class__.__name__}.take_async_snapshot"
        ):
            self._async_snapshot(snapshot_path, app_state, wait=False)

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        epoch = unit.train_progress.num_epochs_completed
        save_every_n_epochs = self._save_every_n_epochs
        if save_every_n_epochs is None or epoch % save_every_n_epochs != 0:
            return

        app_state = _get_app_state(state, unit, self._replicated, intra_epoch=False)
        num_steps_completed = unit.train_progress.num_steps_completed
        snapshot_path = _get_snapshot_save_path(
            self._dirpath, epoch, num_steps_completed
        )
        with get_timing_context(
            state, f"{self.__class__.__name__}.take_async_snapshot"
        ):
            self._async_snapshot(snapshot_path, app_state, wait=True)

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        app_state = _get_app_state(state, unit, self._replicated, intra_epoch=False)
        num_steps_completed = unit.train_progress.num_steps_completed
        epoch = unit.train_progress.num_epochs_completed
        snapshot_path = _get_snapshot_save_path(
            self._dirpath, epoch, num_steps_completed
        )
        with get_timing_context(
            state, f"{self.__class__.__name__}.take_async_snapshot"
        ):
            self._async_snapshot(snapshot_path, app_state, wait=True)
            self._wait()

    def on_exception(
        self,
        state: State,
        unit: Union[TTrainUnit, TEvalUnit, TPredictUnit],
        exc: BaseException,
    ) -> None:
        self._wait()

    def _wait(self) -> None:
        if self._prev_snapshot is not None:
            self._prev_snapshot.wait()

    def _async_snapshot(
        self, snapshot_path: str, app_state: Dict[str, _TStateful], *, wait: bool
    ) -> bool:
        prev_snapshot = self._prev_snapshot
        if prev_snapshot is not None:
            if prev_snapshot.path == snapshot_path:
                # Snapshot for this step already has been saved.
                # This can happen if we call _async_snapshot twice at the same step.
                return True
            still_pending = not prev_snapshot.done()
            if still_pending and wait:
                prev_snapshot.wait()
            elif still_pending:
                rank_zero_warn(
                    f"Still writing previous snapshot, will skip this one. Consider increasing 'frequency' (current {self._save_every_n_train_steps})",
                    logger=logger,
                )
                return False

        self._prev_snapshot = Snapshot.async_take(
            str(snapshot_path),
            app_state=app_state,
            replicated=list(self._replicated),
            storage_options=self._storage_options,
        )
        rank_zero_info(f"Saving snapshot to path: {snapshot_path}", logger=logger)
        return True

    @staticmethod
    def restore(
        path: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[_TStateful] = None,
        restore_train_progress: bool = True,
        restore_eval_progress: bool = True,
        storage_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Utility method to restore snapshot state from a path.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            path: Path of the snapshot to restore.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore.
            restore_train_progress: Whether to restore the training progress state.
            restore_eval_progress: Whether to restore the evaluation progress state.
            storage_options: Additional keyword options for the storage plugin to use, to be passed to `torchsnapshot.Snapshot <https://pytorch.org/torchsnapshot/stable/api_reference.html#torchsnapshot.Snapshot>`_. See each storage plugin's documentation for customizations.
        """

        _validate_snapshot_available()
        app_state = _app_state(unit)
        _check_app_state_collision(app_state)

        snapshot = torchsnapshot.Snapshot(path, storage_options=storage_options)

        rng_state = torchsnapshot.RNGState()
        app_state[_RNG_STATE_KEY] = rng_state

        if not restore_train_progress:
            del app_state[_TRAIN_PROGRESS_STATE_KEY]

        if not restore_eval_progress:
            del app_state[_EVAL_PROGRESS_STATE_KEY]

        if train_dataloader is not None:
            # request to restore the dataloader state only if
            # the persisted snapshot state includes the dataloader entry
            manifest = snapshot.get_manifest()
            for key in manifest:
                if _TRAIN_DL_STATE_KEY in key:
                    app_state[_TRAIN_DL_STATE_KEY] = train_dataloader
                    break
            rank_zero_warn(
                "train_dataloader was passed to `restore` but no train dataloader exists in the Snapshot"
            )

        snapshot.restore(app_state)
        rank_zero_info(f"Restored snapshot from path: {path}", logger=logger)

    @staticmethod
    def restore_from_latest(
        dirpath: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[_TStateful] = None,
        restore_train_progress: bool = True,
        restore_eval_progress: bool = True,
        storage_options: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Given a parent directory where checkpoints are saved, restore the snapshot state from the latest checkpoint in the directory.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            dirpath: Parent directory from which to get the latest snapshot.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore.
            restore_train_progress: Whether to restore the training progress state.
            restore_eval_progress: Whether to restore the evaluation progress state.
            storage_options: Additional keyword options for the storage plugin to use, to be passed to `torchsnapshot.Snapshot <https://pytorch.org/torchsnapshot/stable/api_reference.html#torchsnapshot.Snapshot>`_. See each storage plugin's documentation for customizations.

        Returns:
            True if the latest snapshot directory was found and successfully restored, otherwise False.
        """
        path = get_latest_checkpoint_path(dirpath)
        if path is None:
            return False
        TorchSnapshotSaver.restore(
            path,
            unit,
            train_dataloader=train_dataloader,
            restore_train_progress=restore_train_progress,
            restore_eval_progress=restore_eval_progress,
            storage_options=storage_options,
        )
        return True


def get_latest_checkpoint_path(dirpath: str) -> Optional[str]:
    """
    Given a parent directory where checkpoints are saved, return the latest checkpoint subdirectory.

    Args:
        dirpath: parent directory where checkpoints are saved.

    Raises:
        AssertionError if the checkpoint subdirectories are not named in the format epoch_{epoch}_step_{step}.
    """

    ret = None
    rank = get_global_rank()
    # Do all filesystem reads from rank 0 only
    if rank == 0:
        ret = _latest_checkpoint_path(dirpath)

    # If not running in a distributed setting, return as is
    if not (dist.is_available() and dist.is_initialized()):
        return ret

    # Otherwise, broadcast result from rank 0 to all ranks
    pg = PGWrapper(dist.group.WORLD)
    path_container = [ret] if rank == 0 else [None]
    pg.broadcast_object_list(path_container, 0)
    val = path_container[0]
    return val


def _latest_checkpoint_path(dirpath: str) -> Optional[str]:
    fs = get_filesystem(dirpath)

    if not fs.exists(dirpath):
        logger.warning(f"Input dirpath doesn't exist: {dirpath}")
        return None

    contents = fs.ls(dirpath, detail=True)
    contents = [item["name"] for item in contents if item["type"] == "directory"]
    if len(contents) == 0:
        logger.warning(f"Input dirpath doesn't contain any subdirectories: {dirpath}")
        return None

    # Define the regex pattern to match the directory names
    pattern = rf"^{dirpath}/epoch_\d+_step_\d+"
    snapshot_dirpath_pattern: Pattern[str] = re.compile(pattern)
    candidate_dirpaths = list(filter(snapshot_dirpath_pattern.match, contents))

    if len(candidate_dirpaths) == 0:
        logger.warning(
            f"No valid checkpoint directories were found in input dirpath: {dirpath}"
        )
        return None

    # Initialize variables to store the largest epoch and step numbers
    largest_subdirectory = None
    largest_epoch = -1
    largest_step = -1

    # Iterate through all files and directories in the specified directory
    for candidate in candidate_dirpaths:
        # Extract the epoch and step numbers from the directory name
        dirname = os.path.basename(candidate)

        # dirname will be of the format epoch_N_step_M
        # where N is the epoch number and M is the step number as integers
        split = dirname.split("_")
        if len(split) != 4:
            raise AssertionError(
                f"Expected exactly 4 elements for pattern of epoch_N_step_M, but received {split})"
            )

        epoch_num, step_num = int(split[1]), int(split[3])
        # Check if the current epoch and step numbers are larger than the largest ones found so far
        if epoch_num > largest_epoch:
            largest_epoch = epoch_num
            largest_step = step_num
            largest_subdirectory = dirname
        elif largest_epoch == epoch_num and step_num > largest_step:
            largest_step = step_num
            largest_subdirectory = dirname

    # Rejoin with the parent directory path and return the largest subdirectory
    return os.path.join(dirpath, none_throws(largest_subdirectory))


def _validate_snapshot_available() -> None:
    if not _TORCHSNAPSHOT_AVAILABLE:
        raise RuntimeError(
            "TorchSnapshotSaver support requires torchsnapshot. "
            "Please make sure ``torchsnapshot`` is installed. "
            "Installation: https://github.com/pytorch/torchsnapshot#install"
        )


def _get_snapshot_save_path(dirpath: str, epoch: int, step: int) -> str:
    # TODO: discuss whether this path should be customized
    return os.path.join(dirpath, f"epoch_{epoch}_step_{step}")


def _app_state(unit: AppStateMixin) -> Dict[str, Any]:
    """Join together all of the tracked stateful entities to simplify registration of snapshottable states, deals with FSDP case"""
    app_state = unit.app_state()
    tracked_optimizers = _construct_tracked_optimizers(unit)  # handles fsdp
    app_state.update(tracked_optimizers)
    return app_state


def _get_app_state(
    state: State, unit: AppStateMixin, replicated: Set[str], *, intra_epoch: bool
) -> Dict[str, _TStateful]:
    train_state = none_throws(state.train_state)
    app_state = _app_state(unit)

    rng_state = torchsnapshot.RNGState()
    app_state[_RNG_STATE_KEY] = rng_state

    # for intra-epoch checkpointing, include dataloader states
    train_dl = train_state.dataloader
    if intra_epoch and isinstance(train_dl, _TStateful):
        app_state[_TRAIN_DL_STATE_KEY] = train_dl

    # add progress to replicated
    train_prog_glob = f"{_TRAIN_PROGRESS_STATE_KEY}/*"
    replicated.add(train_prog_glob)

    if state.entry_point == EntryPoint.FIT:
        eval_prog_glob = f"{_EVAL_PROGRESS_STATE_KEY}/*"
        replicated.add(eval_prog_glob)

    return app_state


def _check_app_state_collision(app_state: Dict[str, _TStateful]) -> None:
    keys_to_check = (
        _TRAIN_DL_STATE_KEY,
        _RNG_STATE_KEY,
    )
    for key in keys_to_check:
        if key in app_state:
            raise RuntimeError(
                f"Detected collision for key in app state: {key}. TorchSnapshotSaver expects to save and load this key."
            )
