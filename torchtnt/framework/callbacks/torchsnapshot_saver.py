# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
from contextlib import contextmanager, ExitStack
from typing import (
    Any,
    cast,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Pattern,
    Set,
    Union,
)

import torch.distributed as dist

from pyre_extensions import none_throws

from torchtnt.framework.callback import Callback
from torchtnt.framework.callbacks.checkpointer_types import KnobOptions, RestoreOptions
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import (
    AppStateMixin,
    TEvalUnit,
    TPredictUnit,
    TTrainData,
    TTrainUnit,
)
from torchtnt.framework.utils import get_timing_context
from torchtnt.utils.distributed import get_global_rank, PGWrapper
from torchtnt.utils.fsspec import get_filesystem
from torchtnt.utils.optimizer import init_optim_state
from torchtnt.utils.rank_zero_log import rank_zero_info, rank_zero_warn
from torchtnt.utils.stateful import Stateful

try:
    import torchsnapshot
    from torchsnapshot.knobs import override_max_per_rank_io_concurrency
    from torchsnapshot.snapshot import (
        PendingSnapshot,
        Snapshot,
        SNAPSHOT_METADATA_FNAME,
    )

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
        keep_last_n_checkpoints: Number of most recent checkpoints to keep. If None, all checkpoints are kept. If an excess of existing checkpoints are present, the oldest ones will be deleted to clean the difference.
        process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
        replicated: A glob-pattern of replicated key names that indicate which application state entries have the same state across all processes.
            For more information, see https://pytorch.org/torchsnapshot/main/api_reference.html#torchsnapshot.Snapshot.take.

            .. warning:: The replication property is safer to not set, and should only be used if really needed.
                         Things like metrics, grad_scalers, etc should not be marked as replicated as they
                         may contain different values across processes. If unsure, leave this field unset.

        storage_options: Additional keyword options for the storage plugin to use, to be passed to `torchsnapshot.Snapshot <https://pytorch.org/torchsnapshot/stable/api_reference.html#torchsnapshot.Snapshot>`_.
            See each storage plugin's documentation for customizations.
        knob_options: Additional keyword options for the snapshot knobs

    Note:
        If torch.distributed is available and default process group is initialized, the constructor will call a collective operation for rank 0 to broadcast the dirpath to all other ranks

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
        replicated: Optional[List[str]] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        knob_options: Optional[KnobOptions] = None,
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
        if keep_last_n_checkpoints is not None and keep_last_n_checkpoints <= 0:
            raise ValueError(
                f"Invalid value passed for keep_last_n_checkpoints. Expected to receive either None or positive number, but received {keep_last_n_checkpoints}"
            )
        self._save_every_n_epochs = save_every_n_epochs
        self._save_every_n_train_steps = save_every_n_train_steps

        self._keep_last_n_checkpoints = keep_last_n_checkpoints
        self._ckpt_dirpaths: List[str] = []
        if self._keep_last_n_checkpoints:
            self._ckpt_dirpaths = _retrieve_checkpoint_dirpaths(dirpath)

        self._process_group = process_group
        self._pg_wrapper = PGWrapper(process_group)
        self._sync_dirpath_to_all_ranks(dirpath)
        self._replicated: Set[str] = set(replicated or [])

        self._prev_snapshot: Optional[PendingSnapshot] = None
        self._storage_options = storage_options
        self._knob_options: KnobOptions = knob_options or KnobOptions()

    def _sync_dirpath_to_all_ranks(self, dirpath: str) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            self._dirpath: str = dirpath
            return

        dirpath_container = [dirpath] if get_global_rank() == 0 else [""]
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

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        """Validate there's no key collision for the app state."""
        app_state = _app_state(unit)
        _check_app_state_collision(app_state)

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
                self._remove_snapshot(state)

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        num_steps_completed = unit.train_progress.num_steps_completed
        save_every_n_train_steps = self._save_every_n_train_steps
        if (
            save_every_n_train_steps is None
            or num_steps_completed % save_every_n_train_steps != 0
        ):
            return

        epoch = unit.train_progress.num_epochs_completed
        if state.entry_point == EntryPoint.FIT:
            num_steps_completed += cast(
                TEvalUnit, unit
            ).eval_progress.num_steps_completed
        snapshot_path = _get_snapshot_save_path(
            self._dirpath, epoch, num_steps_completed
        )
        self._checkpoint_impl(
            state,
            unit,
            snapshot_path=snapshot_path,
            intra_epoch=True,
            prev_snapshot_wait=False,
            curr_snapshot_wait=False,
        )

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        epoch = unit.train_progress.num_epochs_completed
        save_every_n_epochs = self._save_every_n_epochs
        if save_every_n_epochs is None or epoch % save_every_n_epochs != 0:
            return

        num_steps_completed = unit.train_progress.num_steps_completed
        if state.entry_point == EntryPoint.FIT:
            num_steps_completed += cast(
                TEvalUnit, unit
            ).eval_progress.num_steps_completed
        snapshot_path = _get_snapshot_save_path(
            self._dirpath, epoch, num_steps_completed
        )
        self._checkpoint_impl(
            state,
            unit,
            snapshot_path=snapshot_path,
            intra_epoch=False,
            prev_snapshot_wait=True,
            curr_snapshot_wait=False,
        )

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        num_steps_completed = unit.train_progress.num_steps_completed
        if state.entry_point == EntryPoint.FIT:
            num_steps_completed += cast(
                TEvalUnit, unit
            ).eval_progress.num_steps_completed
        epoch = unit.train_progress.num_epochs_completed
        snapshot_path = _get_snapshot_save_path(
            self._dirpath, epoch, num_steps_completed
        )

        fs = get_filesystem(snapshot_path)
        if fs.exists(snapshot_path):
            if fs.exists(os.path.join(snapshot_path, SNAPSHOT_METADATA_FNAME)):
                rank_zero_warn(
                    "Final checkpoint already exists, skipping.", logger=logger
                )
                return

        self._checkpoint_impl(
            state,
            unit,
            snapshot_path=snapshot_path,
            intra_epoch=False,
            prev_snapshot_wait=True,
            curr_snapshot_wait=True,
        )

    def on_exception(
        self,
        state: State,
        unit: Union[TTrainUnit, TEvalUnit, TPredictUnit],
        exc: BaseException,
    ) -> None:
        self._wait()

    def _checkpoint_impl(
        self,
        state: State,
        unit: AppStateMixin,
        *,
        snapshot_path: str,
        intra_epoch: bool,
        prev_snapshot_wait: bool,
        curr_snapshot_wait: bool,
    ) -> None:
        """
        Checkpoint the current state of the application.

        Args:
            state: State of the application
            unit: The training/evaluation/prediction unit
            snapshot_path: Path to save the snapshot
            intra_epoch: Whether in middle of epoch or not
            prev_snapshot_wait: Whether to wait for previous snapshot to finish writing
            curr_snapshot_wait: Whether to wait for current snapshot to finish writing
        """
        app_state = _get_app_state(
            state,
            unit,
            intra_epoch=intra_epoch,
        )
        with get_timing_context(
            state, f"{self.__class__.__name__}.take_async_snapshot"
        ):
            # TODO checkpoint is not truly successful
            # since this is async checkpointed, so in
            # future, add logic to set  successful flag
            # only when checkpoint is fully written
            checkpoint_success = self._async_snapshot(
                snapshot_path, app_state, wait=prev_snapshot_wait
            )
            if curr_snapshot_wait:
                self._wait()

        # remove and book keep snapshots related to keep_last_n_checkpoints
        if checkpoint_success:
            if self._should_remove_snapshot():
                self._remove_snapshot(state)
            self._ckpt_dirpaths.append(snapshot_path)

    def _should_remove_snapshot(self) -> bool:
        keep_last_n_checkpoints = self._keep_last_n_checkpoints
        return (
            keep_last_n_checkpoints is not None
            and len(self._ckpt_dirpaths) >= keep_last_n_checkpoints
        )

    def _remove_snapshot(self, state: State) -> None:
        # remove oldest snapshot directory
        oldest_ckpt_path = self._ckpt_dirpaths.pop(0)
        with get_timing_context(state, f"{self.__class__.__name__}.delete_snapshot"):
            if self._pg_wrapper.get_rank() == 0:
                # only delete on rank 0
                _delete_snapshot(oldest_ckpt_path)
            self._pg_wrapper.barrier()

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
                return False
            still_pending = not prev_snapshot.done()
            if still_pending and wait:
                prev_snapshot.wait()
            elif still_pending:
                rank_zero_warn(
                    f"Still writing previous snapshot, will skip this one. Consider increasing 'frequency' (current {self._save_every_n_train_steps})",
                    logger=logger,
                )
                return False

        with _override_knobs(self._knob_options):
            self._prev_snapshot = Snapshot.async_take(
                str(snapshot_path),
                app_state=app_state,
                pg=self._process_group,
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
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        knob_options: Optional[KnobOptions] = None,
    ) -> None:
        """Utility method to restore snapshot state from a path.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            path: Path of the snapshot to restore.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
            restore_options: Controls what to  filter when restoring the state.
            storage_options: Additional keyword options for the storage plugin to use, to be passed to `torchsnapshot.Snapshot <https://pytorch.org/torchsnapshot/stable/api_reference.html#torchsnapshot.Snapshot>`_. See each storage plugin's documentation for customizations.
            knob_options: Additional keyword options for the snapshot knobs
        """

        _validate_snapshot_available()

        # initialize optimizer state skeletons for in-place loading of optimizer state with torchsnapshot
        for optimizer in unit.tracked_optimizers().values():
            init_optim_state(optimizer)

        app_state = _app_state(unit)
        _check_app_state_collision(app_state)

        snapshot = torchsnapshot.Snapshot(
            path, pg=process_group, storage_options=storage_options
        )

        rng_state = torchsnapshot.RNGState()
        app_state[_RNG_STATE_KEY] = rng_state

        restore_options = restore_options or RestoreOptions()
        if not restore_options.restore_train_progress:
            app_state.pop(_TRAIN_PROGRESS_STATE_KEY, None)

        if not restore_options.restore_eval_progress:
            app_state.pop(_EVAL_PROGRESS_STATE_KEY, None)

        if not restore_options.restore_optimizers:
            # remove all optimizer keys from app_state
            for optim_keys in unit.tracked_optimizers().keys():
                app_state.pop(optim_keys, None)

        if not restore_options.restore_lr_schedulers:
            # remove all lr scheduler keys from app_state
            for lr_scheduler_keys in unit.tracked_lr_schedulers().keys():
                app_state.pop(lr_scheduler_keys, None)

        if train_dataloader is not None:
            if not isinstance(train_dataloader, _TStateful):
                rank_zero_warn(
                    "train_dataloader was passed to `restore` but the dataloader does not implement the Stateful protocol to load states"
                )
            else:
                # request to restore the dataloader state only if
                # the persisted snapshot state includes the dataloader entry
                manifest = snapshot.get_manifest()
                for key in manifest:
                    if _TRAIN_DL_STATE_KEY in key:
                        app_state[_TRAIN_DL_STATE_KEY] = train_dataloader
                        break

                if _TRAIN_DL_STATE_KEY not in app_state:
                    rank_zero_warn(
                        "train_dataloader was passed to `restore` but no train dataloader exists in the Snapshot"
                    )

        knob_options = knob_options or KnobOptions()
        with _override_knobs(knob_options):
            snapshot.restore(app_state)
        rank_zero_info(f"Restored snapshot from path: {path}", logger=logger)

    @staticmethod
    def restore_from_latest(
        dirpath: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        knob_options: Optional[KnobOptions] = None,
    ) -> bool:
        """
        Given a parent directory where checkpoints are saved, restore the snapshot state from the latest checkpoint in the directory.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            dirpath: Parent directory from which to get the latest snapshot.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
            restore_options: Controls what to  filter when restoring the state.
            storage_options: Additional keyword options for the storage plugin to use, to be passed to `torchsnapshot.Snapshot <https://pytorch.org/torchsnapshot/stable/api_reference.html#torchsnapshot.Snapshot>`_. See each storage plugin's documentation for customizations.
            knob_options: Additional keyword options for the snapshot knobs

        Returns:
            True if the latest snapshot directory was found and successfully restored, otherwise False.
        """
        path = get_latest_checkpoint_path(dirpath, process_group=process_group)
        if path is None:
            return False
        logger.info(f"Restoring from path: {path}")
        TorchSnapshotSaver.restore(
            path,
            unit,
            train_dataloader=train_dataloader,
            process_group=process_group,
            restore_options=restore_options,
            storage_options=storage_options,
            knob_options=knob_options,
        )
        return True


def get_latest_checkpoint_path(
    dirpath: str, process_group: Optional[dist.ProcessGroup] = None
) -> Optional[str]:
    """
    Given a parent directory where checkpoints are saved, return the latest checkpoint subdirectory.

    Args:
        dirpath: parent directory where checkpoints are saved.
        process_group: the process group on which the ranks will communicate on. default: ``None`` (the entire world)

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
    pg = PGWrapper(process_group)
    path_container = [ret] if rank == 0 else [None]
    pg.broadcast_object_list(path_container, 0)
    val = path_container[0]
    return val


def _latest_checkpoint_path(dirpath: str) -> Optional[str]:
    if dirpath[-1] == "/":
        # removes trailing forward slash if present
        # required for regex search to work
        dirpath = dirpath[:-1]

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
        dir_contents = fs.ls(candidate, False)
        if not any(
            SNAPSHOT_METADATA_FNAME == os.path.basename(f) for f in dir_contents
        ):
            logger.warning(
                f"Snapshot metadata is missing from {candidate}! Skipping this path"
            )
            continue

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
    tracked_optimizers = unit._construct_tracked_optimizers()  # handles fsdp
    app_state.update(tracked_optimizers)
    return app_state


def _get_app_state(
    state: State,
    unit: AppStateMixin,
    *,
    intra_epoch: bool,
) -> Dict[str, _TStateful]:
    train_state = none_throws(state.train_state)
    app_state = _app_state(unit)

    rng_state = torchsnapshot.RNGState()
    app_state[_RNG_STATE_KEY] = rng_state

    # for intra-epoch checkpointing, include dataloader states
    train_dl = train_state.dataloader
    if intra_epoch and isinstance(train_dl, _TStateful):
        app_state[_TRAIN_DL_STATE_KEY] = train_dl

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


@contextmanager
def _override_knobs(
    knob_options: KnobOptions,
) -> Generator[None, None, None]:
    knobs = []
    if knob_options.max_per_rank_io_concurrency:
        knobs.append(
            override_max_per_rank_io_concurrency(
                knob_options.max_per_rank_io_concurrency
            ),
        )

    with ExitStack() as stack:
        for mgr in knobs:
            stack.enter_context(mgr)
        yield


def _retrieve_checkpoint_dirpaths(dirpath: str) -> List[str]:
    """
    Given a parent directory where checkpoints are saved, return the sorted checkpoint subdirectories
    from oldest to newest.

    Args:
        dirpath: parent directory where checkpoints are saved.
    """
    fs = get_filesystem(dirpath)

    contents = fs.ls(dirpath, detail=True)
    contents = [item["name"] for item in contents if item["type"] == "directory"]
    ckpt_dirpaths = []
    for path in contents:
        match = re.search(r"epoch_(\d+)_step_(\d+)", path)
        if match:
            ckpt_dirpaths.append(path)

    # sorts by epoch, then step
    ckpt_dirpaths.sort(key=lambda x: (int(x.split("_")[1]), int(x.split("_")[3])))
    return ckpt_dirpaths


def _delete_snapshot(dirpath: str) -> None:
    fs = get_filesystem(dirpath)
    if not fs.exists(os.path.join(dirpath, ".snapshot_metadata")):
        raise RuntimeError(f"{dirpath} does not contain .snapshot_metadata")
    fs.rm(dirpath, recursive=True)
