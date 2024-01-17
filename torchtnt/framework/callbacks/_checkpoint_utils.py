# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re

from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Literal,
    Optional,
    Pattern,
    Tuple,
    TypeVar,
)

import fsspec

from pyre_extensions import none_throws
from torch import distributed as dist
from torchtnt.framework.callbacks.checkpointer_types import RestoreOptions
from torchtnt.framework.state import State
from torchtnt.framework.unit import AppStateMixin
from torchtnt.utils.distributed import get_global_rank, PGWrapper

from torchtnt.utils.fsspec import get_filesystem
from torchtnt.utils.stateful import Stateful

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


def rank_zero_read_and_broadcast(
    func: Callable[..., T],
) -> Callable[..., T]:
    def wrapper(*args: Any, **kwargs: Any) -> T:
        ret = None
        rank = get_global_rank()
        process_group = kwargs.pop("process_group", None)

        # Do all filesystem reads from rank 0 only
        if rank == 0:
            ret = func(*args, **kwargs)

        # If not running in a distributed setting, return as is
        if not (dist.is_available() and dist.is_initialized()):
            # we cast here to avoid type errors, since it is
            # guaranteed the return value is of type T
            return cast(T, ret)

        # Otherwise, broadcast result from rank 0 to all ranks
        pg = PGWrapper(process_group)
        path_container = [ret]
        pg.broadcast_object_list(path_container, 0)
        val = path_container[0]

        # we cast here to avoid type errors, since it is
        # guaranteed the return value is of type T
        return cast(T, val)

    return wrapper


@rank_zero_read_and_broadcast
def get_latest_checkpoint_path(
    dirpath: str,
    metadata_fname: Optional[str] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Optional[str]:
    """
    Given a parent directory where checkpoints are saved, return the latest checkpoint subdirectory.

    Args:
        dirpath: parent directory where checkpoints are saved.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
        process_group: the process group on which the ranks will communicate on. default: ``None`` (the entire world)

    Raises:
        AssertionError if the checkpoint subdirectories are not named in the format epoch_{epoch}_step_{step}.
    """

    return _latest_checkpoint_path(dirpath, metadata_fname)


def _latest_checkpoint_path(
    dirpath: str, metadata_fname: Optional[str]
) -> Optional[str]:
    candidate_dirpaths = _retrieve_checkpoint_dirpaths(dirpath, metadata_fname)

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
        if len(split) < 4:
            raise AssertionError(
                f"Expected 4 or more elements for pattern of epoch_N_step_M, but received {split})"
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

    if largest_subdirectory is None:
        return None

    # Rejoin with the parent directory path and return the largest subdirectory
    return os.path.join(dirpath, none_throws(largest_subdirectory))


@rank_zero_read_and_broadcast
def get_best_checkpoint_path(
    dirpath: str,
    metric_name: str,
    mode: Literal["min", "max"],
    metadata_fname: Optional[str] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Optional[str]:
    """
    Given a parent directory where checkpoints are saved, return the best checkpoint subdirectory.

    Args:
        dirpath: parent directory where checkpoints are saved.
        metric_name: Name of the metric to use to find the best checkpoint.
        mode: Either 'min' or 'max'. If 'min', finds and loads the lowest value metric checkpoint. If 'max', finds and loads the largest.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
        process_group: the process group on which the ranks will communicate on. default: ``None`` (the entire world)
    """

    dirpaths = _retrieve_checkpoint_dirpaths(dirpath, metadata_fname, metric_name)
    if len(dirpaths) == 0:
        # no checkpoints found
        return None

    best_checkpoint_path = None
    best_metric_value = float("inf") if mode == "min" else float("-inf")
    for dirpath in dirpaths:
        dirname = os.path.basename(dirpath)
        metric_value = float(dirname.split("=")[-1])

        if mode == "min":
            if metric_value < best_metric_value:
                best_metric_value = metric_value
                best_checkpoint_path = dirpath
        else:
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_checkpoint_path = dirpath

    return best_checkpoint_path


@rank_zero_read_and_broadcast
def get_checkpoint_dirpaths(
    dirpath: str,
    metadata_fname: Optional[str] = None,
    metric_name: Optional[str] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> List[str]:
    """
    Given a parent directory where checkpoints are saved, returns the checkpoint subdirectories.
    The order of the checkpoints is not guarenteed.

    Args:
        dirpath: parent directory where checkpoints are saved.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
        metric_name: fetches all the checkpoint directories containing the metric name only.
        process_group: the process group on which the ranks will communicate on. default: ``None`` (the entire world)
    """

    return _retrieve_checkpoint_dirpaths(dirpath, metadata_fname, metric_name)


def _sort_by_recency(dirpaths: List[str]) -> List[str]:
    """
    Sorts the given list of directories by oldest to newest.

    Args:
        dirpaths: A list of directory paths.

    Returns:
        A sorted list of directory paths, sorted by recency.
    """

    def sort_fn(path: str) -> Tuple[int, int]:
        x = os.path.basename(path)
        return (int(x.split("_")[1]), int(x.split("_")[3]))

    return sorted(dirpaths, key=sort_fn)


def _sort_by_metric_value(
    dirpaths: List[str], mode: Literal["min", "max"]
) -> List[str]:
    """
    Sorts the given list of directories by the metric values.

    Args:
        dirpaths: A list of directory paths.
        mode: Either 'min' or 'max'. If 'min', sorts in descending order. If 'max', sorts in ascending order

    Returns:
        A sorted list of directory paths, sorted by the metric values.
    """

    def sort_metric_fn(path: str) -> float:
        x = os.path.basename(path)
        metric_val = float(x.split("=")[-1])
        return metric_val

    return sorted(
        dirpaths,
        key=sort_metric_fn,
        # sort descending if min, placing worst metric at top of list
        reverse=(mode == "min"),
    )


def _retrieve_checkpoint_dirpaths(
    dirpath: str,
    metadata_fname: Optional[str],
    metric_name: Optional[str] = None,
) -> List[str]:
    """
    Given a parent directory where checkpoints are saved, return the unsorted checkpoint subdirectories

    Args:
        dirpath: parent directory where checkpoints are saved.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
        metric_name: Name of the metric that must exist in checkpoint name.
    """

    if dirpath[-1] == "/":
        # removes trailing forward slash if present
        # required for regex search to work
        dirpath = dirpath[:-1]

    fs = get_filesystem(dirpath)

    if not fs.exists(dirpath):
        logger.warning(f"Input dirpath doesn't exist: {dirpath}")
        return []

    contents = fs.ls(dirpath, detail=True)
    contents = [item["name"] for item in contents if item["type"] == "directory"]
    if len(contents) == 0:
        logger.warning(f"Input dirpath doesn't contain any subdirectories: {dirpath}")
        return []

    # Define the regex pattern to match the directory names
    pattern = rf"^{dirpath}/epoch_\d+_step_\d+"
    if metric_name:
        # inject metric name in regex search
        pattern += rf"_{metric_name}="
    snapshot_dirpath_pattern: Pattern[str] = re.compile(pattern)
    candidate_dirpaths = list(filter(snapshot_dirpath_pattern.match, contents))

    if not metadata_fname:
        # return early as we don't need to filter out any paths
        return candidate_dirpaths

    # Iterate through all files and directories in the specified directory
    # and check if metedata is present or not
    valid_ckpt_dirpaths = []
    for candidate in candidate_dirpaths:
        if not _metadata_exists(fs, candidate, metadata_fname):
            logger.warning(
                f"Snapshot metadata is missing from {candidate}! Skipping this path"
            )
            continue

        valid_ckpt_dirpaths.append(candidate)

    return valid_ckpt_dirpaths


def _delete_checkpoint(dirpath: str, metadata_fname: Optional[str] = None) -> None:
    fs = get_filesystem(dirpath)
    if metadata_fname and not _metadata_exists(fs, dirpath, metadata_fname):
        raise RuntimeError(f"{dirpath} does not contain {metadata_fname}")
    fs.rm(dirpath, recursive=True)


def _metadata_exists(
    fs: fsspec.AbstractFileSystem, dirpath: str, metadata_fname: str
) -> bool:
    return fs.exists(os.path.join(dirpath, metadata_fname))


# keys for use when checkpointing
_TRAIN_PROGRESS_STATE_KEY = "train_progress"
_TRAIN_DL_STATE_KEY = "train_dataloader"
_EVAL_PROGRESS_STATE_KEY = "eval_progress"


def _prepare_app_state(unit: AppStateMixin) -> Dict[str, Any]:
    """Join together all of the tracked stateful entities to simplify registration of snapshottable states, deals with FSDP case"""
    app_state = unit.app_state()
    tracked_optimizers = unit._construct_tracked_optimizers()  # handles fsdp
    app_state.update(tracked_optimizers)
    return app_state


def _prepare_app_state_for_checkpoint(
    state: State, unit: AppStateMixin, intra_epoch: bool
) -> Dict[str, Stateful]:
    """
    Prepares the application state for checkpointing.
    """
    app_state = _prepare_app_state(unit)

    # for intra-epoch checkpointing, include dataloader states
    train_state = none_throws(state.train_state)
    train_dl = train_state.dataloader
    if intra_epoch and isinstance(train_dl, Stateful):
        app_state[_TRAIN_DL_STATE_KEY] = train_dl

    return app_state


def _prepare_app_state_for_restore(
    unit: AppStateMixin, restore_options: RestoreOptions
) -> Dict[str, Any]:
    """
    Prepares the application state for restoring from a checkpoint given a RestoreOptions.
    """
    app_state = _prepare_app_state(unit)

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

    return app_state
