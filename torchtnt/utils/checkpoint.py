# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import bisect
import logging
import os
import re
from dataclasses import dataclass
from functools import total_ordering
from typing import List, Literal, Optional, Pattern

import fsspec
import torch.distributed as dist
from fsspec.core import url_to_fs
from pyre_extensions import none_throws
from torchtnt.utils.distributed import PGWrapper, rank_zero_read_and_broadcast

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """
    Representation of a metric instance. Should provide both a metric name and it's value.
    """

    name: str
    value: float


@dataclass
class BestCheckpointConfig:
    """
    Config for saving the best checkpoints.

    Args:
        monitored_metric: Metric to monitor for saving best checkpoints. Must be an numerical or tensor attribute on the unit.
        mode: One of `min` or `max`. The save file is overwritten based the max or min of the monitored metric.
    """

    monitored_metric: str
    mode: Literal["min", "max"] = "min"


@total_ordering
class CheckpointPath:
    """
    Representation of a checkpoint path. Handles parsing and serialization of the specific path format.
    Currently, the basic compliant path format is: <dirpath>/epoch_<epoch>_step_<step>
    If a metric is being tracked, it's added to the name: <dirpath>/epoch_<epoch>_step_<step>_<metric_name>=<metric_value>

    This class is well-ordered by checkpoint recency, so any comparisons will operate using the epoch + step. Sorting by
    metric can be done by extracting the metric value from the metric_data attribute.
    """

    PATH_REGEX: Pattern = re.compile(
        r"^(.+)epoch_(\d+)_step_(\d+)(?:_(.+)=(-?\d+\.?\d*))?\/?$"
    )

    def __init__(
        self,
        dirpath: str,
        epoch: int,
        step: int,
        metric_data: Optional[MetricData] = None,
    ) -> None:
        """
        Args:
            dirpath: The base directory path that checkpoints are saved in.
            epoch: The epoch number of this checkpoint.
            step: The step number of this checkpoint.
            metric_data: Optional data about the metric being tracked. Should contain both metric name and value.
        """
        self.dirpath: str = dirpath.rstrip("/")
        self.epoch = epoch
        self.step = step
        self.metric_data = metric_data

    @classmethod
    def from_str(cls, checkpoint_path: str) -> "CheckpointPath":
        """
        Given a directory path, try to parse it and extract the checkpoint data.
        The expected format is: <dirpath>/epoch_<epoch>_step_<step>_<metric_name>=<metric_value>,
        where the metric name and value are optional.

        Args:
            checkpoint_path: The path to the checkpoint directory.

        Returns:
            A CheckpointPath instance if the path is valid, otherwise None.

        Raises:
            ValueError: If the path is malformed and can't be parsed.
        """
        path_match = cls.PATH_REGEX.match(checkpoint_path)
        if not path_match:
            raise ValueError(
                f"Attempted to parse malformed checkpoint path: {checkpoint_path}."
            )

        dirpath, epoch, step, metric_name, metric_value = path_match.groups()
        try:
            metric_data: Optional[MetricData] = None
            if metric_name:
                metric_value_f = float(metric_value)
                metric_data = MetricData(name=metric_name, value=metric_value_f)

            return CheckpointPath(
                dirpath=dirpath,
                epoch=int(epoch),
                step=int(step),
                metric_data=metric_data,
            )

        except ValueError:
            # Should never happen since path matches regex
            raise ValueError(
                f"Invalid data types found in checkpoint path: {checkpoint_path}."
            )

    @property
    def path(self) -> str:
        """
        Returns:
            The full path to the checkpoint directory.
        """
        name = f"epoch_{self.epoch}_step_{self.step}"
        if self.metric_data:
            name += f"_{self.metric_data.name}={self.metric_data.value}"

        return os.path.join(self.dirpath, name)

    def newer_than(self, other: "CheckpointPath") -> bool:
        """
        Given another CheckpointPath instance, determine if this checkpoint is strictly newer than the other.

        Returns:
            True if this checkpoint is newer than the other, otherwise False.
        """
        if self.epoch != other.epoch:
            return self.epoch > other.epoch

        return self.step > other.step

    def more_optimal_than(
        self, other: "CheckpointPath", mode: Literal["min", "max"]
    ) -> bool:
        """
        Given another CheckpointPath instance, determine if this checkpoint is strictly more optimal than the other.
        Optimality is determined by comparing the metric value of the two checkpoints. The mode indicates if the
        metric value should be minimized or maximized. This only works for metric-aware checkpoints.

        Args:
            other: The other checkpoint path to compare against.
            mode: The mode to use for comparison.

        Returns:
            True if this checkpoint is more optimal than the other, otherwise False.

        Note: This expects that both checkpoints are metric-aware, and that they are tracking the same metric.
        """

        assert (
            self.metric_data and other.metric_data
        ), f"Attempted to compare optimality of non metric-aware checkpoints: {self} and {other}"

        assert (
            self.metric_data.name == other.metric_data.name
        ), f"Attempted to compare optimality of checkpoints tracking different metrics: {self} and {other}"

        if mode == "min":
            return (
                none_throws(self.metric_data).value
                < none_throws(other.metric_data).value
            )

        return (
            none_throws(self.metric_data).value > none_throws(other.metric_data).value
        )

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return f"CheckpointPath(dirpath={self.dirpath}, epoch={self.epoch}, step={self.step}, metric_data={self.metric_data})"

    def __eq__(self, other: "CheckpointPath") -> bool:
        return (
            self.dirpath == other.dirpath
            and self.epoch == other.epoch
            and self.step == other.step
            and self.metric_data == other.metric_data
        )

    def __gt__(self, other: "CheckpointPath") -> bool:
        return self.newer_than(other)

    def __getstate__(self) -> str:
        # Lightweight pickling to avoid broadcast errors
        return self.path

    def __setstate__(self, state: str) -> None:
        # Match regex directly to avoid creating a new instance with `from_str`
        path_match = self.PATH_REGEX.match(state)
        assert path_match, f"Malformed checkpoint found when unpickling: {state}"

        dirpath, epoch, step, metric_name, metric_value = path_match.groups()
        self.dirpath = dirpath.rstrip("/")
        self.epoch = int(epoch)
        self.step = int(step)
        self.metric_data = (
            MetricData(name=metric_name, value=float(metric_value))
            if metric_name and metric_value
            else None
        )


class CheckpointManager:
    """
    Manage a group of CheckpointPaths that belong to the same base directory. This involves maintaining
    ordering checkpoints by recency or metric value if applicable. Then, this is used to determine if a
    checkpoint should be saved, and what name will be used.

    The checkpoints work in the following format: <dirpath>/epoch_<epoch>_step_<step>
    If a metric is being tracked, it's added to the name: <dirpath>/epoch_<epoch>_step_<step>_<metric_name>=<metric_value>

    The methods in this class are meant to be used in the following order:
    1. Create instance - this will load the existing checkpoints (if any)
    2. `prune_surplus_checkpoints` - this will remove the non-optimal checkpoints to enforce the `keep_last_n_checkpoints`
    3. For every checkpointing iteration:
        a. `generate_checkpoint_path`: Gives the CheckpointPath that would be saved next
        b. `should_save_checkpoint`: Determines if checkpoint should be saved according to the `keep_last_n_checkpoints` and `best_checkpoint_config`
        c. -- The external checkpointing API should be called if above returns True. CheckpointManager does NOT actually generate checkpoints --
        d. `append_checkpoint`: If the checkpoint was successfully saved, this should be called to update the internal state

    In general, every file system read/write operation performed by this class is executed only in rank 0, while state is synced across ranks.
    """

    def __init__(
        self,
        dirpath: str,
        best_checkpoint_config: Optional[BestCheckpointConfig] = None,
        keep_last_n_checkpoints: Optional[int] = None,
        metadata_fname: Optional[str] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        """
        Initialize a checkpoint manager. If a `keep_last_n_checkpoints` value is provided, this will read the
        existing checkpoints in the dirpath (from rank 0 only) to account for them in the max number of checkpoints
        to keep. Note that no checkpoints are deleted.

        Args:
            dirpath: The base directory path that checkpoints are saved in. This is synced from rank 0 to every other rank upon initialization.
            best_checkpoint_config: Optional configuration for the best checkpoint.
            keep_last_n_checkpoints: Optional number of checkpoints to keep.
            metadata_fname: Optional name of the metadata file.
            process_group: Optional process group to use for distributed training. gloo process groups are known
                to perform better.
        """
        self.dirpath: str = self._sync_dirpath_to_all_ranks(
            dirpath=dirpath, process_group=process_group
        )

        self._best_checkpoint_config = best_checkpoint_config
        self._keep_last_n_checkpoints = keep_last_n_checkpoints
        self._metadata_fname = metadata_fname
        self._pg_wrapper = PGWrapper(process_group)

        self._ckpt_paths: List[CheckpointPath] = []
        if not self._keep_last_n_checkpoints:
            return

        # If there is a max limit of checkpoints to store, keep track of existing ones
        metric_name = (
            best_checkpoint_config.monitored_metric if best_checkpoint_config else None
        )
        self._ckpt_paths = get_checkpoint_dirpaths(
            dirpath=dirpath,
            metadata_fname=self._metadata_fname,
            metric_name=metric_name,
            process_group=process_group,
        )
        if best_checkpoint_config:
            self._ckpt_paths.sort(
                key=lambda x: x.metric_data.value,
                # sort descending if min, placing worst metric at top of list
                reverse=(best_checkpoint_config.mode == "min"),
            )
        else:
            self._ckpt_paths.sort()  # Checkpoint paths are well-ordered by recency

    def prune_surplus_checkpoints(self) -> None:
        """
        Prune checkpoints that exceed the maximum number of checkpoints to keep. This should be
        called when training starts, so that the `keep_last_n_checkpoints` config is honored.
        Files are only deleted in rank 0.

        Note:
            This is not called on initialization, in case users want to inpsect previous
            checkpoints. But it should be called before starting training if there is a
            `keep_last_n_checkpoints` config.

        Args:
            state: The training state.
        """
        keep_last_n_checkpoints = self._keep_last_n_checkpoints
        if keep_last_n_checkpoints and len(self._ckpt_paths) > keep_last_n_checkpoints:
            logger.warning(
                (
                    f"{len(self._ckpt_paths)} checkpoints found in {self.dirpath}. ",
                    f"Deleting {len(self._ckpt_paths) - keep_last_n_checkpoints} oldest ",
                    "checkpoints to enforce ``keep_last_n_checkpoints`` argument.",
                )
            )
            for _ in range(len(self._ckpt_paths) - keep_last_n_checkpoints):
                self.remove_checkpoint()

    def generate_checkpoint_path(
        self, epoch: int, step: int, metric_data: Optional[MetricData] = None
    ) -> CheckpointPath:
        """
        Given the current epoch, step, and possibly a metric_data value, determine the path
        where it should be stored. This does not necessarily mean that the checkpoint should
        be created. Instead, `should_save_checkpoint` has to be called to determine that.

        Args:
            unit: The training unit.
            state: The training state.

        Returns:
            The path to the checkpoint to save.

        Raises: AssertionError if there is a mismatch in tracked metric, for example:
            - `best_checkpoint_config` is not set but `metric_data` was provided
            - `best_checkpoint_config` is set and `metric_data` is passed. But they are not tracking the same metric
        """

        if metric_data:
            assert (
                self._best_checkpoint_config
            ), "Attempted to get a checkpoint with metric but best checkpoint config is not set"

            assert self._best_checkpoint_config.monitored_metric == metric_data.name, (
                f"Attempted to get a checkpoint with metric '{metric_data.name}', "
                f"but best checkpoint config is for '{none_throws(self._best_checkpoint_config).monitored_metric}'"
            )

        checkpoint_path = CheckpointPath(
            self.dirpath, epoch, step, metric_data=metric_data
        )

        return checkpoint_path

    def should_save_checkpoint(self, checkpoint: CheckpointPath) -> bool:
        """
        Given a unit and state, determine if a checkpoint should be saved when considering the `keep_last_n_checkpoints`
        and `best_checkpoint_config` configs.

        Args:
            checkpoint: The CheckpointPath to be potentially saved, provided by `generate_checkpoint_path`.

        Returns:
            True if the checkpoint should be saved, otherwise False.
        """

        keep_last_n_checkpoints = self._keep_last_n_checkpoints
        if not keep_last_n_checkpoints:
            # always save candidate checkpoint if no limit of checkpoints is specified
            return True

        if len(self._ckpt_paths) < keep_last_n_checkpoints:
            # limit of checkpoints has not been reached
            return True

        best_checkpoint_config = self._best_checkpoint_config
        if not best_checkpoint_config:
            # we always save the latest checkpoint
            return True

        # otherwise, we need to determine if we should overwrite the worst checkpoint
        return checkpoint.more_optimal_than(
            self._ckpt_paths[0], mode=best_checkpoint_config.mode
        )

    def append_checkpoint(self, ckpt: CheckpointPath) -> None:
        """
        This will update the internal state to keep track of the checkpoint. This function should only be called
        when a checkpoint whose path was returned from `maybe_get_next_checkpoint_path` was successfully created.
        If a previous checkpoint should be removed to honor `keep_last_n_checkpoint`, it will be deleted on rank 0.

        Args:
            ckpt: The checkpoint to save.
            state: The training state.
        """
        # Remove oldest checkpoint if needed
        max_ckpts = self._keep_last_n_checkpoints
        if max_ckpts and len(self._ckpt_paths) >= max_ckpts:
            self.remove_checkpoint()

        # If we are monitoring a metric, but the checkpoint has no metric data, we don't track it
        if self._best_checkpoint_config and ckpt.metric_data:
            keys = [none_throws(c.metric_data).value for c in self._ckpt_paths]
            if self._best_checkpoint_config.mode == "min":
                keys.reverse()

            # Use bisect.bisect() to find the insertion point
            idx = bisect.bisect(keys, none_throws(ckpt.metric_data).value)
            if none_throws(self._best_checkpoint_config).mode == "min":
                idx = len(self._ckpt_paths) - idx
            self._ckpt_paths.insert(idx, ckpt)

        elif not self._best_checkpoint_config:
            # No metric tracked, most recents goes last
            self._ckpt_paths.append(ckpt)

    @rank_zero_read_and_broadcast
    def does_checkpoint_exist(
        self, ckpt: CheckpointPath, process_group: Optional[dist.ProcessGroup] = None
    ) -> bool:
        """
        Checking whether a checkpoint already exists by verifying whether the optional metadata file is present in the directory.
        If the checkpointer doesn't have a metadata file, this function will always return False. Check is executed in rank 0, but
        result is broadcasted to all ranks.
        """
        metadata_fname = self._metadata_fname
        if not metadata_fname:
            return False

        fs, _ = url_to_fs(self.dirpath)
        return _metadata_exists(fs, ckpt.path, metadata_fname)

    @staticmethod
    @rank_zero_read_and_broadcast
    def _sync_dirpath_to_all_ranks(
        dirpath: str, process_group: Optional[dist.ProcessGroup] = None
    ) -> str:
        """Synchronize the dirpath across all ranks."""
        return dirpath

    def remove_checkpoint(self) -> None:
        """
        Delete the weakest checkpoint both from the internal state and from the file system (rank 0). This means:
        - If there is a `best_checkpoint_config`, then the checkpoint with the least optimal metric value
        - If there is no `best_checkpoint_config`, then the oldest checkpoint
        """
        worst_ckpt_path = self._ckpt_paths.pop(0)
        if self._pg_wrapper.get_rank() == 0:
            fs, _ = url_to_fs(self.dirpath)
            fs.rm(worst_ckpt_path.path, recursive=True)


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

    Note:
        When doing distributed training, only rank 0 will read the file system. The result will be broadcasted to all ranks.
        gloo process groups are recommended over nccl.
    """

    candidate_dirpaths = _retrieve_checkpoint_dirpaths(dirpath, metadata_fname)
    if not candidate_dirpaths:
        return None

    latest_checkpoint = candidate_dirpaths[0]
    for candidate in candidate_dirpaths[1:]:
        if candidate.newer_than(latest_checkpoint):
            latest_checkpoint = candidate

    return latest_checkpoint.path


@rank_zero_read_and_broadcast
def get_best_checkpoint_path(
    dirpath: str,
    metric_name: str,
    mode: Literal["min", "max"],
    metadata_fname: Optional[str] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Optional[str]:
    """
    Given a parent directory where checkpoints are saved, return the best checkpoint subdirectory based on a metric.

    The checkpoint paths are assumed to have the following format: <dirpath>/epoch_<epoch>_step_<step>_<metric_name>=<metric_value>
    This will always be the case if the CheckpointManager class is used to produce their names.

    Args:
        dirpath: parent directory where checkpoints are saved.
        metric_name: Name of the metric to use to find the best checkpoint.
        mode: Either 'min' or 'max'. If 'min', finds and loads the lowest value metric checkpoint. If 'max', finds and loads the largest.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
        process_group: the process group on which the ranks will communicate on. default: ``None`` (the entire world)

    Note:
        When doing distributed training, only rank 0 will read the file system. The result will be broadcasted to all ranks.
        gloo process groups are recommended over nccl.
    """

    dirpaths = _retrieve_checkpoint_dirpaths(dirpath, metadata_fname, metric_name)
    if not dirpaths:
        return None

    best_checkpoint = dirpaths[0]
    for checkpoint in dirpaths[1:]:
        if checkpoint.more_optimal_than(best_checkpoint, mode):
            best_checkpoint = checkpoint

    return best_checkpoint.path


@rank_zero_read_and_broadcast
def get_checkpoint_dirpaths(
    dirpath: str,
    metadata_fname: Optional[str] = None,
    metric_name: Optional[str] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> List[CheckpointPath]:
    """
    Given a parent directory where checkpoints are saved, returns the checkpoint subdirectories.
    The order of the checkpoints is not guaranteed.

    The checkpoint paths are assumed to have the following format: <dirpath>/epoch_<epoch>_step_<step>
    If a metric_name is provided the format should be <dirpath>/epoch_<epoch>_step_<step>_<metric_name>=<metric_value>
    This will always be the case if the CheckpointManager class is used to produce their names.

    Args:
        dirpath: parent directory where checkpoints are saved.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
        metric_name: fetches all the checkpoint directories containing the metric name only.
        process_group: the process group on which the ranks will communicate on. default: ``None`` (the entire world)

    Note:
        When doing distributed training, only rank 0 will read the file system. The result will be broadcasted to all ranks.
        gloo process groups are recommended over nccl.
    """

    return _retrieve_checkpoint_dirpaths(dirpath, metadata_fname, metric_name)


def _sort_by_recency(dirpaths: List[CheckpointPath]) -> List[CheckpointPath]:
    """
    Sorts the given list of directories by oldest to newest.

    Args:
        dirpaths: A list of directory paths.

    Returns:
        A sorted list of directory paths, sorted by recency.
    """

    return sorted(dirpaths)  # CheckpointPath is well ordered by recency


def _sort_by_metric_value(
    dirpaths: List[CheckpointPath], mode: Literal["min", "max"]
) -> List[CheckpointPath]:
    """
    Sorts the given list of directories by the metric values.

    Args:
        dirpaths: A list of directory paths.
        mode: Either 'min' or 'max'. If 'min', sorts in descending order. If 'max', sorts in ascending order

    Returns:
        A sorted list of directory paths, sorted by the metric values.
    """
    return sorted(
        dirpaths,
        key=lambda x: x.metric_data.value,
        # sort descending if min, placing worst metric at top of list
        reverse=(mode == "min"),
    )


def _retrieve_checkpoint_dirpaths(
    dirpath: str,
    metadata_fname: Optional[str],
    metric_name: Optional[str] = None,
) -> List[CheckpointPath]:
    """
    Given a parent directory where checkpoints are saved, return the unsorted checkpoint subdirectories

    Args:
        dirpath: parent directory where checkpoints are saved.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
        metric_name: Name of the metric that must exist in checkpoint name.
    """

    fs, _ = url_to_fs(dirpath)

    if not fs.exists(dirpath):
        logger.warning(f"Input dirpath doesn't exist: {dirpath}")
        return []

    contents = fs.ls(dirpath, detail=True)
    contents = [item["name"] for item in contents if item["type"] == "directory"]
    if len(contents) == 0:
        logger.warning(f"Input dirpath doesn't contain any subdirectories: {dirpath}")
        return []

    # Parse the valid checkpoint directories
    candidate_checkpoints: List[CheckpointPath] = []
    for candidate_dirpath in contents:
        try:
            ckpt = CheckpointPath.from_str(candidate_dirpath)
        except ValueError:
            continue

        # If a metric was provided, keep only the checkpoints tracking it
        if metric_name and not (
            ckpt.metric_data and ckpt.metric_data.name == metric_name
        ):
            continue

        candidate_checkpoints.append(ckpt)

    if not metadata_fname:
        # return early as we don't need to filter out any paths
        return candidate_checkpoints

    # Iterate through all files and directories in the specified directory
    # and check if metedata is present or not
    valid_ckpt_dirpaths: List[CheckpointPath] = []
    for candidate in candidate_checkpoints:
        if not _metadata_exists(fs, candidate.path, metadata_fname):
            logger.warning(
                f"Snapshot metadata is missing from {candidate}! Skipping this path"
            )
            continue

        valid_ckpt_dirpaths.append(candidate)

    return valid_ckpt_dirpaths


def _delete_checkpoint(dirpath: str, metadata_fname: Optional[str] = None) -> None:
    fs, _ = url_to_fs(dirpath)
    if metadata_fname and not _metadata_exists(fs, dirpath, metadata_fname):
        raise RuntimeError(f"{dirpath} does not contain {metadata_fname}")
    fs.rm(dirpath, recursive=True)


def _metadata_exists(
    fs: fsspec.AbstractFileSystem, dirpath: str, metadata_fname: str
) -> bool:
    return fs.exists(os.path.join(dirpath, metadata_fname))
