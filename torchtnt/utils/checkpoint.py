# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
import os
import re
from dataclasses import dataclass
from functools import total_ordering
from typing import List, Literal, Optional, Pattern, Tuple

import fsspec
import torch.distributed as dist
from fsspec.core import url_to_fs
from pyre_extensions import none_throws
from torchtnt.utils.distributed import rank_zero_read_and_broadcast

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

    fs, _ = url_to_fs(dirpath)

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
    fs, _ = url_to_fs(dirpath)
    if metadata_fname and not _metadata_exists(fs, dirpath, metadata_fname):
        raise RuntimeError(f"{dirpath} does not contain {metadata_fname}")
    fs.rm(dirpath, recursive=True)


def _metadata_exists(
    fs: fsspec.AbstractFileSystem, dirpath: str, metadata_fname: str
) -> bool:
    return fs.exists(os.path.join(dirpath, metadata_fname))
