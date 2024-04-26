# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import os
import re
from dataclasses import dataclass
from functools import total_ordering
from typing import Literal, Optional, Pattern

from pyre_extensions import none_throws


@dataclass
class MetricData:
    """
    Representation of a metric instance. Should provide both a metric name and it's value.
    """

    name: str
    value: float


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
