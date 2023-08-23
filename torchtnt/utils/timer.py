#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np

import torch
import torch.distributed as dist
from tabulate import tabulate
from torchtnt.utils.distributed import PGWrapper

logger: logging.Logger = logging.getLogger(__name__)

AsyncOperator = TypeVar("AsyncOperator")

logger: logging.Logger = logging.getLogger(__name__)

_TABLE_ROW = Tuple[str, float, int, float, float]
_TABLE_DATA = List[_TABLE_ROW]


@contextmanager
def log_elapsed_time(
    action_name: str, *, cuda_sync: Optional[bool] = None
) -> Generator[None, None, None]:
    """Utility to measure and log elapsed time for a given event.

    Args:
        action_name: the name of the event being timed.
        cuda_sync: Whether to synchronize the stream in order to measure the most accurate timings on CUDA. Defaults to True if CUDA is available.

    Raises:
        ValueError: If cuda_sync is set to True but CUDA is not available.
    """
    if cuda_sync and not torch.cuda.is_available():
        raise ValueError(
            "CUDA must be available in order to enable CUDA synchronization."
        )
    cuda_sync = cuda_sync if cuda_sync is not None else torch.cuda.is_available()
    try:
        if cuda_sync:
            torch.cuda.synchronize()
        start_time: float = perf_counter()
        yield
    finally:
        if cuda_sync:
            torch.cuda.synchronize()
        interval_time: float = perf_counter() - start_time
        logger.info(f"{action_name} took {interval_time} seconds")


@runtime_checkable
class TimerProtocol(Protocol):
    """
    Defines a Timer Protocol with `time` and `reset` methods and an attribute `recorded_durations` for storing timings.
    """

    recorded_durations: Dict[str, List[float]]

    @contextmanager
    def time(self, action_name: str) -> Generator[None, None, None]:
        """
        A context manager for timing a code block.

        Args:
            action_name: the name under which to store the timing of what is enclosed in the context manager.
        """
        ...

    def reset(self) -> None:
        """
        A method to reset the state of the Timer.
        """
        ...


class Timer(TimerProtocol):
    def __init__(
        self,
        *,
        cuda_sync: Optional[bool] = None,
        verbose: bool = False,
    ) -> None:
        """
        A Timer class which implements TimerProtocol and stores timings in a dictionary `recorded_durations`.

        Args:
            cuda_sync: whether to call torch.cuda.synchronize() before and after timing. Defaults to True if CUDA is available.
            verbose: whether to enable verbose logging.
            size_bounds: defines the range of samples that should be kept in the timer. The lower bound should be smaller than
                the upper bound. When the number of samples reaches the upper bound, the oldest (upper-lower) bound samples will
                be removed. This range is applied per action.

        Note:
            Enabling cuda_sync will incur a performance hit, but will ensure accurate timings on GPUs.

        Raises:
            ValueError: If cuda_sync is set to True but CUDA is not available.

        """
        if cuda_sync and not torch.cuda.is_available():
            raise ValueError(
                "CUDA must be available in order to enable CUDA synchronization."
            )
        self.cuda_sync: bool = (
            cuda_sync if cuda_sync is not None else torch.cuda.is_available()
        )
        self.verbose = verbose
        self.recorded_durations: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def time(
        self,
        action_name: str,
    ) -> Generator[None, None, None]:
        """
        A context manager for timing a code block, with optional cuda synchronization and verbose timing.

        Args:
            action_name: the name under which to store the timing of what is enclosed in the context manager.
        """
        start_time: float = perf_counter()
        try:
            if self.cuda_sync:
                torch.cuda.synchronize()
            yield
        finally:
            if self.cuda_sync:
                torch.cuda.synchronize()
            interval_time: float = perf_counter() - start_time
            if self.verbose:
                logger.info(f"{action_name} took {interval_time} seconds.")
        self.recorded_durations[action_name].append(interval_time)

    def reset(self) -> None:
        """
        Reset the recorded_durations to an empty list
        """
        self.recorded_durations = defaultdict(list)


class BoundedTimer(Timer):
    """
    A Timer class which implements TimerProtocol and stores timings in a dictionary `recorded_durations`.

    Same behavior as timer, but with the addition of size_bounds = (lower, upper)

    Args:
        ...
        size_bounds: defines the range of samples that should be kept in the timer. The lower bound should be smaller than
            the upper bound. When the number of samples reaches the upper bound, the oldest (upper-lower) bound samples will
            be removed. This range is applied per action.
    """

    def __init__(self, lower_bound: int, upper_bound: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert lower_bound > 0
        assert lower_bound < upper_bound
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @contextmanager
    def time(
        self,
        action_name: str,
    ) -> Generator[None, None, None]:
        with super().time(action_name):
            yield
        self._apply_bounds(action_name)

    def _apply_bounds(self, action_name: str) -> None:
        # Keep 'lower_bound' most recent samples, if at or over upper bound
        n_samples: int = len(self.recorded_durations[action_name])
        if self.upper_bound <= n_samples:
            self.recorded_durations[action_name] = list(
                self.recorded_durations[action_name][-self.lower_bound :]
            )


def _get_total_time(timer: TimerProtocol) -> float:
    total_time = 0.0
    for _, durations in timer.recorded_durations.items():
        array_value = np.array(durations)
        array_sum = np.sum(array_value)
        total_time += array_sum

    return total_time


def _make_report(timer: TimerProtocol) -> Tuple[_TABLE_DATA, float, float]:
    total_time = _get_total_time(timer)
    report = [
        (
            a,
            np.mean(d),
            len(d),
            np.sum(d),
            100.0 * np.sum(d) / total_time,
        )
        for a, d in timer.recorded_durations.items()
    ]
    report.sort(key=lambda x: x[4], reverse=True)
    total_calls = sum(x[2] for x in report)
    return report, total_calls, total_time


def get_timer_summary(timer: TimerProtocol) -> str:
    """Given a timer, generate a summary of all the recorded actions.

    Args:
        timer: the Timer object for which to generate a summary

    Raises:
        ValueError
            If the input Timer has no recorded actions
    """
    sep: str = os.linesep
    output_string = f"Timer Report{sep}"

    if len(timer.recorded_durations) == 0:
        return output_string

    max_key = max(len(k) for k in timer.recorded_durations.keys())

    def log_row(action: str, mean: str, num_calls: str, total: str, per: str) -> str:
        row = f"{sep}|  {action:<{max_key}s}\t|  {mean:<15}\t|"
        row += f"  {num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|"
        return row

    header_string = log_row(
        "Action",
        "Mean duration (s)",
        "Num calls",
        "Total time (s)",
        "Percentage %",
    )
    output_string_len = len(header_string.expandtabs()) - 1
    sep_lines = f"{sep}{'-' * output_string_len}"
    output_string += sep_lines + header_string + sep_lines
    report: _TABLE_DATA
    (
        report,
        total_calls,
        total_duration,
    ) = _make_report(timer)
    output_string += log_row(
        "Total", "-", f"{total_calls:}", f"{total_duration:.5}", "100 %"
    )
    output_string += sep_lines
    for (
        action,
        mean_duration,
        num_calls,
        total_duration,
        duration_per,
    ) in report:
        output_string += log_row(
            action,
            f"{mean_duration:.5}",
            f"{num_calls}",
            f"{total_duration:.5}",
            f"{duration_per:.5}",
        )
    output_string += sep_lines

    output_string += sep
    return output_string


def get_durations_histogram(
    recorded_durations: Dict[str, List[float]],
    percentiles: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    """Computes a histogram of percentiles from the recorded durations passed in.

    Args:
        recorded_durations: The mapping of durations to sync and compute histograms from.
        percentiles: The percentiles to compute. Values should be in the range [0, 100].

    Returns:
        A dictionary mapping the action names to a dictionary of the computed percentiles, along with the mean duration of each action.

    Raises:
        ValueError: If the input percentiles are not in the range [0, 100].
    """
    _validate_percentiles(percentiles)
    percentiles = sorted(percentiles)
    return _compute_percentiles(recorded_durations, percentiles=percentiles)


def get_synced_durations_histogram(
    recorded_durations: Dict[str, List[float]],
    percentiles: Sequence[float],
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[str, float]]:
    """Synchronizes the recorded durations across ranks.

    Args:
        recorded_durations: The mapping of durations to sync and compute histograms from.
        percentiles: The percentiles to compute. Values should be in the range [0, 100].
        pg (optional): The process group to use for synchronization. Defaults to the global process group.

    Returns:
        A dictionary mapping the action names to a dictionary of the computed percentiles, along with the mean duration of each action.

    Raises:
        ValueError: If the input percentiles are not in the range [0, 100].
    """
    _validate_percentiles(percentiles)
    synced_durations = _sync_durations(recorded_durations, pg)
    return get_durations_histogram(synced_durations, percentiles=percentiles)


def get_synced_timer_histogram(
    timer: TimerProtocol,
    percentiles: Sequence[float],
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[str, float]]:
    """Synchronizes the input timer's recorded durations across ranks.

    Args:
        timer: The TimerProtocol object whose recorded durations will be synced.
        percentiles: The percentiles to compute. Values should be in the range [0, 100].
        pg (optional): The process group to use for synchronization. Defaults to the global process group.

    Returns:
        A dictionary mapping the action names to a dictionary of the computed percentiles, along with the mean duration of each action.

    Raises:
        ValueError: If the input percentiles are not in the range [0, 100].
    """
    return get_synced_durations_histogram(
        timer.recorded_durations, percentiles=percentiles, pg=pg
    )


def _sync_durations(
    recorded_durations: Dict[str, List[float]], pg: Optional[dist.ProcessGroup]
) -> Dict[str, List[float]]:
    if not (dist.is_available() and dist.is_initialized()):
        return recorded_durations

    pg_wrapper = PGWrapper(pg)
    world_size = pg_wrapper.get_world_size()
    outputs = [None] * world_size
    pg_wrapper.all_gather_object(outputs, recorded_durations)
    ret = defaultdict(list)
    for output in outputs:
        # pyre-ignore [16]: `Optional` has no attribute `__getitem__`.
        for k, v in output.items():
            if k not in ret:
                ret[k] = []
            ret[k].extend(v)
    return ret


def _compute_percentiles(
    durations: Dict[str, List[float]], percentiles: Sequence[float]
) -> Dict[str, Dict[str, float]]:
    ret = {}
    for name, values in durations.items():
        ret[name] = _compute_percentile(name, values, percentiles=percentiles)
    return ret


def _compute_percentile(
    name: str, timings: List[float], percentiles: Sequence[float]
) -> Dict[str, float]:
    ret = {}

    # By default, numpy's percentile function will interpolate between values,
    # but we want to snap to actual metrics that were recorded. For more
    # discussion of percentile interpolation, see:
    # https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
    computed_percentiles = np.percentile(timings, percentiles, interpolation="lower")

    # computed_percentiles is a sequence of floats with the percentile
    # results. We use enumerate to allow us to grab the index for each
    # computed percentile, so that we can grab the corresponding percentile
    # to use as the "name" when we turn these values into the metrics.
    for i, percentile_value in enumerate(computed_percentiles):
        percentile = percentiles[i]

        ret[f"p{percentile}"] = percentile_value

    # include the mean as well in addition to the percentiles passed in
    ret["avg"] = np.mean(timings)
    return ret


def _validate_percentiles(percentiles: Sequence[float]) -> None:
    for p in percentiles:
        if p < 0 or p > 100:
            raise ValueError(f"Percentile must be between 0 and 100. Got {p}")


def get_recorded_durations_table(result: Dict[str, Dict[str, float]]) -> str:
    r"""
    Helper function to generate recorded duration time in tabular format
    """
    if len(result) == 0:
        return ""
    sub_dict = next(iter(result.values()))
    if len(sub_dict) == 0:
        return ""
    column_headers = ["Name"] + list(sub_dict.keys())
    row_output = []
    for key in result:
        row = [key] + ["{:.3f}".format(x) for x in result[key].values()]
        row_output.append(row)
    tabulate_output = tabulate(
        row_output,
        tablefmt="pipe",
        headers=column_headers,
    )
    return "\n" + tabulate_output


class FullSyncPeriodicTimer:
    """
    Measures time (resets if given interval elapses) on rank 0
    and propagates result to other ranks.
    Propagation is done asynchronously from previous step
    in order to avoid blocking of a training process.
    """

    def __init__(self, interval: datetime.timedelta, cpu_pg: dist.ProcessGroup) -> None:
        self._interval = interval
        self._cpu_pg = cpu_pg
        self._prev_time: float = perf_counter()
        self._timeout_tensor: torch.Tensor = torch.zeros(1, dtype=torch.int)
        # pyre-fixme[34]: `Variable[AsyncOperator]` isn't present in the function's parameters.
        self._prev_work: Optional[AsyncOperator] = None

    def check(self) -> bool:
        ret = False
        curr_time = perf_counter()

        if self._prev_work is not None:
            # pyre-fixme[16]: `Variable[AsyncOperator]` has no attribute wait.
            self._prev_work.wait()
            ret = self._timeout_tensor[0].item() == 1
            if ret:
                self._prev_time = curr_time

        self._timeout_tensor[0] = (
            1 if (curr_time - self._prev_time) >= self._interval.total_seconds() else 0
        )
        self._prev_work = dist.broadcast(
            self._timeout_tensor, 0, group=self._cpu_pg, async_op=True
        )

        return ret
