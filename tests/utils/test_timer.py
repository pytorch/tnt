#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import unittest
from datetime import timedelta
from random import random
from unittest import mock
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import spawn_threads_and_init_comms
from torchtnt.utils.timer import (
    FullSyncPeriodicTimer,
    get_durations_histogram,
    get_synced_durations_histogram,
    get_timer_summary,
    log_elapsed_time,
    logger,
    Timer,
)


class TimerTest(unittest.TestCase):
    def assert_within_tolerance(
        self, expected: float, actual: float, percent_tolerance: float = 10
    ) -> None:
        """Assert that a value is correct within a percent tolerance"""
        error = abs(expected - actual)
        tolerance = expected * (percent_tolerance / 100)
        self.assertLess(error, tolerance)

    def test_timer_verbose(self) -> None:
        timer = Timer(verbose=True)
        with mock.patch.object(logger, "info") as mock_info:
            with timer.time("Testing timer"):
                time.sleep(0.2)
            mock_info.assert_called_once()
            self.assertTrue("Testing timer took" in mock_info.call_args.args[0])

    @patch("torch.cuda.synchronize")
    def test_timer_context_manager(self, _) -> None:
        """Test the context manager in the timer class"""

        # Generate 3 intervals between 0.5 and 2 seconds
        intervals = [(random() * 1.5) + 0.5 for _ in range(4)]

        # Basic test of context manager
        timer = Timer()
        with timer.time("action_1"):
            time.sleep(intervals[0])

        with timer.time("action_2"):
            time.sleep(intervals[1])

        # Make sure nested context managers work properly
        with timer.time("action_3"):
            with timer.time("action_4"):
                time.sleep(intervals[2])

        for action in ("action_1", "action_2", "action_3", "action_4"):
            self.assertIn(action, timer.recorded_durations.keys())

        self.assertLess(
            timer.recorded_durations["action_4"][0],
            timer.recorded_durations["action_3"][0],
        )
        for i in range(3):
            self.assert_within_tolerance(
                timer.recorded_durations[f"action_{i+1}"][0], intervals[i]
            )
        self.assert_within_tolerance(
            timer.recorded_durations["action_4"][0], intervals[2]
        )

    @unittest.skipUnless(
        condition=torch.cuda.is_available(), reason="This test needs a GPU host to run."
    )
    @patch("torch.cuda.synchronize")
    def test_timer_synchronize(self, mock_synchornize) -> None:
        """Make sure that torch.cuda.synchronize() is called when GPU is present."""

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        timer = Timer()

        # Do not explicitly call synchronize, timer must call it for test to pass.

        with timer.time("action_1"):
            start_event.record()
            time.sleep(0.5)
            end_event.record()

        # torch.cuda.synchronize() has to be called to compute the elapsed time.
        # Otherwise, there will be runtime error.
        self.assertEqual(mock_synchornize.call_count, 2)

    def test_get_timer_summary(self) -> None:
        """Test the get_timer_summary function"""

        timer = Timer()
        summary = get_timer_summary(timer)
        self.assertEqual(summary, f"Timer Report{os.linesep}")

        with timer.time("action_1"):
            time.sleep(0.5)
        summary = get_timer_summary(timer)
        self.assertTrue("action_1" in summary)

    def test_invalid_get_synced_durations_histogram_percentiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "Percentile must be between 0 and 100"):
            get_synced_durations_histogram({}, percentiles=(-1,))
        with self.assertRaisesRegex(ValueError, "Percentile must be between 0 and 100"):
            get_synced_durations_histogram({}, percentiles=(101,))

    def test_get_synced_durations_histogram(self) -> None:
        recorded_durations = {
            "bar": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "foo": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "baz": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        }
        percentiles = (10.0, 25.0, 50.0, 75.0, 95.0, 99.0)
        durations = get_durations_histogram(recorded_durations, percentiles)
        synced_durations = get_synced_durations_histogram(
            recorded_durations, percentiles
        )
        expected_durations = {
            "bar": {
                "p10.0": 4.0,
                "p25.0": 6.0,
                "p50.0": 8.0,
                "p75.0": 10.0,
                "p95.0": 11.0,
                "p99.0": 11.0,
                "avg": 8.0,
            },
            "foo": {
                "p10.0": 1.0,
                "p25.0": 3.0,
                "p50.0": 5.0,
                "p75.0": 7.0,
                "p95.0": 9.0,
                "p99.0": 9.0,
                "avg": 5.5,
            },
            "baz": {
                "p10.0": 7.0,
                "p25.0": 9.0,
                "p50.0": 11.0,
                "p75.0": 13.0,
                "p95.0": 15.0,
                "p99.0": 15.0,
                "avg": 11.5,
            },
        }
        self.assertEqual(durations, expected_durations)
        self.assertEqual(durations, synced_durations)

    @staticmethod
    def _get_synced_durations_histogram_multi_process() -> None:
        rank = dist.get_rank()
        if rank == 0:
            recorded_durations = {
                "foo": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "bar": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            }
        else:
            recorded_durations = {
                "foo": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "bar": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "baz": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            }
        durations = get_synced_durations_histogram(
            recorded_durations, percentiles=(10.0, 25.0, 50.0, 75.0, 95.0, 99.0)
        )
        expected_durations = {
            "foo": {
                "p10.0": 2.0,
                "p25.0": 4.0,
                "p50.0": 7.0,
                "p75.0": 9.0,
                "p95.0": 11.0,
                "p99.0": 11.0,
                "avg": 6.684210526315789,
            },
            "bar": {
                "p10.0": 2.0,
                "p25.0": 4.0,
                "p50.0": 7.0,
                "p75.0": 9.0,
                "p95.0": 11.0,
                "p99.0": 11.0,
                "avg": 6.684210526315789,
            },
            "baz": {
                "p10.0": 7.0,
                "p25.0": 9.0,
                "p50.0": 11.0,
                "p75.0": 13.0,
                "p95.0": 15.0,
                "p99.0": 15.0,
                "avg": 11.5,
            },
        }
        tc = unittest.TestCase()
        tc.assertEqual(durations, expected_durations)

    @unittest.skipUnless(
        condition=dist.is_available(),
        reason="This test should only run if torch.distributed is available.",
    )
    @spawn_threads_and_init_comms(world_size=2)
    def test_get_synced_durations_histogram_multi_process(self) -> None:
        self._get_synced_durations_histogram_multi_process()

    def test_timer_fn(self) -> None:
        with log_elapsed_time("test"):
            pass


class FullSyncPeriodicTimerTest(unittest.TestCase):
    @classmethod
    def _full_sync_worker_without_timeout(cls) -> bool:
        process_group = dist.group.WORLD
        interval_threshold = timedelta(seconds=5)
        fsp_timer = FullSyncPeriodicTimer(interval_threshold, process_group)
        return fsp_timer.check()

    @classmethod
    def _full_sync_worker_with_timeout(cls, timeout: int) -> bool:
        process_group = dist.group.WORLD
        interval_threshold = timedelta(seconds=5)
        fsp_timer = FullSyncPeriodicTimer(interval_threshold, process_group)
        time.sleep(timeout)
        fsp_timer.check()  # self._prev_work is assigned, next time the check is called, it will be executed
        return fsp_timer.check()  # Since 8>5, we will see flag set to True

    @spawn_threads_and_init_comms(world_size=2)
    def test_full_sync_pt_multi_process_check_false(self) -> None:
        result = self._full_sync_worker_without_timeout()
        # Both processes should return False
        self.assertFalse(result)

    @spawn_threads_and_init_comms(world_size=2)
    def test_full_sync_pt_multi_process_edgecase(self) -> None:
        result = self._full_sync_worker_with_timeout(5)
        # Both processes should return True
        self.assertTrue(result)

        result = self._full_sync_worker_with_timeout(4)
        # Both processes should return False
        self.assertFalse(result)
