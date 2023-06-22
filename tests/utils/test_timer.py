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

import torch
import torch.distributed as dist
import torch.distributed.launcher as launcher
from torchtnt.utils.test_utils import get_pet_launch_config
from torchtnt.utils.timer import (
    FullSyncPeriodicTimer,
    get_durations_histogram,
    get_synced_durations_histogram,
    get_timer_summary,
    Timer,
    VerboseTimer,
)


class TimerTest(unittest.TestCase):
    def assert_within_tolerance(
        self, expected: float, actual: float, percent_tolerance: float = 10
    ) -> None:
        """Assert that a value is correct within a percent tolerance"""
        error = abs(expected - actual)
        tolerance = expected * (percent_tolerance / 100)
        self.assertLess(error, tolerance)

    def test_timer_basic(self) -> None:
        """Basic test of the timer class"""

        # Generate 3 intervals between 0.5 and 2 seconds
        intervals = [(random() * 1.5) + 0.5 for _ in range(3)]

        # Basic start and stop test
        timer = Timer()
        timer.start()
        time.sleep(intervals[0])
        timer.stop()
        self.assertEqual(timer.interval_time_seconds, timer.total_time_seconds)
        self.assert_within_tolerance(timer.total_time_seconds, intervals[0])

        total = timer.total_time_seconds

        # Test that interval time resets and total time accumulates
        timer.start()
        time.sleep(intervals[1])
        timer.stop()
        self.assertLess(timer.interval_time_seconds, timer.total_time_seconds)
        self.assert_within_tolerance(timer.interval_time_seconds, intervals[1])
        self.assert_within_tolerance(timer.total_time_seconds, total + intervals[1])

        # Test that reset works properly
        timer.reset()
        timer.start()
        time.sleep(intervals[2])
        timer.stop()
        self.assertEqual(timer.interval_time_seconds, timer.total_time_seconds)
        self.assert_within_tolerance(timer.total_time_seconds, intervals[2])

    def test_verbose_timer(self) -> None:
        timer = VerboseTimer()

        # TODO: maybe use a magic mock would be better?
        class MockLogger:
            def __init__(self):
                self._info = []

            def info(self, msg: str) -> None:
                self._info.append(msg)

        mock_logger = MockLogger()
        with timer.time("hi", logger=mock_logger):
            self.assertTrue("Starting hi" in mock_logger._info)
            time.sleep(1)
            self.assertTrue(
                not any([x.startswith("Stopping hi") for x in mock_logger._info])
            )
        # Assert that end hi has been called
        self.assertTrue(any([x.startswith("Stopping hi") for x in mock_logger._info]))

    def test_extra_starts_stops(self) -> None:
        """Test behavior with extra starts and stops"""

        # Generate 2 intervals between 0.5 and 2 seconds
        intervals = [(random() * 1.5) + 0.5 for _ in range(2)]

        # Test multiple starts
        timer = Timer()
        timer.start()
        time.sleep(intervals[0])
        with self.assertWarns(Warning):
            timer.start()
        timer.stop()
        self.assertEqual(timer.interval_time_seconds, timer.total_time_seconds)
        self.assert_within_tolerance(timer.total_time_seconds, intervals[0])

        total = timer.total_time_seconds

        # Test multiple stops
        timer.start()
        time.sleep(intervals[1])
        timer.stop()
        with self.assertWarns(Warning):
            timer.stop()
        self.assertLess(timer.interval_time_seconds, timer.total_time_seconds)
        self.assert_within_tolerance(timer.total_time_seconds, total + intervals[1])

    def test_missing_starts_stops(self) -> None:
        """Test behavior with missing starts and stops"""

        # Generate 1 interval between 0.5 and 2 seconds
        intervals = [(random() * 1.5) + 0.5 for _ in range(1)]
        timer = Timer()

        # Test stop without start
        timer.reset()
        with self.assertWarns(Warning):
            timer.stop()
        self.assertEqual(timer.interval_time_seconds, 0)
        self.assertEqual(timer.total_time_seconds, 0)

        # Test start without stop
        timer.reset()
        timer.start()
        time.sleep(intervals[0])

        # Saving values outside of asserts to reduce error from overhead
        interval_time = timer.interval_time_seconds
        total_time = timer.total_time_seconds

        self.assert_within_tolerance(interval_time, intervals[0])
        self.assert_within_tolerance(total_time, intervals[0])

    def test_timer_state_dict(self) -> None:
        """Test the statefulness of the timer class"""

        # Generate 3 intervals between 0.5 and 2 seconds
        intervals = [(random() * 1.5) + 0.5 for _ in range(3)]

        # Test saving state dict
        timer = Timer()
        timer.start()
        time.sleep(intervals[0])
        timer.stop()

        interval = timer.interval_time_seconds
        total = timer.total_time_seconds

        state_dict = timer.state_dict()
        self.assertEqual(len(state_dict), 3)
        self.assertIn("interval_start_time", state_dict)
        self.assertIn("interval_stop_time", state_dict)
        self.assertIn("total_time_seconds", state_dict)

        # Test loading state dict, ensure interval is preserved and total accumulates
        del timer
        timer = Timer()
        timer.load_state_dict(state_dict)
        self.assert_within_tolerance(timer.interval_time_seconds, interval)

        timer.start()
        time.sleep(intervals[1])
        timer.stop()
        self.assert_within_tolerance(timer.total_time_seconds, total + intervals[1])

        total = timer.total_time_seconds

        # Test saving state dict on running timer, ensure timer is paused
        timer.start()
        time.sleep(intervals[2])
        with self.assertRaisesRegex(
            Exception, "Timer must be paused before creating state_dict."
        ):
            state_dict = timer.state_dict()

    def test_timer_context_manager(self) -> None:
        """Test the context manager in the timer class"""

        # Generate 3 intervals between 0.5 and 2 seconds
        intervals = [(random() * 1.5) + 0.5 for _ in range(3)]

        # Basic test of context manager
        timer = Timer()
        with timer.time("action_1"):
            time.sleep(intervals[0])
        self.assertEqual(timer.interval_time_seconds, timer.total_time_seconds)
        self.assert_within_tolerance(timer.total_time_seconds, intervals[0])

        total = timer.total_time_seconds

        # Ensure total accumulates with multiple context managers
        with timer.time("action_2"):
            time.sleep(intervals[1])
        self.assertLess(timer.interval_time_seconds, timer.total_time_seconds)
        self.assert_within_tolerance(timer.interval_time_seconds, intervals[1])
        self.assert_within_tolerance(timer.total_time_seconds, total + intervals[1])

        total = timer.total_time_seconds

        # Make sure nested context managers work properly
        with self.assertWarns(Warning):
            with timer.time("action_3"):
                with timer.time("action_4"):
                    time.sleep(intervals[2])
        self.assertLess(timer.interval_time_seconds, timer.total_time_seconds)
        self.assert_within_tolerance(timer.interval_time_seconds, intervals[2])
        self.assert_within_tolerance(timer.total_time_seconds, total + intervals[2])

    @unittest.skipUnless(
        condition=torch.cuda.is_available(), reason="This test needs a GPU host to run."
    )
    def test_timer_synchronize(self) -> None:
        """Make sure that torch.cuda.synchronize() is called when GPU is present."""

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        timer = Timer()

        # Do not explicitly call synchronize, timer must call it for test to pass.
        timer.start()
        start_event.record()

        time.sleep(0.5)

        end_event.record()
        timer.stop()

        # torch.cuda.synchronize() has to be called to compute the elapsed time.
        # Otherwise, there will be runtime error.
        elapsed_time_ms = start_event.elapsed_time(end_event)

        self.assertEqual(timer.interval_time_seconds, timer.total_time_seconds)
        self.assert_within_tolerance(timer.total_time_seconds, 0.5)
        self.assert_within_tolerance(timer.total_time_seconds, elapsed_time_ms / 1000)

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
        dist.init_process_group("gloo")
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
    def test_get_synced_durations_histogram_multi_process(self) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(
            config, entrypoint=self._get_synced_durations_histogram_multi_process
        )()


class FullSyncPeriodicTimerTest(unittest.TestCase):
    @classmethod
    def _full_sync_worker_without_timeout(cls) -> bool:
        dist.init_process_group("gloo")
        process_group = dist.group.WORLD
        interval_threshold = timedelta(seconds=5)
        fsp_timer = FullSyncPeriodicTimer(interval_threshold, process_group)
        return fsp_timer.check()

    @classmethod
    def _full_sync_worker_with_timeout(cls, timeout: int) -> bool:
        dist.init_process_group("gloo")
        process_group = dist.group.WORLD
        interval_threshold = timedelta(seconds=5)
        fsp_timer = FullSyncPeriodicTimer(interval_threshold, process_group)
        time.sleep(timeout)
        fsp_timer.check()  # self._prev_work is assigned, next time the check is called, it will be executed
        return fsp_timer.check()  # Since 8>5, we will see flag set to True

    def test_full_sync_pt_multi_process_check_false(self) -> None:
        config = get_pet_launch_config(2)
        # Launch 2 worker processes. Each will check if time diff > interval threshold
        result = launcher.elastic_launch(
            config, entrypoint=self._full_sync_worker_without_timeout
        )()
        # Both processes should return False
        self.assertFalse(result[0])
        self.assertFalse(result[1])

    def test_full_sync_pt_multi_process_check_true(self) -> None:
        config = get_pet_launch_config(2)
        # Launch 2 worker processes. Each will check time diff > interval threshold
        result = launcher.elastic_launch(
            config, entrypoint=self._full_sync_worker_with_timeout
        )(8)
        # Both processes should return True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

    def test_full_sync_pt_multi_process_edgecase(self) -> None:
        config = get_pet_launch_config(2)
        # Launch 2 worker processes. Each will check time diff >= interval threshold
        result = launcher.elastic_launch(
            config, entrypoint=self._full_sync_worker_with_timeout
        )(5)

        # Both processes should return True
        self.assertTrue(result[0])
        self.assertTrue(result[1])

        # Launch 2 worker processes. Each will check time diff >= interval threshold
        result = launcher.elastic_launch(
            config, entrypoint=self._full_sync_worker_with_timeout
        )(4)

        # Both processes should return False
        self.assertFalse(result[0])
        self.assertFalse(result[1])
