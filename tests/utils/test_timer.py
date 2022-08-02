#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import unittest
from random import random

import torch
from torchtnt.utils.timer import Timer


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
        with timer.time():
            time.sleep(intervals[0])
        self.assertEqual(timer.interval_time_seconds, timer.total_time_seconds)
        self.assert_within_tolerance(timer.total_time_seconds, intervals[0])

        total = timer.total_time_seconds

        # Ensure total accumulates with multiple context managers
        with timer.time():
            time.sleep(intervals[1])
        self.assertLess(timer.interval_time_seconds, timer.total_time_seconds)
        self.assert_within_tolerance(timer.interval_time_seconds, intervals[1])
        self.assert_within_tolerance(timer.total_time_seconds, total + intervals[1])

        total = timer.total_time_seconds

        # Make sure nested context managers work properly
        with self.assertWarns(Warning):
            with timer.time():
                with timer.time():
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
