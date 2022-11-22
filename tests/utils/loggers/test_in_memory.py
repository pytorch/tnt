#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest
from collections import OrderedDict
from contextlib import contextmanager
from io import StringIO

from torchtnt.utils.loggers.in_memory import InMemoryLogger


class InMemoryLoggerTest(unittest.TestCase):
    def test_in_memory_log(self) -> None:
        logger = InMemoryLogger()
        logger.log(name="metric1", data=123.0, step=0)
        logger.log(name="metric1", data=456.0, step=1)
        logger.log(name="metric1", data=789.0, step=2)
        # Test flushing.
        with captured_output() as (out, err):
            logger.flush()
        self.assertTrue(out.getvalue().startswith("OrderedDict(["))
        self.assertEqual(err.getvalue(), "")
        logger.log_dict(payload={"metric2": 1.0, "metric3": 2.0}, step=3)
        # Check the buffer directly.
        buf = logger.log_buffer
        self.assertEqual(len(buf), 4)
        self.assertEqual(buf[0]["metric1"], 123.0)
        self.assertEqual(buf[0]["step"], 0)
        self.assertEqual(buf[1]["metric1"], 456.0)
        self.assertEqual(buf[1]["step"], 1)
        self.assertEqual(buf[2]["metric1"], 789.0)
        self.assertEqual(buf[2]["step"], 2)
        self.assertEqual(buf[3]["metric2"], 1.0)
        self.assertEqual(buf[3]["metric3"], 2.0)
        self.assertEqual(buf[3]["step"], 3)
        # Test flushing.
        with captured_output() as (out, err):
            logger.flush()
        self.assertTrue(out.getvalue().startswith("OrderedDict(["))
        self.assertEqual(err.getvalue(), "")
        # Closing the log clears the buffer.
        logger.close()
        self.assertEqual(logger.log_buffer, OrderedDict([]))


@contextmanager
def captured_output() -> None:
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err
