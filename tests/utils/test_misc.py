#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtnt.utils.misc import days_to_secs


class MiscTest(unittest.TestCase):
    def test_days_to_secs(self) -> None:
        self.assertIsNone(days_to_secs(None))
        self.assertEqual(days_to_secs(1), 60 * 60 * 24)
        with self.assertRaises(ValueError):
            days_to_secs(-1)
