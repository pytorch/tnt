#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtnt.utils.distributed import get_global_rank

from torchtnt.utils.test_utils import skip_if_not_distributed, spawn_multi_process


class TestUtilsTest(unittest.TestCase):
    @staticmethod
    def _test_method(offset_arg: int, offset_kwarg: int) -> int:
        return get_global_rank() + offset_arg - offset_kwarg

    @skip_if_not_distributed
    def test_spawn_multi_process(self) -> None:
        mp_list = spawn_multi_process(2, "gloo", self._test_method, 3, offset_kwarg=2)
        self.assertEqual(mp_list, [1, 2])
