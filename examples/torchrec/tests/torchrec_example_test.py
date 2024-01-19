#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchtnt.utils.test_utils import skip_if_asan, skip_if_not_gpu, spawn_multi_process

from ..main import main


class TorchrecExampleTest(unittest.TestCase):
    @skip_if_asan
    @skip_if_not_gpu
    def test_torchrec_example(self) -> None:
        spawn_multi_process(2, "nccl", main, [])
