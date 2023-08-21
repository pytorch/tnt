#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import uuid

import torch
from torch.distributed import launcher
from torchtnt.utils.test_utils import skip_if_asan

from ..main import main


MIN_NODES = 1
MAX_NODES = 1
PROC_PER_NODE = 2


class TorchrecExampleTest(unittest.TestCase):
    @skip_if_asan
    # pyre-fixme[56]: Pyre was not able to infer the type of argument `not
    #  torch.cuda.is_available()` to decorator factory `unittest.skipIf`.
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Skip when CUDA is not available",
    )
    def test_torchrec_example(self) -> None:
        lc = launcher.LaunchConfig(
            min_nodes=MIN_NODES,
            max_nodes=MAX_NODES,
            nproc_per_node=PROC_PER_NODE,
            run_id=str(uuid.uuid4()),
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:0",
            max_restarts=0,
            monitor_interval=1,
        )

        launcher.elastic_launch(config=lc, entrypoint=main)([])
