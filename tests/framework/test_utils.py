#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import unittest
from typing import Dict, Iterator
from unittest.mock import MagicMock, patch

import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torchtnt.framework.state import State
from torchtnt.framework.utils import (
    _find_optimizers_for_module,
    _step_requires_iterator,
    get_timing_context,
)
from torchtnt.utils.env import init_from_env
from torchtnt.utils.test_utils import spawn_multi_process
from torchtnt.utils.timer import Timer


class UtilsTest(unittest.TestCase):
    cuda_available: bool = torch.cuda.is_available()
    distributed_available: bool = torch.distributed.is_available()

    def test_step_func_requires_iterator(self) -> None:
        class Foo:
            def bar(self, state: State, data: object) -> object:
                return data

            def baz(self, state: State, data: Iterator[torch.Tensor]) -> object:
                pass

        def dummy(a: int, b: str, data: Iterator[str]) -> None:
            pass

        foo = Foo()

        self.assertFalse(_step_requires_iterator(foo.bar))
        self.assertTrue(_step_requires_iterator(foo.baz))
        self.assertTrue(_step_requires_iterator(dummy))

    @patch("torchtnt.framework.utils.record_function")
    def test_get_timing_context(self, mock_record_function: MagicMock) -> None:
        state = MagicMock()
        state.timer = None

        ctx = get_timing_context(state, "a")
        with ctx:
            time.sleep(1)
        mock_record_function.assert_called_with("a")

        state.timer = Timer()
        ctx = get_timing_context(state, "b")
        with ctx:
            time.sleep(1)
        self.assertTrue("b" in state.timer.recorded_durations.keys())
        mock_record_function.assert_called_with("b")

    def test_find_optimizers_for_module(self) -> None:
        module1 = torch.nn.Linear(10, 10)
        module2 = torch.nn.Linear(10, 10)
        optim1 = torch.optim.Adam(module1.parameters())
        optim2 = torch.optim.Adagrad(module2.parameters())

        opts: Dict[str, Optimizer] = {"optim1": optim1, "optim2": optim2}
        optimizers = _find_optimizers_for_module(module1, opts)
        optim_name, _ = optimizers[0]
        self.assertEqual(optim_name, "optim1")
        optimizers = _find_optimizers_for_module(module2, opts)
        optim_name, _ = optimizers[0]
        self.assertEqual(optim_name, "optim2")

    @unittest.skipUnless(
        condition=distributed_available, reason="Torch distributed is needed to run"
    )
    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_find_optimizers_for_FSDP_module(self) -> None:
        spawn_multi_process(2, "nccl", self._find_optimizers_for_FSDP_module)

    @staticmethod
    def _find_optimizers_for_FSDP_module() -> None:
        device = init_from_env()
        module1 = FSDP(torch.nn.Linear(10, 10).to(device))
        module2 = torch.nn.Linear(10, 10)
        optim1 = torch.optim.Adam(module1.parameters())
        optim2 = torch.optim.Adagrad(module2.parameters())

        opts: Dict[str, Optimizer] = {"optim1": optim1, "optim2": optim2}
        optim_list = _find_optimizers_for_module(module1, opts)
        optim_name, _ = optim_list[0]

        tc = unittest.TestCase()
        tc.assertEqual(optim_name, "optim1")
        optim_list = _find_optimizers_for_module(module2, opts)
        optim_name, _ = optim_list[0]
        tc.assertEqual(optim_name, "optim2")
