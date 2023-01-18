#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Tuple

import torch
import torch._dynamo
from torch.distributed import launcher
from torch.nn.parallel import DistributedDataParallel

from torchtnt.framework.auto_ddp_unit import AutoDDPUnit
from torchtnt.framework.state import State
from torchtnt.utils import init_from_env, TLRScheduler
from torchtnt.utils.test_utils import get_pet_launch_config


class TestAutoDDPUnit(unittest.TestCase):
    cuda_available = torch.cuda.is_available()

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_auto_ddp_unit_ddp(self) -> None:
        """
        Launch various tests for the AutoDDPUnit
        """
        config = get_pet_launch_config(2)
        launcher.elastic_launch(config, entrypoint=self._test_app_state_mixin)()

    @staticmethod
    def _test_app_state_mixin() -> None:
        """
        Test AppStateMixin is correctly capturing the module, optimizer, lr_scheduler
        """

        device = init_from_env()
        my_module = torch.nn.Linear(2, 2).to(device)

        auto_ddp_unit = DummyAutoDDPUnit(
            module=my_module,
            device=device,
            gradient_accumulation_steps=2,
        )
        tc = unittest.TestCase()

        for key in ("module", "optimizer", "lr_scheduler"):
            tc.assertTrue(key in auto_ddp_unit.app_state())

    @staticmethod
    def _test_ddp_wrap() -> None:
        """
        Test that the module is correctly wrapped in DDP
        """

        device = init_from_env()
        my_module = torch.nn.Linear(2, 2).to(device)

        auto_ddp_unit = DummyAutoDDPUnit(
            module=my_module,
            device=device,
            gradient_accumulation_steps=2,
        )
        tc = unittest.TestCase()

        tc.assertTrue(isinstance(auto_ddp_unit.module, DistributedDataParallel))


Batch = Tuple[torch.tensor, torch.tensor]


class DummyAutoDDPUnit(AutoDDPUnit[Batch]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, state: State, data: Batch) -> Tuple[torch.Tensor, Any]:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        my_optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            my_optimizer, gamma=0.9
        )
        return my_optimizer, my_lr_scheduler
