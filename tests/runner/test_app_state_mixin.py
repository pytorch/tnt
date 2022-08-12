#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn
from torchtnt.runner.unit import _AppStateMixin


class AppStateMixinTest(unittest.TestCase):
    def test_app_state_mixin(self) -> None:
        """
        Test AppStateMixin
        """

        class MyUnit(_AppStateMixin):
            def __init__(self, input_dim: int):
                super().__init__()
                self.module_a = nn.Linear(input_dim, 1)
                self.loss_fn_b = nn.CrossEntropyLoss()
                self.optimizer_c = torch.optim.SGD(self.module_a.parameters(), lr=0.1)
                self.lr_scheduler_d = torch.optim.lr_scheduler.StepLR(
                    self.optimizer_c, step_size=30, gamma=0.1
                )

        my_unit = MyUnit(input_dim=1)

        # assert that the attributes are stored in tracked_modules/optimizers/lr_schedulers
        self.assertTrue(isinstance(my_unit.tracked_modules()["module_a"], nn.Linear))
        self.assertTrue(
            isinstance(my_unit.tracked_modules()["loss_fn_b"], nn.CrossEntropyLoss)
        )

        self.assertTrue(
            isinstance(my_unit.tracked_optimizers()["optimizer_c"], torch.optim.SGD)
        )

        self.assertTrue(
            isinstance(
                my_unit.tracked_lr_schedulers()["lr_scheduler_d"],
                torch.optim.lr_scheduler.StepLR,
            )
        )

        # test app_state method
        for key in ["module_a", "loss_fn_b", "optimizer_c", "lr_scheduler_d"]:
            self.assertTrue(key in my_unit.app_state())

        # delete the attributes
        delattr(my_unit, "module_a")
        delattr(my_unit, "loss_fn_b")
        delattr(my_unit, "optimizer_c")
        delattr(my_unit, "lr_scheduler_d")

        # the attributes should be removed from the tracked objects
        self.assertFalse("module_a" in my_unit.tracked_modules())
        self.assertFalse("loss_fn_b" in my_unit.tracked_modules())
        self.assertFalse("optimizer_c" in my_unit.tracked_optimizers())
        self.assertFalse("lr_scheduler_d" in my_unit.tracked_lr_schedulers())

        # test app_state method after deleting the attributes
        for key in ["module_a", "loss_fn_b", "optimizer_c", "lr_scheduler_d"]:
            self.assertFalse(key in my_unit.app_state())
