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


class TestUnit(_AppStateMixin):
    def __init__(self):
        super().__init__()
        self.module_a = nn.Linear(1, 1)
        self.loss_fn_b = nn.CrossEntropyLoss()
        self.optimizer_c = torch.optim.SGD(self.module_a.parameters(), lr=0.1)
        self.lr_scheduler_d = torch.optim.lr_scheduler.StepLR(
            self.optimizer_c, step_size=30, gamma=0.1
        )


class AppStateMixinTest(unittest.TestCase):
    def test_tracked_modules(self) -> None:
        """
        Test setting, getting, and deleting tracked_modules
        """

        my_unit = TestUnit()

        # assert that the attributes are stored in tracked_modules
        self.assertEqual(my_unit.tracked_modules()["module_a"], my_unit.module_a)
        self.assertEqual(my_unit.tracked_modules()["loss_fn_b"], my_unit.loss_fn_b)

        # delete the attributes
        my_unit.module_a = None
        my_unit.loss_fn_b = None

        # the attributes should be removed from tracked_modules
        self.assertFalse("module_a" in my_unit.tracked_modules())
        self.assertFalse("loss_fn_b" in my_unit.tracked_modules())

    def test_tracked_optimizers(self) -> None:
        """
        Test setting, getting, and deleting tracked_optimizers
        """
        my_unit = TestUnit()

        # assert that the attribute is stored in tracked_optimizers
        self.assertEqual(
            my_unit.tracked_optimizers()["optimizer_c"], my_unit.optimizer_c
        )

        # delete the attribute
        my_unit.optimizer_c = None

        # the attribute should be removed from tracked_optimizers
        self.assertFalse("optimizer_c" in my_unit.tracked_optimizers())

    def test_tracked_lr_schedulers(self) -> None:
        """
        Test setting, getting, and deleting tracked_lr_schedulers
        """

        my_unit = TestUnit()

        # assert that the attribute is stored in tracked_lr_schedulers
        self.assertEqual(
            my_unit.tracked_lr_schedulers()["lr_scheduler_d"], my_unit.lr_scheduler_d
        )

        # delete the attribute
        my_unit.lr_scheduler_d = None

        # the attribute should be removed from tracked_lr_schedulers
        self.assertFalse("lr_scheduler_d" in my_unit.tracked_lr_schedulers())

    def test_app_state(self) -> None:
        """
        Test the app_state method
        """

        my_unit = TestUnit()

        # the attributes should be in app_state
        for key in ("module_a", "loss_fn_b", "optimizer_c", "lr_scheduler_d"):
            self.assertTrue(key in my_unit.app_state())

        # delete the attributes
        my_unit.module_a = None
        my_unit.loss_fn_b = None
        my_unit.optimizer_c = None
        my_unit.lr_scheduler_d = None

        # the attributes should no longer be in app_state
        for key in ("module_a", "loss_fn_b", "optimizer_c", "lr_scheduler_d"):
            self.assertFalse(key in my_unit.app_state())

    def test_reassigning_attributes(self) -> None:
        """
        Test reassigning attributes to a different type
        """

        my_unit = TestUnit()

        # create new objects
        module_e = nn.Linear(1, 1)
        loss_fn_f = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.SGD(module_e.parameters(), lr=0.1)
        lr_scheduler_h = torch.optim.lr_scheduler.StepLR(
            optimizer_g, step_size=30, gamma=0.1
        )

        # reassigning module_a to be an optimizer should work
        self.assertTrue("module_a" in my_unit.tracked_modules())
        my_unit.module_a = optimizer_g
        self.assertTrue("module_a" not in my_unit.tracked_modules())
        self.assertTrue("module_a" in my_unit.tracked_optimizers())

        # reassigning optimizer_c to be an lr_scheduler should work
        self.assertTrue("optimizer_c" in my_unit.tracked_optimizers())
        my_unit.optimizer_c = lr_scheduler_h
        self.assertTrue("optimizer_c" not in my_unit.tracked_optimizers())
        self.assertTrue("optimizer_c" in my_unit.tracked_lr_schedulers())

        # reassigning lr_scheduler_d to be a nn.module should work
        self.assertTrue("lr_scheduler_d" in my_unit.tracked_lr_schedulers())
        my_unit.lr_scheduler_d = loss_fn_f
        self.assertTrue("lr_scheduler_d" not in my_unit.tracked_lr_schedulers())
        self.assertTrue("lr_scheduler_d" in my_unit.tracked_modules())
