#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Literal, Optional, Tuple
from unittest.mock import MagicMock, patch

import torch
from torchtnt.runner._test_utils import generate_random_dataloader

from torchtnt.runner.auto_unit import AutoTrainUnit
from torchtnt.runner.state import State
from torchtnt.runner.train import init_train_state, train
from torchtnt.utils import init_from_env


class TestAutoUnit(unittest.TestCase):
    cuda_available = torch.cuda.is_available()

    def test_app_state_mixin(self) -> None:
        """
        Test that app_state, tracked_optimizers, tracked_lr_schedulers are set as expected with AutoTrainUnit
        """
        my_module = torch.nn.Linear(2, 2)
        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            my_optimizer, gamma=0.9
        )
        auto_train_unit = DummyAutoTrainUnit(
            module=my_module,
            optimizer=my_optimizer,
            lr_scheduler=my_lr_scheduler,
            log_frequency_steps=100,
            precision="fp16",
        )
        self.assertEqual(auto_train_unit.tracked_modules()["module"], my_module)
        self.assertEqual(
            auto_train_unit.tracked_optimizers()["optimizer"], my_optimizer
        )
        self.assertEqual(
            auto_train_unit.tracked_lr_schedulers()["lr_scheduler"], my_lr_scheduler
        )
        self.assertTrue(
            isinstance(
                auto_train_unit.tracked_misc_statefuls()["grad_scaler"],
                torch.cuda.amp.GradScaler,
            )
        )
        for key in ("module", "optimizer", "lr_scheduler", "grad_scaler"):
            self.assertTrue(key in auto_train_unit.app_state())

    @unittest.skipUnless(
        condition=(not cuda_available), reason="This test shouldn't run on a GPU host."
    )
    def test_lr_scheduler_step(self) -> None:
        """
        Test that the lr scheduler is stepped every optimizer step when step_lr_interval="step"
        """
        my_module = torch.nn.Linear(2, 2)
        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)
        my_lr_scheduler = MagicMock()
        auto_train_unit = DummyAutoTrainUnit(
            module=my_module,
            optimizer=my_optimizer,
            lr_scheduler=my_lr_scheduler,
            step_lr_interval="step",
            log_frequency_steps=100,
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3
        expected_steps_per_epoch = dataset_len / batch_size * max_epochs

        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=train_dl, max_epochs=max_epochs)
        train(state, auto_train_unit)
        self.assertTrue(my_lr_scheduler.step.call_count, expected_steps_per_epoch)

    @unittest.skipUnless(
        condition=(not cuda_available), reason="This test shouldn't run on a GPU host."
    )
    def test_lr_scheduler_epoch(self) -> None:
        """
        Test that the lr scheduler is stepped every epoch when step_lr_interval="epoch"
        """
        my_module = torch.nn.Linear(2, 2)
        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)
        my_lr_scheduler = MagicMock()
        auto_train_unit = DummyAutoTrainUnit(
            module=my_module,
            optimizer=my_optimizer,
            lr_scheduler=my_lr_scheduler,
            step_lr_interval="epoch",
            log_frequency_steps=100,
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3

        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)

        state = init_train_state(dataloader=train_dl, max_epochs=max_epochs)
        train(state, auto_train_unit)
        self.assertTrue(my_lr_scheduler.step.call_count, max_epochs)

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    @patch("torch.autocast")
    def test_mixed_precision_fp16(self, mock_autocast) -> None:
        """
        Test that the mixed precision autocast context is called when fp16 precision is set
        """
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2).to(device)
        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            my_optimizer, gamma=0.9
        )
        auto_train_unit = DummyAutoTrainUnit(
            module=my_module,
            optimizer=my_optimizer,
            lr_scheduler=my_lr_scheduler,
            log_frequency_steps=100,
            precision="fp16",
        )
        dummy_data = (torch.ones(2, 2), torch.ones(2, 2))
        auto_train_unit.train_step(state=MagicMock(), data=dummy_data)
        mock_autocast.assert_called_with(
            device_type="cuda", dtype=torch.float16, enabled=True
        )

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    @patch("torch.autocast")
    def test_mixed_precision_bf16(self, mock_autocast) -> None:
        """
        Test that the mixed precision autocast context is called when bf16 precision is set
        """
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2).to(device)
        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            my_optimizer, gamma=0.9
        )
        auto_train_unit = DummyAutoTrainUnit(
            module=my_module,
            optimizer=my_optimizer,
            lr_scheduler=my_lr_scheduler,
            log_frequency_steps=100,
            precision="bf16",
        )
        dummy_data = (torch.ones(2, 2), torch.ones(2, 2))
        auto_train_unit.train_step(state=MagicMock(), data=dummy_data)
        mock_autocast.assert_called_with(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        )

    def test_mixed_precision_invalid_str(self) -> None:
        """
        Test that an exception is raised with an invalid precision string
        """
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2).to(device)
        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            my_optimizer, gamma=0.9
        )
        with self.assertRaisesRegex(ValueError, "Precision f16 not supported"):
            _ = DummyAutoTrainUnit(
                module=my_module,
                optimizer=my_optimizer,
                lr_scheduler=my_lr_scheduler,
                log_frequency_steps=100,
                precision="f16",
            )


class DummyAutoTrainUnit(AutoTrainUnit[Tuple[torch.tensor, torch.tensor]]):
    def compute_loss(
        self, state: State, data: Tuple[torch.tensor, torch.tensor]
    ) -> Tuple[torch.Tensor, Any]:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs
