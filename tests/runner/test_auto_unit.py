#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import torch
from parameterized import parameterized
from torch.distributed import launcher
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtnt.runner._test_utils import generate_random_dataloader

from torchtnt.runner.auto_unit import AutoTrainUnit
from torchtnt.runner.state import State
from torchtnt.runner.train import init_train_state, train
from torchtnt.utils import init_from_env
from torchtnt.utils.test_utils import get_pet_launch_config


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

    def test_lr_scheduler_step(self) -> None:
        """
        Test that the lr scheduler is stepped every optimizer step when step_lr_interval="step"
        """
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2, device=device)
        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)
        my_lr_scheduler = MagicMock()
        auto_train_unit = DummyAutoTrainUnit(
            module=my_module,
            optimizer=my_optimizer,
            lr_scheduler=my_lr_scheduler,
            step_lr_interval="step",
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3
        expected_steps_per_epoch = dataset_len / batch_size * max_epochs

        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=train_dl, max_epochs=max_epochs)
        train(state, auto_train_unit)
        self.assertEqual(my_lr_scheduler.step.call_count, expected_steps_per_epoch)

    def test_lr_scheduler_epoch(self) -> None:
        """
        Test that the lr scheduler is stepped every epoch when step_lr_interval="epoch"
        """
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2, device=device)
        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)
        my_lr_scheduler = MagicMock()
        auto_train_unit = DummyAutoTrainUnit(
            module=my_module,
            optimizer=my_optimizer,
            lr_scheduler=my_lr_scheduler,
            step_lr_interval="epoch",
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3

        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)

        state = init_train_state(dataloader=train_dl, max_epochs=max_epochs)
        train(state, auto_train_unit)
        self.assertEqual(my_lr_scheduler.step.call_count, max_epochs)

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
                precision="f16",
            )

    @parameterized.expand(
        [
            [1],
            [2],
            [4],
            [5],
        ]
    )
    def test_num_optimizer_steps_completed(self, gradient_accumulation_steps) -> None:
        """
        Test the num_optimizer_steps_completed property of AutoTrainUnit
        """
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2).to(device)
        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)

        input_dim = 2
        dataset_len = 16
        batch_size = 2
        max_epochs = 1

        auto_train_unit = DummyAutoTrainUnit(
            module=my_module,
            optimizer=my_optimizer,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        expected_opt_steps_per_epoch = math.ceil(
            dataset_len / batch_size / gradient_accumulation_steps
        )

        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=train_dl, max_epochs=max_epochs)
        train(state, auto_train_unit)
        self.assertTrue(
            auto_train_unit.num_optimizer_steps_completed, expected_opt_steps_per_epoch
        )

    def test_log_frequency_steps_exception(self) -> None:
        """
        Test that an exception is raised when log_frequency_steps is < 1
        """
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2).to(device)
        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            my_optimizer, gamma=0.9
        )
        with self.assertRaisesRegex(
            ValueError, "log_frequency_steps must be > 0. Got 0"
        ):
            _ = DummyAutoTrainUnit(
                module=my_module,
                optimizer=my_optimizer,
                lr_scheduler=my_lr_scheduler,
                log_frequency_steps=0,
            )

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_no_sync(self) -> None:
        """
        Test that the no_sync autocast context is correctly applied when using gradient accumulation
        """
        config = get_pet_launch_config(2)
        launcher.elastic_launch(config, entrypoint=self._test_ddp_no_sync)()
        launcher.elastic_launch(config, entrypoint=self._test_fsdp_no_sync)()

    @staticmethod
    def _test_ddp_no_sync() -> None:
        """
        Test that the no_sync autocast context is correctly applied when using gradient accumulation and DDP
        """

        device = init_from_env()
        my_module = torch.nn.Linear(2, 2).to(device)
        my_module = DDP(my_module, device_ids=[device.index])

        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            my_optimizer, gamma=0.9
        )
        auto_train_unit = DummyAutoTrainUnit(
            module=my_module,
            optimizer=my_optimizer,
            lr_scheduler=my_lr_scheduler,
            device=device,
            gradient_accumulation_steps=2,
        )

        dummy_data = (torch.ones(2, 2), torch.ones(2, 2))
        state = init_train_state(dataloader=MagicMock(), max_epochs=1)

        # for the first step no_sync should be called since we accumulate gradients
        with patch.object(my_module, "no_sync") as no_sync_mock:
            auto_train_unit.train_step(state=state, data=dummy_data)
            no_sync_mock.assert_called()

        state.train_state.progress.increment_step()
        # for the second step no_sync should not be called since we run optimizer step
        with patch.object(my_module, "no_sync") as no_sync_mock:
            auto_train_unit.train_step(state=state, data=dummy_data)
            no_sync_mock.assert_not_called()

    @staticmethod
    def _test_fsdp_no_sync() -> None:
        """
        Test that the no_sync autocast context is correctly applied when using gradient accumulation and FSDP
        """

        device = init_from_env()
        my_module = torch.nn.Linear(2, 2).to(device)
        my_module = FSDP(my_module)

        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            my_optimizer, gamma=0.9
        )
        auto_train_unit = DummyAutoTrainUnit(
            module=my_module,
            optimizer=my_optimizer,
            lr_scheduler=my_lr_scheduler,
            device=device,
            gradient_accumulation_steps=2,
        )

        dummy_data = (torch.ones(2, 2), torch.ones(2, 2))
        state = init_train_state(dataloader=MagicMock(), max_epochs=1)

        # for the first step no_sync should be called since we accumulate gradients
        with patch.object(my_module, "no_sync") as no_sync_mock:
            auto_train_unit.train_step(state=state, data=dummy_data)
            no_sync_mock.assert_called()

        state.train_state.progress.increment_step()
        # for the second step no_sync should not be called since we run optimizer step
        with patch.object(my_module, "no_sync") as no_sync_mock:
            auto_train_unit.train_step(state=state, data=dummy_data)
            no_sync_mock.assert_not_called()

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_fsdp_fp16_pytorch_version(self) -> None:
        """
        Test that a RuntimeError is thrown when using FSDP, fp16 precision, and PyTorch < v1.12
        """
        config = get_pet_launch_config(2)
        launcher.elastic_launch(
            config, entrypoint=self._test_fsdp_fp16_pytorch_version
        )()

    @staticmethod
    def _test_fsdp_fp16_pytorch_version() -> None:
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2).to(device)
        my_module = FSDP(my_module)

        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            my_optimizer, gamma=0.9
        )
        tc = unittest.TestCase()
        with patch(
            "torchtnt.runner.auto_unit.is_torch_version_geq_1_12", return_value=False
        ), tc.assertRaisesRegex(
            RuntimeError,
            "Using float16 precision with torch.distributed.fsdp.FullyShardedDataParallel requires torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler from PyTorch 1.12.",
        ):
            _ = DummyAutoTrainUnit(
                module=my_module,
                optimizer=my_optimizer,
                lr_scheduler=my_lr_scheduler,
                device=device,
                precision="fp16",
            )


class DummyAutoTrainUnit(AutoTrainUnit[Tuple[torch.tensor, torch.tensor]]):
    def compute_loss(
        self, state: State, data: Tuple[torch.tensor, torch.tensor]
    ) -> Tuple[torch.Tensor, Any]:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs
