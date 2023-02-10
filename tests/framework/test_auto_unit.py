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

from torchtnt.utils.version import is_torch_version_geq_1_13

DYNAMO_AVAIL = False
if is_torch_version_geq_1_13():
    DYNAMO_AVAIL = True
    import torch._dynamo

from parameterized import parameterized
from torch.distributed import launcher
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtnt.framework._test_utils import (
    generate_random_dataloader,
    generate_random_iterable_dataloader,
)

from torchtnt.framework.auto_unit import (
    AutoUnit,
    DDPStrategy,
    SWAParams,
    TorchDynamoParams,
)
from torchtnt.framework.evaluate import evaluate, init_eval_state
from torchtnt.framework.predict import init_predict_state, predict
from torchtnt.framework.state import State
from torchtnt.framework.train import init_train_state, train
from torchtnt.utils import init_from_env, TLRScheduler
from torchtnt.utils.device import copy_data_to_device
from torchtnt.utils.test_utils import get_pet_launch_config


class TestAutoUnit(unittest.TestCase):
    cuda_available = torch.cuda.is_available()

    def test_app_state_mixin(self) -> None:
        """
        Test that app_state, tracked_optimizers, tracked_lr_schedulers are set as expected with AutoUnit
        """
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyAutoUnit(
            module=my_module,
            precision="fp16",
        )

        state = init_train_state(dataloader=MagicMock(), max_epochs=1)
        # call on_train_start to configure optimizers
        auto_unit.on_train_start(state)

        self.assertEqual(auto_unit.tracked_modules()["module"], my_module)
        self.assertTrue(
            isinstance(
                auto_unit.tracked_misc_statefuls()["grad_scaler"],
                torch.cuda.amp.GradScaler,
            )
        )
        for key in ("module", "optimizer", "lr_scheduler", "grad_scaler"):
            self.assertTrue(key in auto_unit.app_state())

    def test_lr_scheduler_step(self) -> None:
        """
        Test that the lr scheduler is stepped every optimizer step when step_lr_interval="step"
        """
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyAutoUnit(
            module=my_module,
            step_lr_interval="step",
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3
        expected_steps_per_epoch = dataset_len / batch_size * max_epochs

        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=train_dl, max_epochs=max_epochs)
        train(state, auto_unit)
        self.assertEqual(
            auto_unit.lr_scheduler.step.call_count, expected_steps_per_epoch
        )

    def test_lr_scheduler_epoch(self) -> None:
        """
        Test that the lr scheduler is stepped every epoch when step_lr_interval="epoch"
        """
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyAutoUnit(
            module=my_module,
            step_lr_interval="epoch",
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2
        max_epochs = 3

        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)

        state = init_train_state(dataloader=train_dl, max_epochs=max_epochs)
        train(state, auto_unit)
        self.assertEqual(auto_unit.lr_scheduler.step.call_count, max_epochs)

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    @patch("torch.autocast")
    def test_mixed_precision_fp16(self, mock_autocast) -> None:
        """
        Test that the mixed precision autocast context is called when fp16 precision is set
        """
        my_module = torch.nn.Linear(2, 2)
        auto_unit = DummyAutoUnit(
            module=my_module,
            precision="fp16",
        )
        state = init_train_state(dataloader=MagicMock(), max_epochs=1)
        # call on_train_start to configure optimizers
        auto_unit.on_train_start(state)

        dummy_data = (torch.ones(2, 2), torch.ones(2, 2))
        auto_unit.train_step(state=MagicMock(), data=dummy_data)
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
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyAutoUnit(
            module=my_module,
            precision="bf16",
        )
        state = init_train_state(dataloader=MagicMock(), max_epochs=1)
        # call on_train_start to configure optimizers
        auto_unit.on_train_start(state)

        dummy_data = (torch.ones(2, 2), torch.ones(2, 2))
        auto_unit.train_step(state=MagicMock(), data=dummy_data)
        mock_autocast.assert_called_with(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        )

    def test_mixed_precision_invalid_str(self) -> None:
        """
        Test that an exception is raised with an invalid precision string
        """
        my_module = torch.nn.Linear(2, 2)

        with self.assertRaisesRegex(ValueError, "Precision f16 not supported"):
            _ = DummyAutoUnit(
                module=my_module,
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
        Test the num_optimizer_steps_completed property of AutoUnit
        """
        my_module = torch.nn.Linear(2, 2)

        input_dim = 2
        dataset_len = 16
        batch_size = 2
        max_epochs = 1

        auto_unit = DummyAutoUnit(
            module=my_module,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        expected_opt_steps_per_epoch = math.ceil(
            dataset_len / batch_size / gradient_accumulation_steps
        )

        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=train_dl, max_epochs=max_epochs)
        train(state, auto_unit)
        self.assertTrue(
            auto_unit.num_optimizer_steps_completed, expected_opt_steps_per_epoch
        )

    def test_stochastic_weight_averaging_basic(self) -> None:
        """
        Basic stochastic weight averaging tests
        """
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyAutoUnit(
            module=my_module,
        )

        auto_unit2 = DummyAutoUnit(
            module=my_module,
            swa_params=SWAParams(epoch_start=1, anneal_epochs=5),
        )

        state = init_train_state(dataloader=MagicMock(), max_epochs=1)
        # call on_train_start to configure optimizers
        auto_unit.on_train_start(state)
        auto_unit2.on_train_start(state)

        self.assertIsNone(auto_unit.swa_model)
        self.assertIsNotNone(auto_unit2.swa_model)

        self.assertTrue("swa_model" not in auto_unit.app_state())
        self.assertTrue("swa_model" not in auto_unit.tracked_modules())
        self.assertTrue("swa_scheduler" not in auto_unit.app_state())
        self.assertTrue("swa_scheduler" not in auto_unit.tracked_lr_schedulers())
        self.assertTrue("swa_model" in auto_unit2.app_state())
        self.assertTrue("swa_model" in auto_unit2.tracked_modules())
        self.assertTrue("swa_scheduler" in auto_unit2.app_state())
        self.assertTrue("swa_scheduler" in auto_unit2.tracked_lr_schedulers())

    def test_stochastic_weight_averaging_e2e(self) -> None:
        """
        e2e stochastic weight averaging test
        """

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = torch.nn.Linear(2, 2)
                self.b1 = torch.nn.BatchNorm1d(2)
                self.l2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.l1(x)
                x = self.b1(x)
                x = self.l2(x)
                return x

        my_module = Net()
        my_swa_params = SWAParams(epoch_start=1, anneal_epochs=5)

        auto_unit = DummyAutoUnit(
            module=my_module,
            swa_params=my_swa_params,
        )

        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 10
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=dataloader, max_epochs=max_epochs)
        train(state, auto_unit)

        orig_module = auto_unit.module
        swa_module = auto_unit.swa_model

        self.assertTrue(
            torch.allclose(
                orig_module.b1.running_mean, swa_module.module.b1.running_mean
            )
        )
        self.assertTrue(
            torch.allclose(orig_module.b1.running_var, swa_module.module.b1.running_var)
        )
        self.assertTrue(
            torch.allclose(orig_module.l1.weight, swa_module.module.l1.weight)
        )
        self.assertTrue(
            torch.allclose(orig_module.l2.weight, swa_module.module.l2.weight)
        )

    @unittest.skipUnless(
        condition=DYNAMO_AVAIL, reason="This test needs PyTorch 1.13 or greater to run."
    )
    def test_dynamo_eager(self) -> None:
        """
        e2e torchdynamo test
        """

        my_module = torch.nn.Linear(2, 2)

        input_dim = 2
        dataset_len = 16
        batch_size = 2
        max_epochs = 1

        auto_unit = DummyAutoUnit(
            module=my_module,
            torchdynamo_params=TorchDynamoParams("eager"),
        )

        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=train_dl, max_epochs=max_epochs)
        self.assertFalse(auto_unit._dynamo_used)
        train(state, auto_unit)
        self.assertTrue(auto_unit._dynamo_used)

    @unittest.skipUnless(
        condition=DYNAMO_AVAIL, reason="This test needs PyTorch 1.13 or greater to run."
    )
    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_dynamo_train(self) -> None:
        """
        e2e torchdynamo on train
        """

        my_module = torch.nn.Linear(2, 2)

        input_dim = 2
        dataset_len = 16
        batch_size = 2
        max_epochs = 1

        auto_unit = DummyAutoUnit(
            module=my_module,
            torchdynamo_params=TorchDynamoParams("inductor"),
        )

        train_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=train_dl, max_epochs=max_epochs)

        self.assertFalse(auto_unit._dynamo_used)
        train(state, auto_unit)
        self.assertTrue(auto_unit._dynamo_used)

    @unittest.skipUnless(
        condition=DYNAMO_AVAIL, reason="This test needs PyTorch 1.13 or greater to run."
    )
    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_dynamo_eval(self) -> None:
        """
        e2e torchdynamo on eval
        """

        my_module = torch.nn.Linear(2, 2)

        input_dim = 2
        dataset_len = 16
        batch_size = 2

        auto_unit = DummyAutoUnit(
            module=my_module,
            torchdynamo_params=TorchDynamoParams("inductor"),
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2

        eval_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_eval_state(dataloader=eval_dl)
        self.assertFalse(auto_unit._dynamo_used)
        evaluate(state, auto_unit)
        self.assertTrue(auto_unit._dynamo_used)

    @unittest.skipUnless(
        condition=DYNAMO_AVAIL, reason="This test needs PyTorch 1.13 or greater to run."
    )
    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_dynamo_predict(self) -> None:
        """
        e2e torchdynamo on predict
        """
        my_module = torch.nn.Linear(2, 2)
        auto_unit = DummyAutoUnit(
            module=my_module,
            torchdynamo_params=TorchDynamoParams("inductor"),
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2

        predict_dl = generate_random_iterable_dataloader(
            dataset_len, input_dim, batch_size
        )
        state = init_predict_state(dataloader=predict_dl)
        self.assertFalse(auto_unit._dynamo_used)
        predict(state, auto_unit)

    @unittest.skipUnless(
        condition=DYNAMO_AVAIL, reason="This test needs PyTorch 1.13 or greater to run."
    )
    def test_dynamo_invalid_backend(self) -> None:
        """
        verify error is thrown on invalid backend
        """

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = torch.nn.Linear(2, 2)
                self.b1 = torch.nn.BatchNorm1d(2)
                self.l2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.l1(x)
                x = self.b1(x)
                x = self.l2(x)
                return x

        my_module = Net()
        my_dynamo_params = TorchDynamoParams(backend="foo")

        self.failUnlessRaises(
            RuntimeError,
            DummyAutoUnit,
            **{
                "module": my_module,
                "torchdynamo_params": my_dynamo_params,
            },
        )

    def test_log_frequency_steps_exception(self) -> None:
        """
        Test that an exception is raised when log_frequency_steps is < 1
        """
        my_module = torch.nn.Linear(2, 2)

        with self.assertRaisesRegex(
            ValueError, "log_frequency_steps must be > 0. Got 0"
        ):
            _ = DummyAutoUnit(
                module=my_module,
                log_frequency_steps=0,
            )

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    @patch("torch.autocast")
    def test_eval_mixed_precision_bf16(self, mock_autocast) -> None:
        """
        Test that the mixed precision autocast context is called during evaluate when precision = bf16
        """
        my_module = torch.nn.Linear(2, 2)
        auto_unit = DummyAutoUnit(
            module=my_module,
            precision="bf16",
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2

        eval_dl = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_eval_state(dataloader=eval_dl)
        evaluate(state, auto_unit)
        mock_autocast.assert_called_with(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        )

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    @patch("torch.autocast")
    def test_predict_mixed_precision_bf16(self, mock_autocast) -> None:
        """
        Test that the mixed precision autocast context is called during predict when precision = fp16
        """
        my_module = torch.nn.Linear(2, 2)
        auto_unit = DummyAutoUnit(
            module=my_module,
            precision="fp16",
        )

        input_dim = 2
        dataset_len = 8
        batch_size = 2

        predict_dl = generate_random_iterable_dataloader(
            dataset_len, input_dim, batch_size
        )
        state = init_predict_state(dataloader=predict_dl)
        predict(state, auto_unit)
        mock_autocast.assert_called_with(
            device_type="cuda", dtype=torch.float16, enabled=True
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

        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyAutoUnit(
            module=my_module,
            strategy=DDPStrategy(),
            gradient_accumulation_steps=2,
        )

        dummy_data = (torch.ones(2, 2), torch.ones(2, 2))
        state = init_train_state(dataloader=MagicMock(), max_epochs=1)
        # call on_train_start to configure optimizers
        auto_unit.on_train_start(state)

        # for the first step no_sync should be called since we accumulate gradients
        with patch.object(auto_unit.module, "no_sync") as no_sync_mock:
            auto_unit.train_step(state=state, data=dummy_data)
            no_sync_mock.assert_called()

        state.train_state.progress.increment_step()
        # for the second step no_sync should not be called since we run optimizer step
        with patch.object(auto_unit.module, "no_sync") as no_sync_mock:
            auto_unit.train_step(state=state, data=dummy_data)
            no_sync_mock.assert_not_called()

    @staticmethod
    def _test_fsdp_no_sync() -> None:
        """
        Test that the no_sync autocast context is correctly applied when using gradient accumulation and FSDP
        """
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2).to(device)
        my_module = FSDP(my_module)

        auto_unit = DummyAutoUnit(
            module=my_module,
            device=device,
            gradient_accumulation_steps=2,
        )

        dummy_data = (torch.ones(2, 2), torch.ones(2, 2))
        state = init_train_state(dataloader=MagicMock(), max_epochs=1)
        # call on_train_start to configure optimizers
        auto_unit.on_train_start(state)

        # for the first step no_sync should be called since we accumulate gradients
        with patch.object(auto_unit.module, "no_sync") as no_sync_mock:
            auto_unit.train_step(state=state, data=dummy_data)
            no_sync_mock.assert_called()

        state.train_state.progress.increment_step()
        # for the second step no_sync should not be called since we run optimizer step
        with patch.object(auto_unit.module, "no_sync") as no_sync_mock:
            auto_unit.train_step(state=state, data=dummy_data)
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

        tc = unittest.TestCase()
        with patch(
            "torchtnt.framework.auto_unit.is_torch_version_geq_1_12", return_value=False
        ), tc.assertRaisesRegex(
            RuntimeError,
            "Using float16 precision with torch.distributed.fsdp.FullyShardedDataParallel requires torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler from PyTorch 1.12.",
        ):
            _ = DummyAutoUnit(
                module=my_module,
                device=device,
                precision="fp16",
            )

    def test_move_data_to_device(self) -> None:
        """
        Test that move_data_to_device is called
        """
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyAutoUnit(
            module=my_module,
            device=device,
        )

        state = init_train_state(dataloader=MagicMock(), max_epochs=1)
        # call on_train_start to configure optimizers
        auto_unit.on_train_start(state)

        dummy_data = (torch.ones(2, 2), torch.ones(2, 2))

        with patch.object(
            DummyAutoUnit, "move_data_to_device"
        ) as move_data_to_device_mock:
            dummy_data = copy_data_to_device(dummy_data, device)
            move_data_to_device_mock.return_value = dummy_data
            auto_unit.train_step(state=state, data=dummy_data)
            move_data_to_device_mock.assert_called_once()

    def test_configure_optimizers_and_lr_scheduler(self) -> None:
        """
        Test configure_optimizers_and_lr_scheduler with a complex AutoUnit where configure_optimizers_and_lr_scheduler uses an attribute set in the init
        """
        my_module = torch.nn.Linear(2, 2)

        auto_unit = DummyComplexAutoUnit(
            lr=0.01,
            module=my_module,
        )

        self.assertFalse(hasattr(auto_unit, "optimizer"))
        state = init_train_state(dataloader=MagicMock(), max_epochs=1)
        # call on_train_start to configure optimizers
        auto_unit.on_train_start(state)
        # assert that the optimizer attribute was correctly initialized and set
        self.assertTrue(hasattr(auto_unit, "optimizer"))

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_auto_unit_ddp(self) -> None:
        """
        Launch tests of DDP strategy
        """
        config = get_pet_launch_config(2)
        launcher.elastic_launch(config, entrypoint=self._test_ddp_wrap)()
        launcher.elastic_launch(
            config, entrypoint=self._test_stochastic_weight_averaging_with_ddp
        )()

    @staticmethod
    def _test_ddp_wrap() -> None:
        """
        Test that the module is correctly wrapped in DDP
        """

        my_module = torch.nn.Linear(2, 2)

        auto_ddp_unit = DummyAutoUnit(
            module=my_module,
            strategy=DDPStrategy(),
            gradient_accumulation_steps=2,
        )
        tc = unittest.TestCase()

        tc.assertTrue(isinstance(auto_ddp_unit.module, DDP))

    @staticmethod
    def _test_stochastic_weight_averaging_with_ddp() -> None:
        """
        e2e stochastic weight averaging test with ddp
        """

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = torch.nn.Linear(2, 2)
                self.b1 = torch.nn.BatchNorm1d(2)
                self.l2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.l1(x)
                x = self.b1(x)
                x = self.l2(x)
                return x

        my_module = Net()
        my_swa_params = SWAParams(epoch_start=1, anneal_epochs=5)

        auto_unit = DummyAutoUnit(
            module=my_module,
            strategy=DDPStrategy(),
            swa_params=my_swa_params,
        )

        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 10
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=dataloader, max_epochs=max_epochs)
        train(state, auto_unit)

        orig_module = auto_unit.module.module
        swa_module = auto_unit.swa_model.module.module

        tc = unittest.TestCase()
        tc.assertTrue(
            torch.allclose(
                orig_module.b1.running_mean,
                swa_module.b1.running_mean,
            )
        )
        tc.assertTrue(
            torch.allclose(
                orig_module.b1.running_var,
                swa_module.b1.running_var,
            )
        )
        tc.assertTrue(torch.allclose(orig_module.l1.weight, swa_module.l1.weight))
        tc.assertTrue(torch.allclose(orig_module.l2.weight, swa_module.l2.weight))


Batch = Tuple[torch.tensor, torch.tensor]


class DummyAutoUnit(AutoUnit[Batch]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dynamo_used = False

    def compute_loss(self, state: State, data: Batch) -> Tuple[torch.Tensor, Any]:
        if DYNAMO_AVAIL:
            self._dynamo_used = torch._dynamo.is_compiling()
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        my_optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        my_lr_scheduler = MagicMock()
        return my_optimizer, my_lr_scheduler


class DummyComplexAutoUnit(AutoUnit[Batch]):
    def __init__(self, lr: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def compute_loss(self, state: State, data: Batch) -> Tuple[torch.Tensor, Any]:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        my_optimizer = torch.optim.SGD(module.parameters(), lr=self.lr)
        my_lr_scheduler = MagicMock()
        return my_optimizer, my_lr_scheduler
