#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import unittest
from typing import cast, Dict, Iterator
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtnt.framework._test_utils import DummyAutoUnit, generate_random_dataset
from torchtnt.framework.state import State
from torchtnt.framework.utils import (
    _construct_tracked_optimizers_and_schedulers,
    _find_optimizers_for_module,
    _is_done,
    _is_epoch_done,
    _maybe_set_distributed_sampler_epoch,
    _reset_module_training_mode,
    _set_module_training_mode,
    _step_requires_iterator,
    get_timing_context,
)
from torchtnt.utils.env import init_from_env
from torchtnt.utils.lr_scheduler import TLRScheduler
from torchtnt.utils.prepare_module import FSDPOptimizerWrapper
from torchtnt.utils.progress import Progress
from torchtnt.utils.test_utils import spawn_multi_process
from torchtnt.utils.timer import Timer


class UtilsTest(unittest.TestCase):
    cuda_available: bool = torch.cuda.is_available()
    distributed_available: bool = torch.distributed.is_available()

    def test_maybe_set_distributed_sampler_epoch(self) -> None:
        mp_dict = spawn_multi_process(
            2,
            "gloo",
            self._test_maybe_set_distributed_sampler_epoch,
        )
        self.assertTrue(mp_dict[0])
        self.assertTrue(mp_dict[1])

    @staticmethod
    def _test_maybe_set_distributed_sampler_epoch() -> bool:
        """
        Test _maybe_set_distributed_sampler_epoch util function
        """
        random_dataset = generate_random_dataset(10, 3)
        dummy_dataloader_with_distributed_sampler = DataLoader(
            random_dataset, sampler=DistributedSampler(random_dataset)
        )

        _maybe_set_distributed_sampler_epoch(
            dummy_dataloader_with_distributed_sampler, 20
        )

        sampler = cast(
            DistributedSampler[object],
            dummy_dataloader_with_distributed_sampler.sampler,
        )
        return sampler.epoch == 20

    def test_set_module_training_mode(self) -> None:
        """
        Test _set_module_training_mode
        """
        module = nn.Linear(1, 1)
        loss_fn = nn.CrossEntropyLoss()

        tracked_modules: Dict[str, torch.nn.Module] = {
            "module": module,
            "loss_fn": loss_fn,
        }

        # set module training mode to False
        prior_module_train_states = _set_module_training_mode(tracked_modules, False)

        self.assertFalse(module.training)
        self.assertFalse(loss_fn.training)

        self.assertTrue(prior_module_train_states["module"])
        self.assertTrue(prior_module_train_states["loss_fn"])

        # set back to True
        prior_module_train_states = _set_module_training_mode(tracked_modules, True)

        self.assertTrue(module.training)
        self.assertTrue(loss_fn.training)

        self.assertFalse(prior_module_train_states["module"])
        self.assertFalse(prior_module_train_states["loss_fn"])

    def test_reset_module_training_mode(self) -> None:
        """
        Test _reset_module_training_mode
        """
        module = nn.Linear(1, 1)
        loss_fn = nn.CrossEntropyLoss()

        tracked_modules: Dict[str, torch.nn.Module] = {
            "module": module,
            "loss_fn": loss_fn,
        }

        # set module training mode to False
        prior_module_train_states = _set_module_training_mode(tracked_modules, False)

        self.assertFalse(module.training)
        self.assertFalse(loss_fn.training)

        # set back to True using reset
        _reset_module_training_mode(tracked_modules, prior_module_train_states)

        self.assertTrue(module.training)
        self.assertTrue(loss_fn.training)

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

    def test_is_done(self) -> None:
        p = Progress(
            num_epochs_completed=2,
            num_steps_completed=100,
            num_steps_completed_in_epoch=5,
        )

        self.assertTrue(_is_done(p, max_epochs=2, max_steps=200))
        self.assertTrue(_is_done(p, max_epochs=2, max_steps=None))
        self.assertTrue(_is_done(p, max_epochs=3, max_steps=100))
        self.assertTrue(_is_done(p, max_epochs=None, max_steps=100))

        self.assertFalse(_is_done(p, max_epochs=3, max_steps=200))
        self.assertFalse(_is_done(p, max_epochs=None, max_steps=200))
        self.assertFalse(_is_done(p, max_epochs=3, max_steps=None))
        self.assertFalse(_is_done(p, max_epochs=None, max_steps=None))

    def test_is_epoch_done(self) -> None:
        p = Progress(
            num_epochs_completed=2,
            num_steps_completed=100,
            num_steps_completed_in_epoch=5,
        )

        self.assertTrue(_is_epoch_done(p, max_steps_per_epoch=5, max_steps=200))
        self.assertTrue(_is_epoch_done(p, max_steps_per_epoch=5, max_steps=None))
        self.assertTrue(_is_epoch_done(p, max_steps_per_epoch=100, max_steps=100))
        self.assertTrue(_is_epoch_done(p, max_steps_per_epoch=None, max_steps=100))

        self.assertFalse(_is_epoch_done(p, max_steps_per_epoch=6, max_steps=200))
        self.assertFalse(_is_epoch_done(p, max_steps_per_epoch=None, max_steps=200))
        self.assertFalse(_is_epoch_done(p, max_steps_per_epoch=6, max_steps=None))
        self.assertFalse(_is_epoch_done(p, max_steps_per_epoch=None, max_steps=None))

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

    @unittest.skipUnless(
        condition=distributed_available, reason="Torch distributed is needed to run"
    )
    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_construct_tracked_optimizers_and_schedulers(self) -> None:
        spawn_multi_process(2, "nccl", self._construct_optimizers)

    @staticmethod
    def _construct_optimizers() -> None:
        device = init_from_env()
        module = torch.nn.Linear(10, 10)

        auto_unit = DummyAutoUnit(module=module, device=device, strategy="fsdp")
        auto_unit.module2 = torch.nn.Linear(10, 10).to(device)
        auto_unit.optim2 = torch.optim.Adam(auto_unit.module2.parameters())

        result = _construct_tracked_optimizers_and_schedulers(auto_unit)
        tc = unittest.TestCase()
        tc.assertTrue(isinstance(result["optimizer"], FSDPOptimizerWrapper))
        tc.assertTrue(isinstance(result["optim2"], torch.optim.Optimizer))
        tc.assertTrue(isinstance(result["lr_scheduler"], TLRScheduler))
