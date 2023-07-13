#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import unittest
from typing import Any, Iterator, Tuple
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torch import nn

from torchtnt.utils.version import is_torch_version_geq_2_0

if is_torch_version_geq_2_0():
    from torch.distributed._composable import fully_shard


from torch.distributed import launcher
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtnt.framework._test_utils import generate_random_dataset
from torchtnt.framework.auto_unit import AutoUnit
from torchtnt.framework.state import ActivePhase, EntryPoint, PhaseState, State
from torchtnt.framework.utils import (
    _construct_tracked_optimizers_and_schedulers,
    _find_optimizers_for_module,
    _FSDPOptimizerWrapper,
    _get_timing_context,
    _is_done,
    _is_epoch_done,
    _is_fsdp_module,
    _maybe_set_distributed_sampler_epoch,
    _reset_module_training_mode,
    _set_module_training_mode,
    _step_requires_iterator,
    get_current_progress,
)
from torchtnt.utils.env import init_from_env
from torchtnt.utils.lr_scheduler import TLRScheduler
from torchtnt.utils.progress import Progress
from torchtnt.utils.test_utils import get_pet_launch_config
from torchtnt.utils.timer import Timer


class UtilsTest(unittest.TestCase):
    cuda_available = torch.cuda.is_available()

    @staticmethod
    def _test_is_fsdp_module() -> None:
        dist.init_process_group("gloo")
        model = nn.Linear(1, 1)
        assert not _is_fsdp_module(model)
        model = FSDP(nn.Linear(1, 1))
        assert _is_fsdp_module(model)
        if is_torch_version_geq_2_0():
            fully_shard(model)
            assert _is_fsdp_module(model)

    @unittest.skipUnless(
        dist.is_available(), reason="Torch distributed is needed to run"
    )
    @unittest.skipUnless(
        condition=torch.cuda.is_available() and torch.cuda.device_count() > 2,
        reason="This test needs 2 GPUs to run.",
    )
    def test_is_fsdp_module(self) -> None:
        config = get_pet_launch_config(2)
        dist.launcher.elastic_launch(config, entrypoint=self._test_is_fsdp_module)()

    def test_maybe_set_distributed_sampler_epoch(self) -> None:
        config = get_pet_launch_config(3)
        result = dist.launcher.elastic_launch(
            config, entrypoint=self._test_maybe_set_distributed_sampler_epoch
        )()
        self.assertEqual(result[0], True)
        self.assertEqual(result[1], True)

    @staticmethod
    def _test_maybe_set_distributed_sampler_epoch() -> bool:
        """
        Test _maybe_set_distributed_sampler_epoch util function
        """
        dist.init_process_group("gloo")
        _maybe_set_distributed_sampler_epoch(None, 10)

        random_dataset = generate_random_dataset(10, 3)
        dummy_dataloader_with_distributed_sampler = DataLoader(
            random_dataset, sampler=DistributedSampler(random_dataset)
        )

        _maybe_set_distributed_sampler_epoch(
            dummy_dataloader_with_distributed_sampler, 20
        )
        return dummy_dataloader_with_distributed_sampler.sampler.epoch == 20

    def test_set_module_training_mode(self) -> None:
        """
        Test _set_module_training_mode
        """
        module = nn.Linear(1, 1)
        loss_fn = nn.CrossEntropyLoss()

        tracked_modules = {"module": module, "loss_fn": loss_fn}

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

        tracked_modules = {"module": module, "loss_fn": loss_fn}

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
            def bar(self) -> None:
                pass

            def baz(self, data: Iterator[int], b: int, c: str) -> int:
                return b

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

    def test_get_current_progress(self) -> None:
        train_state = PhaseState(
            dataloader=[], progress=Progress(num_steps_completed=0)
        )
        eval_state = PhaseState(dataloader=[], progress=Progress(num_steps_completed=1))
        predict_state = PhaseState(
            dataloader=[], progress=Progress(num_steps_completed=2)
        )
        state = State(
            entry_point=EntryPoint.TRAIN,
            train_state=train_state,
            eval_state=eval_state,
            predict_state=predict_state,
        )

        progress = get_current_progress(state)
        self.assertEqual(
            progress.num_steps_completed, train_state.progress.num_steps_completed
        )

        state._active_phase = ActivePhase.EVALUATE
        progress = get_current_progress(state)
        self.assertEqual(
            progress.num_steps_completed, eval_state.progress.num_steps_completed
        )

        state._active_phase = ActivePhase.PREDICT
        progress = get_current_progress(state)
        self.assertEqual(
            progress.num_steps_completed, predict_state.progress.num_steps_completed
        )

        state._entry_point = EntryPoint.FIT
        state._active_phase = ActivePhase.EVALUATE
        progress = get_current_progress(state)
        self.assertEqual(
            progress.num_steps_completed, train_state.progress.num_steps_completed
        )

    @patch("torchtnt.framework.utils.record_function")
    def test_get_timing_context(self, mock_record_function) -> None:
        state = MagicMock()
        state.timer = None

        ctx = _get_timing_context(state, "a")
        with ctx:
            time.sleep(1)
        mock_record_function.assert_called_with("a")

        state.timer = Timer()
        ctx = _get_timing_context(state, "b")
        with ctx:
            time.sleep(1)
        self.assertTrue("b" in state.timer.recorded_durations.keys())
        mock_record_function.assert_called_with("b")

        state.timer = Timer()
        ctx = _get_timing_context(state, "c", skip_timer=True)
        with ctx:
            time.sleep(1)
        # "c" should not be in the recorded_durations because we set skip_timer to True
        self.assertFalse("c" in state.timer.recorded_durations.keys())
        mock_record_function.assert_called_with("c")

    def test_find_optimizers_for_module(self) -> None:
        module1 = torch.nn.Linear(10, 10)
        module2 = torch.nn.Linear(10, 10)
        optim1 = torch.optim.Adam(module1.parameters())
        optim2 = torch.optim.Adagrad(module2.parameters())

        opts = {"optim1": optim1, "optim2": optim2}
        optimizers = _find_optimizers_for_module(module1, opts)
        optim_name, _ = optimizers[0]
        self.assertEqual(optim_name, "optim1")
        optimizers = _find_optimizers_for_module(module2, opts)
        optim_name, _ = optimizers[0]
        self.assertEqual(optim_name, "optim2")

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_find_optimizers_for_FSDP_module(self) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(
            config, entrypoint=self._find_optimizers_for_FSDP_module
        )()

    @staticmethod
    def _find_optimizers_for_FSDP_module() -> None:
        device = init_from_env()
        module1 = FSDP(torch.nn.Linear(10, 10).to(device))
        module2 = torch.nn.Linear(10, 10)
        optim1 = torch.optim.Adam(module1.parameters())
        optim2 = torch.optim.Adagrad(module2.parameters())

        opts = {"optim1": optim1, "optim2": optim2}
        optim_list = _find_optimizers_for_module(module1, opts)
        optim_name, _ = optim_list[0]

        tc = unittest.TestCase()
        tc.assertEqual(optim_name, "optim1")
        optim_list = _find_optimizers_for_module(module2, opts)
        optim_name, _ = optim_list[0]
        tc.assertEqual(optim_name, "optim2")

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_construct_tracked_optimizers_and_schedulers(self) -> None:
        config = get_pet_launch_config(2)
        launcher.elastic_launch(config, entrypoint=self._construct_optimizers)()

    @staticmethod
    def _construct_optimizers() -> None:
        device = init_from_env()
        module = torch.nn.Linear(10, 10)

        auto_unit = DummyAutoUnit(module=module, device=device, strategy="fsdp")

        result = _construct_tracked_optimizers_and_schedulers(auto_unit)
        tc = unittest.TestCase()
        tc.assertTrue(isinstance(result["optim"], _FSDPOptimizerWrapper))
        tc.assertTrue(isinstance(result["optim2"], torch.optim.Optimizer))
        tc.assertTrue(isinstance(result["lr_scheduler"], TLRScheduler))


Batch = Tuple[torch.tensor, torch.tensor]


class DummyAutoUnit(AutoUnit[Batch]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module2 = torch.nn.Linear(10, 10).to(self.device)
        self.optim = torch.optim.SGD(self.module.parameters(), lr=0.01)
        self.optim2 = torch.optim.Adam(self.module2.parameters())

    def compute_loss(self, state: State, data: Batch) -> Tuple[torch.Tensor, Any]:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        my_optimizer = self.optim
        my_lr_scheduler = ExponentialLR(my_optimizer, gamma=0.9)
        return my_optimizer, my_lr_scheduler
