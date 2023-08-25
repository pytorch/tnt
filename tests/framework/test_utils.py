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
from torch import nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtnt.framework._test_utils import generate_random_dataset
from torchtnt.framework.auto_unit import AutoUnit
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
    # pyre-fixme[4]: Attribute must be annotated.
    cuda_available = torch.cuda.is_available()

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
        # pyre-fixme[6]: For 1st argument expected `Iterable[typing.Any]` but got
        #  `None`.
        _maybe_set_distributed_sampler_epoch(None, 10)

        random_dataset = generate_random_dataset(10, 3)
        dummy_dataloader_with_distributed_sampler = DataLoader(
            random_dataset, sampler=DistributedSampler(random_dataset)
        )

        _maybe_set_distributed_sampler_epoch(
            dummy_dataloader_with_distributed_sampler, 20
        )
        # pyre-fixme[16]: Item `Sampler` of `Union[Sampler[typing.Any],
        #  Iterable[typing.Any]]` has no attribute `epoch`.
        return dummy_dataloader_with_distributed_sampler.sampler.epoch == 20

    def test_set_module_training_mode(self) -> None:
        """
        Test _set_module_training_mode
        """
        module = nn.Linear(1, 1)
        loss_fn = nn.CrossEntropyLoss()

        tracked_modules = {"module": module, "loss_fn": loss_fn}

        # set module training mode to False
        # pyre-fixme[6]: For 1st argument expected `Dict[str, Module]` but got
        #  `Dict[str, Union[Linear, CrossEntropyLoss]]`.
        prior_module_train_states = _set_module_training_mode(tracked_modules, False)

        self.assertFalse(module.training)
        self.assertFalse(loss_fn.training)

        self.assertTrue(prior_module_train_states["module"])
        self.assertTrue(prior_module_train_states["loss_fn"])

        # set back to True
        # pyre-fixme[6]: For 1st argument expected `Dict[str, Module]` but got
        #  `Dict[str, Union[Linear, CrossEntropyLoss]]`.
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
        # pyre-fixme[6]: For 1st argument expected `Dict[str, Module]` but got
        #  `Dict[str, Union[Linear, CrossEntropyLoss]]`.
        prior_module_train_states = _set_module_training_mode(tracked_modules, False)

        self.assertFalse(module.training)
        self.assertFalse(loss_fn.training)

        # set back to True using reset
        # pyre-fixme[6]: For 1st argument expected `Dict[str, Module]` but got
        #  `Dict[str, Union[Linear, CrossEntropyLoss]]`.
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

        # pyre-fixme[6]: For 1st argument expected `(State, object) -> object` but
        #  got `BoundMethod[typing.Callable(Foo.bar)[[Named(self, Foo)], None], Foo]`.
        self.assertFalse(_step_requires_iterator(foo.bar))
        # pyre-fixme[6]: For 1st argument expected `(State, object) -> object` but
        #  got `BoundMethod[typing.Callable(Foo.baz)[[Named(self, Foo), Named(data,
        #  Iterator[int]), Named(b, int), Named(c, str)], int], Foo]`.
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
    # pyre-fixme[2]: Parameter must be annotated.
    def test_get_timing_context(self, mock_record_function) -> None:
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

        opts = {"optim1": optim1, "optim2": optim2}
        # pyre-fixme[6]: For 2nd argument expected `Dict[str, Optimizer]` but got
        #  `Dict[str, Union[Adagrad, Adam]]`.
        optimizers = _find_optimizers_for_module(module1, opts)
        optim_name, _ = optimizers[0]
        self.assertEqual(optim_name, "optim1")
        # pyre-fixme[6]: For 2nd argument expected `Dict[str, Optimizer]` but got
        #  `Dict[str, Union[Adagrad, Adam]]`.
        optimizers = _find_optimizers_for_module(module2, opts)
        optim_name, _ = optimizers[0]
        self.assertEqual(optim_name, "optim2")

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
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

        opts = {"optim1": optim1, "optim2": optim2}
        # pyre-fixme[6]: For 2nd argument expected `Dict[str, Optimizer]` but got
        #  `Dict[str, Union[Adagrad, Adam]]`.
        optim_list = _find_optimizers_for_module(module1, opts)
        optim_name, _ = optim_list[0]

        tc = unittest.TestCase()
        tc.assertEqual(optim_name, "optim1")
        # pyre-fixme[6]: For 2nd argument expected `Dict[str, Optimizer]` but got
        #  `Dict[str, Union[Adagrad, Adam]]`.
        optim_list = _find_optimizers_for_module(module2, opts)
        optim_name, _ = optim_list[0]
        tc.assertEqual(optim_name, "optim2")

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.distributed.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
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

        result = _construct_tracked_optimizers_and_schedulers(auto_unit)
        tc = unittest.TestCase()
        tc.assertTrue(isinstance(result["optim"], FSDPOptimizerWrapper))
        tc.assertTrue(isinstance(result["optim2"], torch.optim.Optimizer))
        tc.assertTrue(isinstance(result["lr_scheduler"], TLRScheduler))


# pyre-fixme[5]: Global expression must be annotated.
Batch = Tuple[torch.tensor, torch.tensor]


# pyre-fixme[11]: Annotation `Batch` is not defined as a type.
class DummyAutoUnit(AutoUnit[Batch]):
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # pyre-fixme[4]: Attribute must be annotated.
        self.module2 = torch.nn.Linear(10, 10).to(self.device)
        self.optim = torch.optim.SGD(self.module.parameters(), lr=0.01)
        self.optim2 = torch.optim.Adam(self.module2.parameters())

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
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
