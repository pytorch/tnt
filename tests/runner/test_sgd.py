#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.distributed import launcher as pet
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtnt.runner.sgd import SGDEngine
from torchtnt.tests.runner.utils import get_dummy_fit_state
from torchtnt.utils.env import init_from_env
from torchtnt.utils.test_utils import get_pet_launch_config


class SGDTest(unittest.TestCase):
    cuda_available = torch.cuda.is_available()

    def test_forward(self) -> None:
        """
        Test the forward method
        """
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def get_loss(state, batch):
            inputs, targets = batch
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            return loss, outputs

        engine = SGDEngine(model, optimizer, get_loss)
        inputs = torch.rand(1, 1)
        outputs = engine.forward(inputs)
        self.assertTrue(isinstance(outputs, torch.Tensor))

    def test_single_process_train_step(self) -> None:
        """
        Test single process train step
        """
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def get_loss(state, batch):
            inputs, targets = batch
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            return loss, outputs

        engine = SGDEngine(model, optimizer, get_loss)
        batch = (torch.rand(1, 1), torch.rand(1, 1))
        engine.train()

        loss, outputs = engine.step(
            state=get_dummy_fit_state(),
            batch=batch,
        )
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertTrue(isinstance(outputs, torch.Tensor))

    @staticmethod
    def _test_ddp_train_step() -> None:
        """
        Test ddp train step
        """
        tc = unittest.TestCase()

        torch.distributed.init_process_group("gloo")
        model = torch.nn.Linear(1, 1)
        model = DDP(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def get_loss(state, batch):
            inputs, targets = batch
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            return loss, outputs

        engine = SGDEngine(model, optimizer, get_loss)
        batch = (torch.rand(1, 1), torch.rand(1, 1))
        engine.train()

        loss, outputs = engine.step(
            state=get_dummy_fit_state(),
            batch=batch,
        )
        tc.assertTrue(isinstance(loss, torch.Tensor))
        tc.assertTrue(isinstance(outputs, torch.Tensor))

    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_ddp_train_step(self) -> None:
        """
        Launch ddp train step test
        """
        lc = get_pet_launch_config(2)
        pet.elastic_launch(lc, entrypoint=self._test_ddp_train_step)()

    @staticmethod
    def _test_fsdp_train_step() -> None:
        """
        Test fsdp train step
        """
        tc = unittest.TestCase()

        device = init_from_env()
        model = torch.nn.Linear(1, 1)
        model = model.to(device)
        model = FSDP(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def get_loss(state, batch):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            return loss, outputs

        engine = SGDEngine(model, optimizer, get_loss)
        batch = (torch.rand(1, 1), torch.rand(1, 1))
        engine.train()

        loss, outputs = engine.step(
            state=get_dummy_fit_state(),
            batch=batch,
        )
        tc.assertTrue(isinstance(loss, torch.Tensor))
        tc.assertTrue(isinstance(outputs, torch.Tensor))

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_fsdp_train_step(self) -> None:
        """
        Launch fsdp train step test
        """
        lc = get_pet_launch_config(4)
        pet.elastic_launch(lc, entrypoint=self._test_fsdp_train_step)()
