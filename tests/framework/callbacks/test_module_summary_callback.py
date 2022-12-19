#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Tuple
from unittest.mock import MagicMock

import torch
from torchtnt.framework import AutoUnit
from torchtnt.framework._test_utils import DummyTrainUnit, generate_random_dataloader
from torchtnt.framework.callbacks.module_summary_callback import ModuleSummaryCallback
from torchtnt.framework.state import EntryPoint, PhaseState, State
from torchtnt.framework.train import init_train_state, train


class ModuleSummaryCallbackTest(unittest.TestCase):
    def test_module_summary_callback_max_depth(self) -> None:
        """
        Test  ModuleSummaryCallback callback with train entry point
        """
        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 1

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = State(
            entry_point=EntryPoint.TRAIN,
            train_state=PhaseState(
                dataloader=dataloader,
                max_epochs=max_epochs,
            ),
        )

        my_unit = MagicMock(spec=DummyTrainUnit)
        module_summary_callback = ModuleSummaryCallback(max_depth=2)
        module_summary_callback.on_train_epoch_start(state, my_unit)
        self.assertEqual(module_summary_callback._max_depth, 2)

    def test_module_summary_callback_train(self) -> None:
        """
        Test  ModuleSummaryCallback callback in train
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
        my_optimizer = torch.optim.SGD(my_module.parameters(), lr=0.01)

        auto_unit = DummyAutoUnit(
            module=my_module,
            optimizer=my_optimizer,
        )

        input_dim = 2
        dataset_len = 10
        batch_size = 2
        max_epochs = 10
        module_summary_callback = ModuleSummaryCallback()
        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)
        state = init_train_state(dataloader=dataloader, max_epochs=max_epochs)
        train(state, auto_unit, callbacks=[module_summary_callback])

        self.assertEqual(len(module_summary_callback._module_summaries), 1)
        ms = module_summary_callback._module_summaries[0]
        self.assertEqual(ms.module_name, "module")
        self.assertEqual(ms.module_type, "Net")
        self.assertTrue("l1" in ms.submodule_summaries)
        self.assertTrue("b1" in ms.submodule_summaries)
        self.assertTrue("l2" in ms.submodule_summaries)


Batch = Tuple[torch.tensor, torch.tensor]


class DummyAutoUnit(AutoUnit[Batch]):
    def compute_loss(self, state: State, data: Batch) -> Tuple[torch.Tensor, Any]:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs
