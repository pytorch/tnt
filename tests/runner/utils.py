#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchtnt.runner.progress import Progress
from torchtnt.runner.state import EntryPoint, PhaseState, State
from torchtnt.runner.unit import EvalUnit, PredictUnit, TrainUnit

Batch = Tuple[torch.Tensor, torch.Tensor]


class DummyEvalUnit(EvalUnit):
    def __init__(self, input_dim: int):
        super().__init__()
        # initialize module & loss_fn
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def eval_step(
        self, state: State, batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = batch

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs


class DummyPredictUnit(PredictUnit):
    def __init__(self, input_dim: int):
        super().__init__()
        # initialize module
        self.module = nn.Linear(input_dim, 2)

    def predict_step(self, state: State, batch: Batch) -> torch.Tensor:
        inputs, targets = batch

        outputs = self.module(inputs)
        return outputs


class DummyTrainUnit(TrainUnit):
    def __init__(self, input_dim: int):
        super().__init__()
        # initialize module, loss_fn, & optimizer
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)

    def train_step(
        self, state: State, batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = batch

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss, outputs


def generate_random_dataset(num_samples: int, input_dim: int) -> Dataset[Batch]:
    """Returns a dataset of random inputs and labels for binary classification."""
    data = torch.randn(num_samples, input_dim)
    labels = torch.randint(low=0, high=2, size=(num_samples,))
    return TensorDataset(data, labels)


def generate_random_dataloader(
    num_samples: int, input_dim: int, batch_size: int
) -> DataLoader:
    return DataLoader(
        generate_random_dataset(num_samples, input_dim),
        batch_size=batch_size,
    )


def get_dummy_fit_state() -> State:
    return State(
        entry_point=EntryPoint.FIT,
        train_state=PhaseState(
            progress=Progress(),
            dataloader=[1, 2, 3, 4, 5],
            max_epochs=2,
            max_steps_per_epoch=10,
        ),
        eval_state=PhaseState(
            progress=Progress(),
            dataloader=[1, 2, 3],
            max_steps_per_epoch=5,
        ),
    )
