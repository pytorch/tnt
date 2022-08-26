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
from torchtnt.runner.state import State
from torchtnt.runner.unit import EvalUnit, PredictUnit, TrainUnit

Batch = Tuple[torch.Tensor, torch.Tensor]


class DummyEvalUnit(EvalUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module & loss_fn
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def eval_step(self, state: State, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs


class DummyPredictUnit(PredictUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module
        self.module = nn.Linear(input_dim, 2)

    def predict_step(self, state: State, data: Batch) -> torch.Tensor:
        inputs, targets = data

        outputs = self.module(inputs)
        return outputs


class DummyTrainUnit(TrainUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module, loss_fn, & optimizer
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)

    def train_step(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss, outputs


class DummyFitUnit(TrainUnit[Batch], EvalUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module, loss_fn, & optimizer
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)

    def train_step(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss, outputs

    def eval_step(self, state: State, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
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
