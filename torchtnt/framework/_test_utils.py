#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Iterator, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset
from torchtnt.framework.auto_unit import AutoUnit
from torchtnt.framework.state import EntryPoint, PhaseState, State
from torchtnt.framework.unit import EvalUnit, PredictUnit, TrainUnit
from torchtnt.utils.lr_scheduler import TLRScheduler

Batch = Tuple[torch.Tensor, torch.Tensor]


def get_dummy_train_state(dataloader: Optional[Iterable[object]] = None) -> State:
    return State(
        entry_point=EntryPoint.TRAIN,
        train_state=PhaseState(
            dataloader=dataloader or [1, 2, 3, 4],
            max_epochs=1,
            max_steps=1,
            max_steps_per_epoch=1,
        ),
        timer=None,
    )


def get_dummy_fit_state() -> State:
    return State(
        entry_point=EntryPoint.FIT,
        train_state=PhaseState(
            dataloader=[1, 2, 3, 4],
            max_epochs=1,
            max_steps=1,
            max_steps_per_epoch=1,
        ),
        eval_state=PhaseState(
            dataloader=[1, 2, 3, 4],
            max_epochs=1,
            max_steps=1,
            max_steps_per_epoch=1,
        ),
        timer=None,
    )


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


class RandomIterableDataset(IterableDataset):
    def __init__(self, size: int, count: int) -> None:
        self.count: int = count
        self.size: int = size

    def __iter__(self) -> Iterator[Tensor]:
        for _ in range(self.count):
            yield torch.randn(self.size)


def generate_random_iterable_dataloader(
    num_samples: int, input_dim: int, batch_size: int
) -> DataLoader:
    return DataLoader(
        dataset=RandomIterableDataset(input_dim, num_samples),
        batch_size=batch_size,
    )


class DummyAutoUnit(AutoUnit[Batch]):
    def compute_loss(self, state: State, data: Batch) -> Tuple[torch.Tensor, object]:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        my_optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            my_optimizer, gamma=0.9
        )
        return my_optimizer, my_lr_scheduler
