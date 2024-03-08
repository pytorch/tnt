#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import logging
import sys
import tempfile
from argparse import Namespace
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data.datapipes.map.combinatorics import ShufflerIterDataPipe

from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper
from torcheval.metrics import BinaryAccuracy
from torchtnt.framework.state import State
from torchtnt.framework.train import train
from torchtnt.framework.unit import TrainUnit
from torchtnt.utils import copy_data_to_device, init_from_env, seed, TLRScheduler

from torchtnt.utils.loggers import TensorBoardLogger

_logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Batch = Tuple[torch.Tensor, torch.Tensor]


def prepare_model(input_dim: int, device: torch.device) -> nn.Module:
    """
    Instantiate a nn.Module which will define the architecture of your model.
    See https://pytorch.org/docs/stable/generated/torch.nn.Module.html for docs.
    """
    return nn.Linear(input_dim, 1, device=device)


def _generate_datapipe(
    num_samples: int, input_dim: int
) -> ShufflerIterDataPipe[object]:
    """
    Returns a shuffled datapipe of random inputs and labels.
    """
    data = IterableWrapper(
        [torch.rand(input_dim, input_dim) for _ in range(num_samples)]
    )
    labels = IterableWrapper(
        [
            torch.rand(
                input_dim,
            )
            for _ in range(num_samples)
        ]
    )
    train_data_pipe = data.zip(labels).shuffle()
    return train_data_pipe


def prepare_dataloader(
    num_samples: int, input_dim: int, num_workers: int, device: torch.device
) -> DataLoader2:
    """
    Prepare the datapipe and instantiate the DataLoader2 for training.
    """
    # pin memory enables faster host to GPU copies
    # enable_pin_memory = device.type == "cuda"
    # To be replaced by pin_memory DataPipe https://github.com/pytorch/data/issues/1013

    train_data_pipe = _generate_datapipe(num_samples, input_dim)
    dataloader = DataLoader2(
        datapipe=train_data_pipe,
        reading_service=MultiProcessingReadingService(num_workers=num_workers),
    )
    return dataloader


class MyTrainUnit(TrainUnit[Batch]):
    def __init__(
        self,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: TLRScheduler,
        device: torch.device,
        train_accuracy: BinaryAccuracy,
        tb_logger: TensorBoardLogger,
        log_every_n_steps: int,
    ) -> None:
        super().__init__()
        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

        # create an accuracy Metric to compute the accuracy of training
        self.train_accuracy = train_accuracy
        self.log_every_n_steps = log_every_n_steps

        self.tb_logger = tb_logger

    def train_step(self, state: State, data: Batch) -> None:
        data = copy_data_to_device(data, self.device)
        inputs, targets = data

        # convert targets to float Tensor for binary_cross_entropy_with_logits
        targets = targets.float()

        outputs = self.module(inputs)
        outputs = torch.squeeze(outputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        # update metrics & logs
        self.train_accuracy.update(outputs, targets)
        step_count = self.train_progress.num_steps_completed
        if (step_count + 1) % self.log_every_n_steps == 0:
            accuracy = self.train_accuracy.compute()
            self.tb_logger.log("loss", loss, step_count)
            self.tb_logger.log("accuracy", accuracy, step_count)

    def on_train_epoch_end(self, state: State) -> None:
        # compute and log the metrics at the end of epoch
        step_count = self.train_progress.num_steps_completed
        accuracy = self.train_accuracy.compute()
        self.tb_logger.log("accuracy_epoch", accuracy, step_count)

        # reset the metric at the end of every epoch
        self.train_accuracy.reset()

        # step the learning rate scheduler
        self.lr_scheduler.step()


def main(argv: List[str]) -> None:
    args = get_args(argv)
    seed(args.seed)

    # device and process group initialization
    device = init_from_env()

    path = tempfile.mkdtemp()
    tb_logger = TensorBoardLogger(path)

    model = prepare_model(args.input_dim, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_accuracy = BinaryAccuracy(device=device)

    dataloader = prepare_dataloader(
        args.num_batches_per_epoch,
        args.input_dim,
        args.num_reading_service_workers,
        device,
    )

    train_unit = MyTrainUnit(
        module=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        train_accuracy=train_accuracy,
        tb_logger=tb_logger,
        log_every_n_steps=args.log_every_n_steps,
    )

    train(
        train_unit,
        train_dataloader=dataloader,
        max_epochs=args.max_epochs,
    )


def get_args(argv: List[str]) -> Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--input-dim", type=int, default=32, help="input dimension")
    parser.add_argument("--max-epochs", type=int, default=2, help="training epochs")
    parser.add_argument(
        "--num-batches-per-epoch",
        type=int,
        default=1024,
        help="number of batches per epoch",
    )
    parser.add_argument(
        "--num-reading-service-workers",
        type=int,
        default=2,
        help="num of reading service workers",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--log-every-n-steps", type=int, default=10, help="log every n steps"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
