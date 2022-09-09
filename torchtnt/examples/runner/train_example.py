#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys
import tempfile
from argparse import Namespace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataset import Dataset, TensorDataset
from torcheval.metrics import BinaryAccuracy
from torchtnt.data import CudaDataPrefetcher
from torchtnt.loggers.tensorboard import TensorBoardLogger
from torchtnt.runner.state import State
from torchtnt.runner.train import train
from torchtnt.runner.unit import TrainUnit
from torchtnt.utils import init_from_env, seed, Timer

_logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Batch = Tuple[torch.Tensor, torch.Tensor]


def prepare_module(
    input_dim: int,
    device: torch.device,
    strategy: Optional[str],
    precision: Optional[torch.dtype],
) -> nn.Module:
    """
    Instantiate a nn.Module which will define the architecture of your model. If using a data parallel technique, wrap the module in DDP or FSDP.
    See https://pytorch.org/docs/stable/generated/torch.nn.Module.html for docs.
    """
    module = nn.Linear(input_dim, 1)
    # move module to device
    module = module.to(device)

    if strategy == "ddp":
        # wrap module in DDP
        device_ids = None
        if device.type == "cuda":
            device_ids = [device.index]
        return DDP(module, device_ids=device_ids)
    elif strategy == "fsdp":
        # wrap module in FSDP
        return FSDP(
            module,
            cpu_offload=CPUOffload(offload_params=True),
            mixed_precision=MixedPrecision(
                param_dtype=precision,
                reduce_dtype=precision,
                buffer_dtype=precision,
            ),
        )
    else:
        return module


def _generate_dataset(num_samples: int, input_dim: int) -> Dataset[Batch]:
    """Returns a dataset of random inputs and labels for binary classification."""
    # TODO: use datapipes/dataloaderV2
    data = torch.randn(num_samples, input_dim)
    labels = torch.randint(low=0, high=2, size=(num_samples,))
    return TensorDataset(data, labels)


def prepare_dataloader(
    num_samples: int, input_dim: int, batch_size: int, device: torch.device
) -> Union[CudaDataPrefetcher, torch.utils.data.DataLoader]:
    """Instantiate DataLoader and CudaDataPrefetcher"""
    # pin_memory enables faster host to GPU copies
    on_cuda = device.type == "cuda"
    dataloader = torch.utils.data.DataLoader(
        _generate_dataset(num_samples, input_dim),
        batch_size=batch_size,
        pin_memory=on_cuda,
    )
    if on_cuda:
        dataloader = CudaDataPrefetcher(dataloader, device)
    return dataloader


class MyTrainUnit(TrainUnit[Batch]):
    def __init__(
        self,
        input_dim: int,
        device: torch.device,
        strategy: str,
        precision: Optional[torch.dtype],
        lr: float,
        log_frequency_steps: int,
    ):
        super().__init__()
        # initialize module & optimizer
        self.module = prepare_module(input_dim, device, strategy, precision)
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9
        )

        # create an accuracy Metric to compute the accuracy of training
        self.train_accuracy = BinaryAccuracy(device=device)
        self.log_frequency_steps = log_frequency_steps

        path = tempfile.mkdtemp()
        self.tb_logger = TensorBoardLogger(path)

    def train_step(self, state: State, data: Batch) -> None:
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
        progress = state.train_state.progress
        if (progress.num_steps_completed + 1) % self.log_frequency_steps == 0:
            acc = self.train_accuracy.compute()
            self.tb_logger.log("loss", loss, progress.num_steps_completed)
            self.tb_logger.log("accuracy", acc, progress.num_steps_completed)

    def on_train_epoch_end(self, state: State) -> None:
        # reset the metric every epoch
        self.train_accuracy.reset()

        # step the learning rate scheduler
        self.lr_scheduler.step()

    def on_train_end(self, state: State) -> None:
        self.tb_logger.close()


def main(argv: List[str]) -> None:
    # parse command line arguments
    args = get_args(argv)

    precision = None
    if args.precision == "fp16":
        precision = torch.float16
    elif args.precision == "bf16":
        precision = torch.bfloat16

    # seed the RNG for better reproducibility. see docs https://pytorch.org/docs/stable/notes/randomness.html
    seed(args.seed)

    # device and process group initialization
    device = init_from_env()

    _logger.info(
        f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', '0')}\n"
        f"RANK: {os.environ.get('RANK', '0')}\n"
        f"GROUP_RANK: {os.environ.get('GROUP_RANK', '0')}\n"
        f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', '0')}"
    )

    my_unit = MyTrainUnit(
        args.input_dim,
        device,
        args.strategy,
        precision,
        args.lr,
        args.log_frequency_steps,
    )

    num_samples = 10240
    train_dataloader = prepare_dataloader(
        num_samples, args.input_dim, args.batch_size, device
    )

    t = Timer()
    t.start()

    train(my_unit, train_dataloader, max_epochs=args.max_epochs)

    t.stop()
    print(f"Total time for training {args.max_epochs} epochs: {t.total_time_seconds}")


class ValidateAccumGradBatchesAction(argparse.Action):
    # Validate input to accum_grad_batches
    def __call__(self, parser, namespace, values, option_string=None):  # pyre-ignore
        if values < 1:
            raise ValueError(f"`accum_grad_batches` must be an int >= 1. Got {values}.")
        setattr(namespace, self.dest, values)


def get_args(argv: List[str]) -> Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--input-dim", type=int, default=32, help="input dimension")
    parser.add_argument("--max-epochs", type=int, default=2, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--log-frequency-steps", type=int, default=10, help="log every n steps"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="fp16 or bf16",
        choices=["fp16", "bf16"],
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="define the training strategy (ddp or fsdp)",
        choices=["ddp", "fsdp"],
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
