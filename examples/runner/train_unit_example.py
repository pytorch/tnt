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
from torchtnt.loggers import TensorBoardLogger
from torchtnt.runner import init_train_state, State, train, TrainUnit
from torchtnt.runner.callbacks import (
    LearningRateMonitor,
    PyTorchProfiler,
    TensorBoardParameterMonitor,
)
from torchtnt.utils import get_timer_summary, init_from_env, seed

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
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_accuracy: BinaryAccuracy,
        tb_logger: TensorBoardLogger,
        log_frequency_steps: int,
    ):
        super().__init__()
        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # create an accuracy Metric to compute the accuracy of training
        self.train_accuracy = train_accuracy
        self.log_frequency_steps = log_frequency_steps

        self.tb_logger = tb_logger

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
        step_count = state.train_state.progress.num_steps_completed
        if (step_count + 1) % self.log_frequency_steps == 0:
            acc = self.train_accuracy.compute()
            self.tb_logger.log("loss", loss, step_count)
            self.tb_logger.log("accuracy", acc, step_count)

    def on_train_epoch_end(self, state: State) -> None:
        # compute and log the metric at the end of the epoch
        step_count = state.train_state.progress.num_steps_completed
        acc = self.train_accuracy.compute()
        self.tb_logger.log("accuracy_epoch", acc, step_count)

        # reset the metric every epoch
        self.train_accuracy.reset()

        # step the learning rate scheduler
        self.lr_scheduler.step()


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

    path = tempfile.mkdtemp()
    tb_logger = TensorBoardLogger(path)

    module = prepare_module(args.input_dim, device, args.strategy, precision)
    optimizer = torch.optim.SGD(module.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_accuracy = BinaryAccuracy(device=device)

    my_unit = MyTrainUnit(
        module,
        optimizer,
        lr_scheduler,
        train_accuracy,
        tb_logger,
        args.log_frequency_steps,
    )

    profiler = PyTorchProfiler(
        profiler=torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=path),
            with_stack=True,
        )
    )
    parameter_monitor = TensorBoardParameterMonitor(tb_logger)
    lr_monitor = LearningRateMonitor(tb_logger)

    num_samples = 10240
    train_dataloader = prepare_dataloader(
        num_samples, args.input_dim, args.batch_size, device
    )

    state = init_train_state(
        dataloader=train_dataloader,
        max_epochs=args.max_epochs,
    )

    train(
        state,
        my_unit,
        callbacks=[lr_monitor, parameter_monitor, profiler],
    )
    print(get_timer_summary(state.timer))


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
