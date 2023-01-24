#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import tempfile
import uuid
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import launcher as pet
from torch.utils.data.dataset import Dataset, TensorDataset
from torcheval.metrics import BinaryAccuracy
from torchtnt.framework import AutoUnit, fit, init_fit_state, State
from torchtnt.utils import get_timer_summary, init_from_env, seed, TLRScheduler
from torchtnt.utils.loggers import TensorBoardLogger
from typing_extensions import Literal

_logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


Batch = Tuple[torch.Tensor, torch.Tensor]
NUM_PROCESSES = 2


def prepare_module(input_dim: int, device: torch.device) -> nn.Module:
    """
    Instantiate a nn.Module which will define the architecture of your model. If using a data parallel technique, wrap the module in DDP or FSDP.
    See https://pytorch.org/docs/stable/generated/torch.nn.Module.html for docs.
    """
    return nn.Linear(input_dim, 1, device=device)


def _generate_dataset(num_samples: int, input_dim: int) -> Dataset[Batch]:
    """Returns a dataset of random inputs and labels for binary classification."""
    # TODO: use datapipes/dataloaderV2
    data = torch.randn(num_samples, input_dim)
    labels = torch.randint(low=0, high=2, size=(num_samples,))
    return TensorDataset(data, labels)


def prepare_dataloader(
    num_samples: int, input_dim: int, batch_size: int, device: torch.device
) -> torch.utils.data.DataLoader:
    """Instantiate DataLoader"""
    # pin_memory enables faster host to GPU copies
    on_cuda = device.type == "cuda"
    return torch.utils.data.DataLoader(
        _generate_dataset(num_samples, input_dim),
        batch_size=batch_size,
        pin_memory=on_cuda,
    )


class MyUnit(AutoUnit[Batch]):
    def __init__(
        self,
        *,
        tb_logger: TensorBoardLogger,
        train_accuracy: BinaryAccuracy,
        **kwargs: Dict[str, Any],  # kwargs to be passed to AutoUnit
    ):
        super().__init__(**kwargs)
        self.tb_logger = tb_logger
        # create an accuracy Metric to compute the accuracy of training
        self.train_accuracy = train_accuracy
        self.loss = None

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return optimizer, lr_scheduler

    def compute_loss(self, state: State, data: Batch) -> Tuple[torch.Tensor, Any]:
        inputs, targets = data
        # convert targets to float Tensor for binary_cross_entropy_with_logits
        targets = targets.float()
        outputs = self.module(inputs)
        outputs = torch.squeeze(outputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)

        return loss, outputs

    def update_metrics(
        self,
        state: State,
        data: Batch,
        loss: torch.Tensor,
        outputs: Any,
    ) -> None:
        self.loss = loss
        _, targets = data
        self.train_accuracy.update(outputs, targets)

    def log_metrics(
        self, state: State, step: int, interval: Literal["step", "epoch"]
    ) -> None:
        self.tb_logger.log("loss", self.loss, step)

        accuracy = self.train_accuracy.compute()
        self.tb_logger.log("accuracy", accuracy, step)

    def on_train_epoch_end(self, state: State) -> None:
        super().on_train_epoch_end(state)
        # reset the metric every epoch
        self.train_accuracy.reset()


def main(args: Namespace) -> None:
    # seed the RNG for better reproducibility. see docs https://pytorch.org/docs/stable/notes/randomness.html
    seed(args.seed)

    # device and process group initialization
    device = init_from_env()

    path = tempfile.mkdtemp()
    tb_logger = TensorBoardLogger(path)

    module = prepare_module(args.input_dim, device)
    train_accuracy = BinaryAccuracy(device=device)

    my_unit = MyUnit(
        tb_logger=tb_logger,
        train_accuracy=train_accuracy,
        module=module,
        device=device,
        strategy="ddp",
        log_frequency_steps=args.log_frequency_steps,
        precision=args.precision,
        gradient_accumulation_steps=4,
        detect_anomaly=True,
        clip_grad_norm=1.0,
    )

    train_dataloader = prepare_dataloader(
        args.num_batches_per_epoch, args.input_dim, args.batch_size, device
    )
    eval_dataloader = prepare_dataloader(
        args.num_batches_per_epoch, args.input_dim, args.batch_size, device
    )
    state = init_fit_state(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_epochs=args.max_epochs,
    )

    fit(state, my_unit)
    print(get_timer_summary(state.timer))


def get_args() -> Namespace:
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
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    lc = pet.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=NUM_PROCESSES,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        monitor_interval=1,
    )

    pet.elastic_launch(lc, entrypoint=main)(args)
