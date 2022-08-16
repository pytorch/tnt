# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[3]

from typing import Any, Callable, List, Tuple, TypeVar, Union

import torch
from torch import nn
from torch.optim import Optimizer

from torchtnt.runner.engine import _Engine
from torchtnt.runner.state import State

TBatch = TypeVar("TBatch")
TOutput = TypeVar("TOutput")
TSelf = TypeVar("TSelf")


class SGDEngine(_Engine):
    def __init__(
        self,
        module: nn.Module,
        optimizers: Union[Optimizer, List[Optimizer]],
        # pyre-fixme[2]: Missing parameter annotation
        get_loss: Callable[[State, TBatch], Tuple[torch.Tensor, Any]],
    ) -> None:

        self.module = module
        self.optimizers: List[Optimizer] = (
            optimizers if isinstance(optimizers, List) else [optimizers]
        )
        self.get_loss = get_loss

    def train(self: TSelf, mode: bool = True) -> TSelf:
        """Puts the module in train mode
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train
        """
        # pyre-ignore: Undefined attribute [16]
        self.module.train(mode)
        return self

    # pyre-ignore: Invalid type variable [34]:
    def forward(self, *inputs: Any) -> TOutput:
        """Runs one forward pass"""
        outputs = self.module(*inputs)
        return outputs

    def step(
        self,
        state: State,
        batch: TBatch,
    ) -> Tuple[torch.Tensor, Any]:
        """Runs either train step or eval step depending on the mode of the module"""
        if self.module.training:
            return self._train_step(state, batch)
        else:
            return self._eval_step(state, batch)

    def _train_step(
        self,
        state: State,
        batch: TBatch,
    ) -> Tuple[torch.Tensor, Any]:
        loss, outputs = self.get_loss(state, batch)

        loss.backward()

        for optimizer in self.optimizers:
            # optimizer step
            optimizer.step()
            # sets gradients to zero
            optimizer.zero_grad()

        return loss, outputs

    def _eval_step(
        self,
        state: State,
        batch: TBatch,
    ) -> Tuple[torch.Tensor, Any]:
        loss, outputs = self.get_loss(state, batch)
        return loss, outputs
