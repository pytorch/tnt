# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Tuple, TypeVar

import torch

from torchtnt.runner.state import State

TBatch = TypeVar("TBatch")
TOutput = TypeVar("TOutput")
TSelf = TypeVar("TSelf")


class _Engine(ABC):
    @abstractmethod
    def train(self: TSelf, mode: bool = True) -> TSelf:
        """Puts the module in train mode
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train
        """

    @abstractmethod
    # pyre-fixme: Invalid type variable [34]
    def forward(self, *inputs: Any) -> TOutput:
        """Runs one forward pass"""

    @abstractmethod
    # pyre-fixme: Missing return annotation [3]
    def step(
        self,
        state: State,
        batch: TBatch,
    ) -> Tuple[torch.Tensor, Any]:
        """Runs either train step or eval step depending on the mode of the module"""
