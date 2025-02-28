# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, Union

import torch
from torchtnt.utils.lr_scheduler import TLRScheduler
from torchtnt.utils.prepare_module import FSDP2OptimizerWrapper, FSDPOptimizerWrapper
from torchtnt.utils.progress import Progress

from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Stateful(Protocol):
    """Defines the interface for checkpoint saving and loading."""

    def state_dict(self) -> Dict[str, Any]: ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...


StatefulDict = Dict[str, Stateful]
ModuleDict = Dict[str, torch.nn.Module]
OptimizerAndLRSchedulerDict = Dict[
    str,
    Union[
        TLRScheduler, torch.optim.Optimizer, FSDPOptimizerWrapper, FSDP2OptimizerWrapper
    ],
]
ProgressDict = Dict[str, Progress]


class MultiStateful:
    """
    Wrapper for multiple stateful objects. Necessary because we might have multiple nn.Modules or multiple optimizers,
    but save/load_checkpoint APIs may only accept one stateful object.

    Stores state_dict as a dict of state_dicts.
    """

    def __init__(
        self,
        stateful_objs: Union[
            StatefulDict, ModuleDict, OptimizerAndLRSchedulerDict, ProgressDict
        ],
    ) -> None:
        self.stateful_objs = stateful_objs

    def state_dict(self) -> Dict[str, Any]:
        return {k: v.state_dict() for k, v in self.stateful_objs.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for k in state_dict:
            self.stateful_objs[k].load_state_dict(state_dict[k])


@runtime_checkable
class MetricStateful(Protocol):
    """
    Defines the interfaces for metric objects that can be saved and loaded from checkpoints.
    This conforms to the API exposed by major metric libraries like torcheval.
    """

    def update(self, *_: Any, **__: Any) -> None: ...

    # pyre-ignore[3]: Metric computation may return any type depending on the implementation
    def compute(self) -> Any: ...

    def state_dict(self) -> Dict[str, Any]: ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...
