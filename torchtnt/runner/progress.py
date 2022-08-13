# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class Progress:
    """Use a dataclass for typed access to fields and to add state_dict/load_state_dict for convenience for checkpointing."""

    num_epochs_completed: int = 0
    num_steps_completed: int = 0
    num_steps_completed_in_epoch: int = 0

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
