# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict


class Progress:
    """Class to track progress during the loop. Includes state_dict/load_state_dict for convenience for checkpointing."""

    def __init__(
        self,
        num_epochs_completed: int = 0,
        num_steps_completed: int = 0,
        num_steps_completed_in_epoch: int = 0,
    ) -> None:
        self._num_epochs_completed: int = num_epochs_completed
        self._num_steps_completed: int = num_steps_completed
        self._num_steps_completed_in_epoch: int = num_steps_completed_in_epoch

    @property
    def num_epochs_completed(self) -> int:
        return self._num_epochs_completed

    @property
    def num_steps_completed(self) -> int:
        return self._num_steps_completed

    @property
    def num_steps_completed_in_epoch(self) -> int:
        return self._num_steps_completed_in_epoch

    def increment_step(self) -> None:
        self._num_steps_completed += 1
        self._num_steps_completed_in_epoch += 1

    def increment_epoch(self) -> None:
        self._num_epochs_completed += 1
        self._num_steps_completed_in_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "num_epochs_completed": self._num_epochs_completed,
            "num_steps_completed": self._num_steps_completed,
            "num_steps_completed_in_epoch": self._num_steps_completed_in_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._num_epochs_completed = state_dict["num_epochs_completed"]
        self._num_steps_completed = state_dict["num_steps_completed"]
        self._num_steps_completed_in_epoch = state_dict["num_steps_completed_in_epoch"]
