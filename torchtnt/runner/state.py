# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Iterable, Optional

from torchtnt.runner.progress import Progress


class EntryPoint(Enum):
    FIT = auto()
    TRAIN = auto()
    EVALUATE = auto()
    PREDICT = auto()


@dataclass
class PhaseState:
    """State for each phase (train, eval, predict)"""

    progress: Progress

    # input arguments
    # pyre-ignore: Invalid type variable [34]
    dataloader: Iterable[Any]
    max_epochs: Optional[int] = None  # used only for train
    max_steps_per_epoch: Optional[int] = None
    evaluate_every_n_steps: Optional[int] = None  # used only for evaluate
    evaluate_every_n_epochs: Optional[int] = None  # used only for evaluate

    # contains the output from the last call to the user's `*_step` method
    # pyre-ignore: Invalid type variable [34]
    step_output: Any = None


@dataclass
class State:
    """Parent State class which can contain up to 3 instances of PhaseState, for the 3 phases.

    A new State class is created (and the previous one erased) each time an entry point is called.
    """

    entry_point: EntryPoint

    train_state: Optional[PhaseState] = None
    eval_state: Optional[PhaseState] = None
    predict_state: Optional[PhaseState] = None
