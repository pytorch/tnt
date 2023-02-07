# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ignore errors due to `Any` type
# pyre-ignore-all-errors[2]
# pyre-ignore-all-errors[3]
# pyre-ignore-all-errors[4]

import logging
from enum import auto, Enum
from typing import Any, Iterable, Optional

from torchtnt.framework.progress import Progress
from torchtnt.utils.timer import Timer

_logger: logging.Logger = logging.getLogger(__name__)


def _check_loop_condition(name: str, val: Optional[int]) -> None:
    if val is not None and val < 0:
        raise ValueError(
            f"Invalid value provided for {name}. Expected a non-negative integer or None, but received {val}."
        )


class EntryPoint(Enum):
    """
    Enum for the user-facing functions offered by the TorchTNT framework.
    - :py:func:`~torchtnt.framework.fit`
    - :py:func:`~torchtnt.framework.train`
    - :py:func:`~torchtnt.framework.evaluate`
    - :py:func:`~torchtnt.framework.predict`
    """

    FIT = auto()
    TRAIN = auto()
    EVALUATE = auto()
    PREDICT = auto()


class ActivePhase(Enum):
    """Enum for the currently active phase.

    This class complements :class:`EntryPoint` by specifying the active phase for each function.
    More than one phase value can be set while a :class:`EntryPoint` is running:
        - ``EntryPoint.FIT`` - ``ActivePhase.{TRAIN,EVALUATE}``
        - ``EntryPoint.TRAIN`` - ``ActivePhase.TRAIN``
        - ``EntryPoint.EVALUATE`` - ``ActivePhase.EVALUATE``
        - ``EntryPoint.PREDICT`` - ``ActivePhase.PREDICT``

    This can be used within hooks such as :meth:`~torchtnt.framework.unit._OnExceptionMixin.on_exception`
    to determine within which of training, evaluation, or prediction the hook is being called.
    """

    TRAIN = auto()
    EVALUATE = auto()
    PREDICT = auto()


class PhaseState:
    """State for each phase (train, eval, predict).
    Modified by the framework, read-only for the user.
    """

    def __init__(
        self,
        *,
        dataloader: Iterable[Any],
        progress: Optional[Progress] = None,
        max_epochs: Optional[int] = None,  # used only for train
        max_steps: Optional[int] = None,  # used only for train
        max_steps_per_epoch: Optional[int] = None,
        evaluate_every_n_steps: Optional[int] = None,  # used only for evaluate
        evaluate_every_n_epochs: Optional[int] = None,  # used only for evaluate
    ) -> None:
        _check_loop_condition("max_epochs", max_epochs)
        _check_loop_condition("max_steps", max_steps)
        _check_loop_condition("max_steps_per_epoch", max_steps_per_epoch)
        _check_loop_condition("evaluate_every_n_steps", evaluate_every_n_steps)
        _check_loop_condition("evaluate_every_n_epochs", evaluate_every_n_epochs)

        self._dataloader: Iterable[Any] = dataloader
        self._progress: Progress = progress or Progress()
        self._max_epochs = max_epochs
        self._max_steps = max_steps
        self._max_steps_per_epoch = max_steps_per_epoch
        self._evaluate_every_n_steps = evaluate_every_n_steps
        self._evaluate_every_n_epochs = evaluate_every_n_epochs

        self._step_output: Any = None
        self._is_last_batch: bool = False  # only used for train

    @property
    def dataloader(self) -> Iterable[Any]:
        """Dataloader defined by the user."""
        return self._dataloader

    @property
    def progress(self) -> Progress:
        """An instance of :class:`~torchtnt.framework.Progress` which contains information about the current progress of the loop."""
        return self._progress

    @property
    def max_epochs(self) -> Optional[int]:
        """Maximum number of epochs to train, defined by the user."""
        return self._max_epochs

    @property
    def max_steps(self) -> Optional[int]:
        """Maximum number of steps to train, defined by the user."""
        return self._max_steps

    @property
    def max_steps_per_epoch(self) -> Optional[int]:
        """Maximum number of steps to run per epoch, defined by the user."""
        return self._max_steps_per_epoch

    @property
    def evaluate_every_n_steps(self) -> Optional[int]:
        """Frequency with which to evaluate in terms of training steps, when running :func:`~torchtnt.framework.fit`. Defined by the user."""
        return self._evaluate_every_n_steps

    @property
    def evaluate_every_n_epochs(self) -> Optional[int]:
        """Frequency with which to evaluate in terms of training epochs, when running :func:`~torchtnt.framework.fit`. Defined by the user."""
        return self._evaluate_every_n_epochs

    @property
    def step_output(self) -> Any:
        """Output of the last step."""
        return self._step_output

    @property
    def is_last_batch(self) -> bool:
        """Returns true if current batch of data is the last batch."""
        return self._is_last_batch


class State:
    """Parent State class which can contain up to 3 instances of PhaseState, for the 3 phases.
    Modified by the framework, read-only for the user.
    """

    def __init__(
        self,
        *,
        entry_point: EntryPoint,
        timer: Optional[Timer] = None,
        train_state: Optional[PhaseState] = None,
        eval_state: Optional[PhaseState] = None,
        predict_state: Optional[PhaseState] = None,
    ) -> None:
        self._entry_point = entry_point
        self._timer: Timer = timer or Timer()
        self._train_state = train_state
        self._eval_state = eval_state
        self._predict_state = predict_state
        self._should_stop: bool = False
        self._active_phase: ActivePhase = ActivePhase.TRAIN

    @property
    def entry_point(self) -> EntryPoint:
        """Entry point used to start loop execution. (One of FIT, TRAIN, EVALUATE, PREDICT)."""
        return self._entry_point

    @property
    def active_phase(self) -> ActivePhase:
        """Current active phase of the loop. (One of TRAIN, EVALUATE, PREDICT)."""
        return self._active_phase

    @property
    def timer(self) -> Timer:
        """A :class:`~torchtnt.framework.Timer` object which records latencies of key events during loop execution."""
        return self._timer

    @property
    def train_state(self) -> Optional[PhaseState]:
        """A :class:`~torchtnt.framework.PhaseState` object which contains meta information about the train phase."""
        return self._train_state

    @property
    def eval_state(self) -> Optional[PhaseState]:
        """A :class:`~torchtnt.framework.PhaseState` object which contains meta information about the eval phase."""
        return self._eval_state

    @property
    def predict_state(self) -> Optional[PhaseState]:
        """A :class:`~torchtnt.framework.PhaseState` object which contains meta information about the predict phase."""
        return self._predict_state

    @property
    def should_stop(self) -> bool:
        """Read-only property for whether to terminate the loop after the current step completes."""
        return self._should_stop

    def stop(self) -> None:
        """Signal to the loop to end after the current step completes."""
        _logger.warning("Received signal to stop")
        self._should_stop = True
