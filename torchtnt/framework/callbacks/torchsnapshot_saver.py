# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Dict, List, Optional, Set, Union

from pyre_extensions import none_throws
from torchsnapshot.snapshot import PendingSnapshot, Snapshot

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import (
    _Stateful as StatefulProtocol,
    TEvalUnit,
    TPredictUnit,
    TTrainUnit,
)
from torchtnt.utils import rank_zero_info, rank_zero_warn

try:
    import torchsnapshot

    _TStateful = torchsnapshot.Stateful
    _TORCHSNAPSHOT_AVAILABLE = True
except Exception:
    _TStateful = StatefulProtocol
    _TORCHSNAPSHOT_AVAILABLE = False

_EVAL_PROGRESS_STATE_KEY = "eval_progress"
_RNG_STATE_KEY = "rng_state"
_TRAIN_PROGRESS_STATE_KEY = "train_progress"
_TRAIN_DL_STATE_KEY = "train_dataloader"

logger: logging.Logger = logging.getLogger(__name__)


class TorchSnapshotSaver(Callback):
    """
    A callback which periodically saves the application state during training using TorchSnapshot: https://pytorch.org/torchsnapshot.

    This callback supplements the application state provided by :class:`torchtnt.unit.AppStateMixin`
    with the train progress, train dataloader (if applicable), and random number generator state.

    If used with :func:`torchtnt.framework.fit`, this class will also save the evaluation progress state.

    Args:
        dirpath: Parent directory to save snapshots to.
        save_every_n_train_steps: Frequency of steps with which to save snapshots during the train epoch. If None, no intra-epoch snapshots are generated.
        save_every_n_epochs: Frequency of epochs with which to save snapshots during training. If None, no end-of-epoch snapshots are generated.
        replicated: A glob-pattern of replicated key names that indicate which application state entries have the same state across all processes.
            For more information, see https://pytorch.org/torchsnapshot/main/api_reference.html#torchsnapshot.Snapshot.take .
    """

    def __init__(
        self,
        dirpath: str,
        *,
        save_every_n_train_steps: Optional[int] = None,
        save_every_n_epochs: Optional[int] = None,
        replicated: Optional[List[str]] = None,
    ) -> None:
        _validate_snapshot_available()
        if save_every_n_train_steps is not None and save_every_n_train_steps < 0:
            raise ValueError(
                f"Invalid value passed for save_every_n_train_steps. Expected to receive either None or non-negative number, but received {save_every_n_train_steps}"
            )
        if save_every_n_epochs is not None and save_every_n_epochs < 0:
            raise ValueError(
                f"Invalid value passed for save_every_n_epochs. Expected to receive either None or non-negative number, but received {save_every_n_epochs}"
            )
        self._save_every_n_epochs = save_every_n_epochs
        self._save_every_n_train_steps = save_every_n_train_steps
        self._dirpath: str = dirpath
        self._replicated: Set[str] = set(replicated or [])

        self._prev_snapshot: Optional[PendingSnapshot] = None

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        """Validate there's no key collision for the app state."""
        app_state = unit.app_state()
        _check_app_state_collision(app_state)

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        train_state = none_throws(state.train_state)

        global_step = train_state.progress.num_steps_completed
        every_n_train_steps = self._save_every_n_train_steps
        if every_n_train_steps is None or global_step % every_n_train_steps != 0:
            return

        app_state = _get_app_state(state, unit, self._replicated, intra_epoch=True)

        # save snapshot to predetermined path
        # TODO: discuss whether this path should be customized
        epoch = train_state.progress.num_epochs_completed
        snapshot_path = _get_snapshot_save_path(self._dirpath, epoch, global_step)
        self._async_snapshot(snapshot_path, app_state, wait=False)

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        train_state = none_throws(state.train_state)

        train_progress = train_state.progress
        epoch = train_progress.num_epochs_completed
        every_n_epochs = self._save_every_n_epochs
        if every_n_epochs is None or epoch % every_n_epochs != 0:
            return

        app_state = _get_app_state(
            state, unit, replicated=self._replicated, intra_epoch=False
        )

        # save snapshot to predetermined path
        # TODO: discuss whether this path should be customized
        global_step = train_progress.num_steps_completed
        snapshot_path = _get_snapshot_save_path(self._dirpath, epoch, global_step)
        self._async_snapshot(snapshot_path, app_state, wait=True)

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        self._wait()

    def on_exception(
        self,
        state: State,
        unit: Union[TTrainUnit, TEvalUnit, TPredictUnit],
        exc: BaseException,
    ) -> None:
        self._wait()

    def _wait(self) -> None:
        if self._prev_snapshot is not None:
            self._prev_snapshot.wait()

    def _async_snapshot(
        self, snapshot_path: str, app_state: Dict[str, _TStateful], *, wait: bool
    ) -> bool:
        prev_snapshot = self._prev_snapshot
        if prev_snapshot is not None:
            if prev_snapshot.path == snapshot_path:
                # Snapshot for this step already has been saved.
                # This can happen if we call _async_snapshot twice at the same step.
                return True
            still_pending = not prev_snapshot.done()
            if still_pending and wait:
                prev_snapshot.wait()
            elif still_pending:
                rank_zero_warn(
                    f"Still writing previous snapshot, will skip this one. Consider increasing 'frequency' (current {self._save_every_n_train_steps})",
                    logger=logger,
                )
                return False

        self._prev_snapshot = Snapshot.async_take(
            str(snapshot_path), app_state=app_state, replicated=list(self._replicated)
        )
        rank_zero_info(f"Saving snapshot to path: {snapshot_path}", logger=logger)
        return True

    @staticmethod
    def restore(
        path: str,
        state: State,
        unit: TTrainUnit,
        *,
        restore_train_progress: bool = True,
        restore_train_dataloader: bool = True,
        restore_eval_progress: bool = True,
    ) -> None:
        """Utility method to restore snapshot state from a path.

        Since the class also manages saving the progress and dataloader states,
        this method handles their restoration. There are additional flags offered
        should the user want to skip loading these states. By default, the train progress,
        train dataloader, and eval progress are restored, if applicable.
        """

        _validate_snapshot_available()
        app_state = unit.app_state()
        _check_app_state_collision(app_state)

        snapshot = torchsnapshot.Snapshot(path)

        train_state = none_throws(state.train_state)

        rng_state = torchsnapshot.RNGState()
        app_state[_RNG_STATE_KEY] = rng_state

        if restore_train_progress:
            train_progress = train_state.progress
            app_state[_TRAIN_PROGRESS_STATE_KEY] = train_progress

        if restore_train_dataloader:
            # request to restore the dataloader state only if
            # the persisted snapshot state includes the dataloader entry
            manifest = snapshot.get_manifest()
            for key in manifest:
                if _TRAIN_DL_STATE_KEY in key:
                    app_state[_TRAIN_DL_STATE_KEY] = train_state.dataloader
                    break

        if state.entry_point == EntryPoint.FIT and restore_eval_progress:
            # include evaluation states if fitting
            eval_state = none_throws(state.eval_state)
            app_state[_EVAL_PROGRESS_STATE_KEY] = eval_state.progress

        snapshot.restore(app_state)


def _validate_snapshot_available() -> None:
    if not _TORCHSNAPSHOT_AVAILABLE:
        raise RuntimeError(
            "TorchSnapshotSaver support requires torchsnapshot. "
            "Please make sure ``torchsnapshot`` is installed. "
            "Installation: https://github.com/pytorch/torchsnapshot#install"
        )


def _get_snapshot_save_path(dirpath: str, epoch: int, step: int) -> str:
    return os.path.join(dirpath, f"epoch_{epoch}_step_{step}")


def _get_app_state(
    state: State, unit: TTrainUnit, replicated: Set[str], *, intra_epoch: bool
) -> Dict[str, _TStateful]:
    train_state = none_throws(state.train_state)

    train_progress = train_state.progress
    app_state = unit.app_state()

    rng_state = torchsnapshot.RNGState()
    app_state[_RNG_STATE_KEY] = rng_state
    app_state[_TRAIN_PROGRESS_STATE_KEY] = train_progress
    train_prog_glob = f"{_TRAIN_PROGRESS_STATE_KEY}/*"
    replicated.add(train_prog_glob)

    # for intra-epoch checkpointing, include dataloader states
    train_dl = train_state.dataloader
    if intra_epoch and isinstance(train_dl, _TStateful):
        app_state[_TRAIN_DL_STATE_KEY] = train_dl

    if state.entry_point == EntryPoint.FIT:
        # include evaluation states if fitting
        eval_state = none_throws(state.eval_state)

        app_state[_EVAL_PROGRESS_STATE_KEY] = eval_state.progress
        eval_prog_glob = f"{_EVAL_PROGRESS_STATE_KEY}/*"
        replicated.add(eval_prog_glob)

    return app_state


def _check_app_state_collision(app_state: Dict[str, _TStateful]) -> None:
    keys_to_check = (
        _TRAIN_PROGRESS_STATE_KEY,
        _TRAIN_DL_STATE_KEY,
        _RNG_STATE_KEY,
        _EVAL_PROGRESS_STATE_KEY,
    )
    for key in keys_to_check:
        if key in app_state:
            raise RuntimeError(
                f"Detected collision for key in app state: {key}. TorchSnapshotSaver expects to save and load this key."
            )
