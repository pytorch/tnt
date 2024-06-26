# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import time
from concurrent.futures import Future
from typing import Any, Dict, Iterable, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.planner import LoadPlanner, SavePlanner
from torch.distributed.checkpoint.storage import StorageReader, StorageWriter

from torchtnt.framework.callbacks._checkpoint_utils import (
    _prepare_app_state_for_checkpoint,
    _prepare_app_state_for_restore,
    _TRAIN_DL_STATE_KEY,
)

from torchtnt.framework.callbacks.base_checkpointer import BaseCheckpointer
from torchtnt.framework.callbacks.checkpointer_types import KnobOptions, RestoreOptions
from torchtnt.framework.state import State
from torchtnt.framework.unit import (
    AppStateMixin,
    TEvalUnit,
    TPredictUnit,
    TTrainData,
    TTrainUnit,
)
from torchtnt.framework.utils import get_timing_context
from torchtnt.utils.checkpoint import BestCheckpointConfig, CheckpointPath
from torchtnt.utils.optimizer import init_optim_state
from torchtnt.utils.rank_zero_log import rank_zero_info, rank_zero_warn
from torchtnt.utils.stateful import MultiStateful, Stateful


logger: logging.Logger = logging.getLogger(__name__)

_LATEST_DCP_AVAIL: bool = True
try:
    from torch.distributed.checkpoint._fsspec_filesystem import (
        FsspecReader as Reader,
        FsspecWriter as Writer,
    )
except ModuleNotFoundError:
    logger.warn(
        "To use FsspecReader / FsspecWriter, please install latest pytorch version"
    )
    _LATEST_DCP_AVAIL = False
    from torch.distributed.checkpoint import (
        FileSystemReader as Reader,
        FileSystemWriter as Writer,
    )


class DistributedCheckpointSaver(BaseCheckpointer):
    """
    A callback which periodically saves the application state during training using `Distributed Checkpoint <https://pytorch.org/docs/stable/distributed.checkpoint.html>`_.

    This callback supplements the application state provided by :class:`torchtnt.unit.AppStateMixin`
    with the train progress, and train dataloader (if applicable).

    If used with :func:`torchtnt.framework.fit`, this class will also save the evaluation progress state.

    Checkpoints will be saved under ``dirpath/epoch_{epoch}_step_{step}`` where step is the *total* number of training steps completed across all epochs.

    Args:
        dirpath: Parent directory to save snapshots to.
        save_every_n_train_steps: Frequency of steps with which to save snapshots during the train epoch. If None, no intra-epoch snapshots are generated.
        save_every_n_epochs: Frequency of epochs with which to save snapshots during training. If None, no end-of-epoch snapshots are generated.
        save_every_n_eval_epochs: Frequency of evaluation epochs with which to save checkpoints during training. Use this if wanting to save checkpoints after every eval epoch during fit.
        keep_last_n_checkpoints: Number of most recent checkpoints to keep. If None, all checkpoints are kept. If an excess of existing checkpoints are present, the oldest ones will be deleted to clean the difference. If best checkpoint config is enabled, this param will manage the top n checkpoints instead.
        best_checkpoint_config: Configuration for saving the best checkpoint based on a monitored metric. The metric is read off the attribute of the unit prior to checkpoint.
        process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
        async_checkpoint: Whether to perform asynchronous checkpointing. Default: ``True``.
        knob_options: Additional keyword options for StorageWriter. <https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter/>

    Note:
        If torch.distributed is available and a process group is initialized, dcp assumes the intention is to save/load checkpoints in distributed fashion.
        Additionally, a gloo process group must be initialized for async_checkpoint. For workloads that require nccl, the recommended initialization is 'cpu:gloo,cuda:nccl'

    Note:
        If checkpointing FSDP model, you can set state_dict type calling `set_state_dict_type <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type>`_ prior to starting training.

    Note:
        If best_checkpoint_config is enabled, the attribute must be on the unit upon checkpoint time, and must be castable to "float". This value must be maintained by the unit, and updated
        appropriately. For example, if logging validation accuracy, the unit must be responsible for maintaining the value and resetting it when the epoch ends.
    """

    metadata_fname: Optional[str] = ".metadata"

    def __init__(
        self,
        dirpath: str,
        *,
        save_every_n_train_steps: Optional[int] = None,
        save_every_n_epochs: Optional[int] = None,
        save_every_n_eval_epochs: Optional[int] = None,
        keep_last_n_checkpoints: Optional[int] = None,
        best_checkpoint_config: Optional[BestCheckpointConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        async_checkpoint: bool = False,
        knob_options: Optional[KnobOptions] = None,
    ) -> None:
        super().__init__(
            dirpath=dirpath,
            save_every_n_train_steps=save_every_n_train_steps,
            save_every_n_epochs=save_every_n_epochs,
            save_every_n_eval_epochs=save_every_n_eval_epochs,
            keep_last_n_checkpoints=keep_last_n_checkpoints,
            best_checkpoint_config=best_checkpoint_config,
            process_group=process_group,
        )
        self._async_checkpoint = async_checkpoint

        self._knob_options: KnobOptions = knob_options or KnobOptions()
        self._prev_snapshot: Optional[Future] = None

    def _checkpoint_impl(
        self,
        state: State,
        unit: AppStateMixin,
        *,
        checkpoint_path: str,
        hook: str,
        planner: Optional[SavePlanner] = None,
        storage_writer: Optional[StorageWriter] = None,
    ) -> bool:
        if hook not in [
            "on_train_step_end",
            "on_train_epoch_end",
            "on_train_end",
            "on_eval_epoch_end",
        ]:
            raise RuntimeError(f"Unexpected hook encountered '{hook}'")

        intra_epoch = hook == "on_train_step_end"
        curr_snapshot_wait = hook == "on_train_end"

        app_state = _prepare_app_state_for_checkpoint(state, unit, intra_epoch)
        # TODO: evaluate whether we need to implement the equivalent of torchsnapshot.RNGState()
        if self._async_checkpoint:
            with get_timing_context(state, f"{self.__class__.__name__}.async_save"):
                # TODO checkpoint is not truly successful
                # since this is async checkpointed, so in
                # future, add logic to set  successful flag
                # only when checkpoint is fully written
                checkpoint_success = self._async_save(
                    checkpoint_path, app_state, planner, storage_writer
                )
                if curr_snapshot_wait:
                    self._wait()
        else:
            with get_timing_context(state, f"{self.__class__.__name__}.save"):
                checkpoint_success = self._save(
                    checkpoint_path, app_state, planner, storage_writer
                )

        return checkpoint_success

    def _wait(self) -> None:
        if self._prev_snapshot is not None:
            self._prev_snapshot.result()

    def _async_save(
        self,
        checkpoint_id: str,
        app_state: Dict[str, Stateful],
        planner: Optional[SavePlanner] = None,
        storage_writer: Optional[StorageWriter] = None,
    ) -> bool:

        if planner is None:
            planner = DefaultSavePlanner()

        if storage_writer is None:
            storage_writer = Writer(checkpoint_id, **self.default_writer_options)

        if self._prev_snapshot is not None:
            if not self._prev_snapshot.done():
                rank_zero_warn(
                    (
                        "Waiting on previous checkpoint to finish... Consider modifying checkpointing "
                        f"frequency if this is an issue. Current value (current {self._save_every_n_train_steps})"
                    ),
                    logger=logger,
                )
                t0 = time.monotonic()
                self._wait()
                rank_zero_warn(
                    f"Waiting on previous checkpoint for {time.monotonic()-t0:.3f} seconds",
                    logger=logger,
                )
            else:
                self._wait()

        self._prev_snapshot = dcp.async_save(
            state_dict={"app_state": MultiStateful(app_state)},
            checkpoint_id=checkpoint_id,
            process_group=self._process_group,
            storage_writer=storage_writer,
            planner=planner,
        )

        return True

    def _save(
        self,
        checkpoint_id: str,
        app_state: Dict[str, Stateful],
        planner: Optional[SavePlanner] = None,
        storage_writer: Optional[StorageWriter] = None,
    ) -> bool:
        # Initialize DefaultSavePlanner and FsspecWriter if not provided
        if planner is None:
            planner = DefaultSavePlanner()

        if storage_writer is None:
            storage_writer = Writer(checkpoint_id, **self.default_writer_options)

        try:
            dcp.save(
                state_dict={"app_state": MultiStateful(app_state)},
                checkpoint_id=checkpoint_id,
                process_group=self._process_group,
                storage_writer=storage_writer,
                planner=planner,
            )
        except AttributeError:
            dcp.save_state_dict(
                state_dict={"app_state": MultiStateful(app_state)},
                process_group=self._process_group,
                storage_writer=storage_writer,
                planner=planner,
            )

        return True

    def on_exception(
        self,
        state: State,
        unit: Union[TTrainUnit, TEvalUnit, TPredictUnit],
        exc: BaseException,
    ) -> None:
        self._wait()

    @staticmethod
    def restore(
        path: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        knob_options: Optional[KnobOptions] = None,
        planner: Optional[LoadPlanner] = None,
        storage_reader: Optional[StorageReader] = None,
    ) -> None:
        """Utility method to restore dcp checkpoint from a path.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            path: Path of the snapshot to restore.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world) Note:
                If torch.distributed is available and a process group is initialized, dcp assumes the intention is to save/load checkpoints in distributed fashion.
            restore_options: Controls what to  filter when restoring the state.
            knob_options: Additional keyword options for StorageWriter and StorageReader
            planner: Instance of LoadPlanner. If this is not specificed, the default planner will be used. (Default: ``None``)
            storage_reader: Instance of StorageReader used to perform reads. If this is not specified, it will automatically infer
                            the reader based on the checkpoint_id. If checkpoint_id is also None, an exception will be raised. (Default: ``None``)
        """

        restore_options = restore_options or RestoreOptions()
        app_state = _prepare_app_state_for_restore(unit, restore_options)

        if storage_reader is None:
            storage_reader = Reader(path)

        if planner is None:
            allow_partial_load = not restore_options.strict
            planner = DefaultLoadPlanner(allow_partial_load=allow_partial_load)

        if train_dataloader is not None:
            if not isinstance(train_dataloader, Stateful):
                rank_zero_warn(
                    "train_dataloader was passed to `restore` but the dataloader does not implement the Stateful protocol to load states"
                )
            else:
                # request to restore the dataloader state only if
                # the persisted snapshot state includes the dataloader entry
                metadata = storage_reader.read_metadata()

                for key in metadata.state_dict_metadata.keys():
                    if _TRAIN_DL_STATE_KEY in key:
                        app_state[_TRAIN_DL_STATE_KEY] = train_dataloader
                        break

                if _TRAIN_DL_STATE_KEY not in app_state:
                    rank_zero_warn(
                        "train_dataloader was passed to `restore` but no train dataloader exists in the Snapshot"
                    )

        # necessary for loading optimizers since states are initialized lazy
        for obj in app_state.values():
            # sometimes optimizers are actually held in a wrapper which handles calling
            # state_dict and load_state_dict, sa is the case for
            # `torchtnt.utils.prepare_module.FSDPOptimizerWrapper`, this handles that case.
            optimizer = getattr(obj, "optimizer", obj)
            if isinstance(optimizer, torch.optim.Optimizer):
                init_optim_state(optimizer)

        try:
            dcp.load(
                {"app_state": MultiStateful(app_state)},
                checkpoint_id=path,
                storage_reader=storage_reader,
                planner=planner,
                process_group=process_group,
            )
        except AttributeError:
            dcp.load_state_dict(
                {"app_state": MultiStateful(app_state)},
                storage_reader=storage_reader,
                process_group=process_group,
                planner=planner,
            )
        rank_zero_info(f"Restored snapshot from path: {path}", logger=logger)

    def _does_checkpoint_exist(
        self,
        checkpoint_path: CheckpointPath,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> bool:
        # if we are still checkpointing, this might cause a collective hang.
        # so wait here instead
        self._wait()

        return super()._does_checkpoint_exist(
            checkpoint_path=checkpoint_path, process_group=process_group
        )

    @property
    def default_writer_options(self) -> Dict[str, Any]:
        # defaults are picked to to match TSS defaults
        # TODO: expose these options in KnobOptions
        dcp_options = {
            "thread_count": self._knob_options.max_per_rank_io_concurrency or 16,
            "sync_files": False,
            "overwrite": True,
        }
        if dcp_options["thread_count"] > 1:
            dcp_options["single_file_per_rank"] = False

        return dcp_options
