# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import logging
from datetime import timedelta
from typing import Any, cast, Iterable, Literal, Optional, Union

import torch.distributed as dist
from pyre_extensions import none_throws
from torchtnt.framework.callback import Callback
from torchtnt.framework.callbacks._checkpoint_utils import _get_step_phase_mapping
from torchtnt.framework.callbacks.checkpointer_types import RestoreOptions
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import AppStateMixin, TEvalUnit, TTrainData, TTrainUnit
from torchtnt.utils.checkpoint import (
    BestCheckpointConfig,
    CheckpointManager,
    CheckpointPath,
    get_best_checkpoint_path,
    get_latest_checkpoint_path,
    MetricData,
    Phase,
)
from torchtnt.utils.distributed import PGWrapper
from torchtnt.utils.rank_zero_log import rank_zero_info, rank_zero_warn

logger: logging.Logger = logging.getLogger(__name__)


class BaseCheckpointer(Callback, metaclass=abc.ABCMeta):
    """
    Abstract base class for file-based state_dict checkpointing. This class can be used as the base of a checkpointing callback, and handles
    checkpointing frequency logic, checkpoint naming, checkpoint purging / upkeep, and process group synchronization. There are only two methods
    that need to be implemented by subclasses:

    1) ``_checkpoint_impl`` which implements the checkpoint saving logic, given the relevant checkpoint items and path.
    2) ``restore`` which implements restoring the checkpoint given the relevant checkpoint path.

    The subclass may override the ``metadata_fname`` attribute to specify the filename of the metadata file that will be written within the checkpoint directory.
    This will be used by this base class to ensure the integrity of the checkpoint.

    Args:
        dirpath: Parent directory to save checkpoints to.
        save_every_n_train_steps: Frequency of steps with which to save checkpoints during the train epoch. If None, no intra-epoch checkpoints are generated.
        save_every_n_epochs: Frequency of epochs with which to save checkpoints during training. If None, no end-of-epoch checkpoints are generated.
        save_every_n_eval_epochs: Frequency of evaluation epochs with which to save checkpoints during training. Use this if wanting to save checkpoints after every eval epoch during fit.
        keep_last_n_checkpoints: Number of most recent checkpoints to keep. If None, all checkpoints are kept. If an excess of existing checkpoints are present, the oldest ones will be deleted to clean the difference. If best checkpoint config is enabled, this param will manage the top n checkpoints instead.
        best_checkpoint_config: Configuration for saving the best checkpoint based on a monitored metric. The metric is read off the attribute of the unit prior to checkpoint.
        process_group: The process group on which the ranks will communicate on. If the process group is not gloo-based, a new gloo-based process group will be created.

    Note:
        If torch.distributed is available and default process group is initialized, the constructor will call a collective operation for rank 0 to broadcast the dirpath to all other ranks

    Note:
        This class assumes checkpoint items are saved in the directory provided in ``_checkpoint_impl`` and will be in the form of ``<dirpath>/<epoch>-<step>/``. Checkpoint contents
        should be stored within this directory, as deleting and retrieving latest checkpoint relies on reading the <epoch>-<step> directory name within <dirpath>

    Note:
        If best_checkpoint_config is enabled, the attribute must be on the unit upon checkpoint time, and must be castable to "float". This value must be maintained by the unit, and updated
        appropriately. For example, if logging validation accuracy, the unit must be responsible for maintaining the value and resetting it when the epoch ends. If the metric value is None, the
        checkpoint will be saved, without the metric value in the checkpoint name
    """

    metadata_fname: Optional[str] = None

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
    ) -> None:
        if save_every_n_train_steps is not None and save_every_n_train_steps <= 0:
            raise ValueError(
                f"Invalid value passed for save_every_n_train_steps. Expected to receive either None or positive number, but received {save_every_n_train_steps}"
            )
        if save_every_n_epochs is not None and save_every_n_epochs <= 0:
            raise ValueError(
                f"Invalid value passed for save_every_n_epochs. Expected to receive either None or positive number, but received {save_every_n_epochs}"
            )
        if keep_last_n_checkpoints is not None and keep_last_n_checkpoints <= 0:
            raise ValueError(
                f"Invalid value passed for keep_last_n_checkpoints. Expected to receive either None or positive number, but received {keep_last_n_checkpoints}"
            )

        self._best_checkpoint_config = best_checkpoint_config
        if best_checkpoint_config and best_checkpoint_config.mode not in {"min", "max"}:
            raise ValueError(
                f"Invalid value passed for best_checkpoint_config.mode. Expected to receive 'min' or 'max', but received {best_checkpoint_config.mode}"
            )

        self._save_every_n_train_steps = save_every_n_train_steps
        self._save_every_n_epochs = save_every_n_epochs
        self._save_every_n_eval_epochs = save_every_n_eval_epochs
        self._keep_last_n_checkpoints = keep_last_n_checkpoints

        self._process_group: Optional[dist.ProcessGroup] = None
        self._setup_gloo_pg(process_group)
        self._pg_wrapper = PGWrapper(process_group)

        self._checkpoint_manager = CheckpointManager(
            dirpath,
            best_checkpoint_config,
            keep_last_n_checkpoints,
            metadata_fname=self.metadata_fname,
            process_group=self._process_group,
        )

    def _setup_gloo_pg(self, process_group: Optional[dist.ProcessGroup]) -> None:
        """
        Setups gloo process group to be used for any collectives called during
        checkpointing. If global process group is already gloo, no action is required.
        Gloo is used over nccl for better compatibility.
        """
        if not dist.is_initialized():
            # there can be no process group
            return

        if process_group is None:
            # use global process group
            process_group = dist.group.WORLD

        # we create a new gloo process group if different backend is being used
        if dist.get_backend(process_group) != dist.Backend.GLOO:
            rank_zero_info("Creating new gloo process group for checkpointing.")
            self._process_group = dist.new_group(
                timeout=timedelta(seconds=3600), backend=dist.Backend.GLOO
            )
        else:
            self._process_group = process_group

    @property
    def dirpath(self) -> str:
        """Returns parent directory to save to."""
        return self._checkpoint_manager.dirpath

    def _generate_checkpoint_and_upkeep(
        self, state: State, unit: Union[TTrainUnit, TEvalUnit], hook: str
    ) -> bool:
        """
        Implementation for saving checkpoint while taking care of checkpoint
        name generation and cleanup of oldest checkpoints.

        Args:
            state: Current state of the trainer.
            unit: Current training unit.
            hook: Hook at which checkpoint is being saved.

        Returns:
            True if checkpoint was successfully saved. False otherwise.
        """
        # 1) generate checkpoint name
        epoch = cast(TTrainUnit, unit).train_progress.num_epochs_completed
        step_mapping = _get_step_phase_mapping(state, unit)

        metric_data: Optional[MetricData] = None
        if metric_value := self._get_tracked_metric_value(unit):
            metric_data = MetricData(
                name=none_throws(self._best_checkpoint_config).monitored_metric,
                value=metric_value,
            )

        checkpoint_path = self._checkpoint_manager.generate_checkpoint_path(
            epoch, step_mapping, metric_data
        )

        # 2) Determine if we should save checkpoint
        if not self._checkpoint_manager.should_save_checkpoint(checkpoint_path):
            return False

        if hook == "on_train_end":
            # 2.1) Make sure that last checkpoint does not already exist
            if self._does_checkpoint_exist(checkpoint_path, self._process_group):
                rank_zero_warn(
                    "Final checkpoint already exists, skipping.", logger=logger
                )
                return False

            # 2.2) If doing fit without eval checkpointing, only consider training progress when
            # checking if last checkpoint exists.
            if (
                state.entry_point == EntryPoint.FIT
                and self._save_every_n_eval_epochs is None
                and self._checkpoint_manager._ckpt_paths
                and self._checkpoint_manager._ckpt_paths[-1].step[Phase.TRAIN]
                == cast(TTrainUnit, unit).train_progress.num_steps_completed
            ):
                rank_zero_info(
                    "Omitting final checkpoint since train progress is unchanged, and eval checkpointing is not configured.",
                    logger=logger,
                )
                return False

        # 3) try to save checkpoint
        if not self._checkpoint_impl(
            state, unit, checkpoint_path=checkpoint_path.path, hook=hook
        ):
            return False

        # 4) track checkpoint and clean up surplus if needed
        self._checkpoint_manager.append_checkpoint(checkpoint_path)

        return True

    def _does_checkpoint_exist(
        self,
        checkpoint_path: CheckpointPath,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> bool:
        # Only keep this function as a hook for downstream checkpointer
        return self._checkpoint_manager.does_checkpoint_exist(
            checkpoint_path, process_group
        )

    def _get_tracked_metric_value(
        self, unit: Union[TTrainUnit, TEvalUnit]
    ) -> Optional[float]:
        """
        If the checkpointer has a tracked metric, look the value in the unit using reflection, and cast to float.

        Args:
            unit: The training unit to look for the tracked metric in.

        Returns:
            The value of the tracked metric, or None if there is no best_checkpoint config defined.

        Raises:
            RuntimeError: If the unit does not have the attribute specified in the best_checkpoint config,
                or if the value cannot be cast to a float.
        """
        if not self._best_checkpoint_config:
            return None

        monitored_metric_name = self._best_checkpoint_config.monitored_metric
        if not hasattr(unit, monitored_metric_name):
            raise RuntimeError(
                f"Unit does not have attribute {monitored_metric_name}, unable to retrieve metric to checkpoint."
            )

        metric_value_f = None
        if (metric_value := getattr(unit, monitored_metric_name)) is not None:
            try:
                metric_value_f = float(metric_value)
            except ValueError as e:
                raise RuntimeError(
                    f"Unable to convert monitored metric {monitored_metric_name} to a float. Please ensure the value "
                    "can be converted to float and is not a multi-element tensor value."
                ) from e

        return metric_value_f

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        # clean up the difference if surplus of checkpoints exist
        self._checkpoint_manager.prune_surplus_checkpoints()

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        num_steps_completed = unit.train_progress.num_steps_completed
        save_every_n_train_steps = self._save_every_n_train_steps
        if (
            save_every_n_train_steps is None
            or num_steps_completed % save_every_n_train_steps != 0
        ):
            return

        self._generate_checkpoint_and_upkeep(state, unit, hook="on_train_step_end")

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        epoch = unit.train_progress.num_epochs_completed
        save_every_n_epochs = self._save_every_n_epochs
        if save_every_n_epochs is None or epoch % save_every_n_epochs != 0:
            return

        self._generate_checkpoint_and_upkeep(state, unit, hook="on_train_epoch_end")

    def on_eval_epoch_end(self, state: State, unit: TEvalUnit) -> None:
        epoch = unit.eval_progress.num_epochs_completed
        save_every_n_eval_epochs = self._save_every_n_eval_epochs
        if save_every_n_eval_epochs is None or epoch % save_every_n_eval_epochs != 0:
            return

        self._generate_checkpoint_and_upkeep(state, unit, hook="on_eval_epoch_end")

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        self._generate_checkpoint_and_upkeep(state, unit, hook="on_train_end")

    @abc.abstractmethod
    def _checkpoint_impl(
        self,
        state: State,
        unit: AppStateMixin,
        *,
        checkpoint_path: str,
        hook: str,
    ) -> bool:
        """
        Implementation of saving checkpoint.

        Args:
            state: current application state
            unit: current unit
            checkpoint_path: path to save checkpoint
            hook: name of callback hook that triggered this function call

        Returns:
            Whether a new checkpoint was created.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def restore(
        path: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
    ) -> None:
        """Method to restore checkpoint state from a path.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            path: Path of the checkpoint to restore.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
            restore_options: Controls what to filter when restoring the state.
        """
        ...

    @classmethod
    def restore_from_latest(
        cls,
        dirpath: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Given a parent directory where checkpoints are saved, restore the checkpoint state from the latest checkpoint in the directory.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            dirpath: Parent directory from which to get the latest checkpoint.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            train_dataloader: An optional train dataloader to restore.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
            restore_options: Controls what to  filter when restoring the state.

        Returns:
            True if the latest checkpoint directory was found and successfully restored, otherwise False.
        """
        path = get_latest_checkpoint_path(
            dirpath, metadata_fname=cls.metadata_fname, process_group=process_group
        )
        if path is None:
            logger.info(
                f"Attempted to restore from the following path but no checkpoint was found: {dirpath=}, {cls.metadata_fname}"
            )
            return False
        logger.info(f"Restoring from path: {path}")
        cls.restore(
            path,
            unit,
            train_dataloader=train_dataloader,
            process_group=process_group,
            restore_options=restore_options,
            **kwargs,
        )
        return True

    @classmethod
    def restore_from_best(
        cls,
        dirpath: str,
        unit: AppStateMixin,
        metric_name: str,
        mode: Literal["min", "max"],
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Given a parent directory where checkpoints are saved, restore the checkpoint state from the best checkpoint in the directory.

        There are additional flags offered should the user want to skip loading the train and eval progress.
        By default, the train and eval progress are restored, if applicable.

        Args:
            dirpath: Parent directory from which to get the latest checkpoint.
            unit: An instance of :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit` containing states to restore.
            metric_name: Name of the metric to use to find the best checkpoint.
            mode: Either 'min' or 'max'. If 'min', finds and loads the lowest value metric checkpoint. If 'max', finds and loads the largest.
            train_dataloader: An optional train dataloader to restore.
            process_group: The process group on which the ranks will communicate on. default: ``None`` (the entire world)
            restore_options: Controls what to  filter when restoring the state.

        Returns:
            True if the best checkpoint directory was found and successfully restored, otherwise False.
        """
        best_checkpoint_path = get_best_checkpoint_path(
            dirpath,
            metric_name=metric_name,
            mode=mode,
            metadata_fname=cls.metadata_fname,
            process_group=process_group,
        )

        if best_checkpoint_path is None:
            rank_zero_warn(
                f"No checkpoints with metric name {metric_name} were found in {dirpath}. Not loading any checkpoint.",
                logger=logger,
            )
            return False

        rank_zero_info(f"Loading checkpoint from {best_checkpoint_path}")

        cls.restore(
            best_checkpoint_path,
            unit,
            train_dataloader=train_dataloader,
            process_group=process_group,
            restore_options=restore_options,
            **kwargs,
        )

        return True
