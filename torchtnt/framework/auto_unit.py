# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ignore errors due to `Any` type
# pyre-ignore-all-errors[2]
# pyre-ignore-all-errors[3]

import contextlib
from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple, TypeVar, Union

import torch
import torch.distributed as dist

from pyre_extensions import none_throws
from torch.cuda.amp import GradScaler
from torch.distributed import ProcessGroup

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.api import OptimStateDictConfig, StateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, SWALR
from torchtnt.framework.state import ActivePhase, State
from torchtnt.framework.unit import EvalUnit, PredictUnit, TPredictData, TrainUnit
from torchtnt.framework.utils import (
    _get_timing_context,
    _is_fsdp_module,
    get_current_progress,
    StatefulInt,
)
from torchtnt.utils import (
    init_from_env,
    is_torch_version_geq_1_12,
    TLRScheduler,
    transfer_batch_norm_stats,
    transfer_weights,
)
from torchtnt.utils.device import copy_data_to_device, record_data_in_stream
from torchtnt.utils.rank_zero_log import rank_zero_warn
from torchtnt.utils.version import is_torch_version_ge_1_13_1
from typing_extensions import Literal

TSWA_avg_fn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


@dataclass
class Strategy:
    """Dataclass representing the parallelization strategy for the AutoUnit"""

    pass


@dataclass
class DDPStrategy(Strategy):
    """
    Dataclass representing the `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ strategy.
    Includes params for registering `DDP communication hooks <https://pytorch.org/docs/stable/ddp_comm_hooks.html>`_ and `syncing batch norm <https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html>`_.
    """

    # DDP Constructor params
    output_device: Optional[Union[int, torch.device]] = None
    dim: int = 0
    broadcast_buffers: bool = True
    process_group: Optional[ProcessGroup] = None
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    check_reduction: bool = False
    gradient_as_bucket_view: bool = False
    static_graph: bool = False

    # DDP Comm Hook params
    comm_state: Optional[object] = None
    comm_hook: Optional[
        Callable[[object, dist.GradBucket], torch.futures.Future[torch.Tensor]]
    ] = None

    # SyncBatchNorm params
    sync_batchnorm: bool = True


@dataclass
class FSDPStrategy(Strategy):
    """Dataclass representing the `FullyShardedDataParallel <https://pytorch.org/docs/stable/fsdp.html>`_ strategy"""

    process_group: Optional[ProcessGroup] = None
    sharding_strategy: Optional[ShardingStrategy] = None
    cpu_offload: Optional[CPUOffload] = None
    auto_wrap_policy: Optional[Callable[[torch.nn.Module, bool, int], bool]] = None
    backward_prefetch: Optional[BackwardPrefetch] = None
    ignored_modules: Optional[Iterable[torch.nn.Module]] = None
    sync_module_states: bool = False
    forward_prefetch: bool = False
    limit_all_gathers: bool = False
    use_orig_params: bool = False

    # FSDP set_state_dict_type params: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type
    # for setting type of state dict for checkpointing
    state_dict_type: Optional[StateDictType] = None
    state_dict_config: Optional[StateDictConfig] = None
    optim_state_dict_config: Optional[OptimStateDictConfig] = None


@dataclass
class SWAParams:
    """
    Dataclass to store parameters for stochastic weight averaging.

    Args:
        epoch_start: number of epochs to wait for before starting SWA
        anneal_epochs: number of epochs to anneal the SWA Scheduler to the learning rate (lr)
        anneal_strategy: method for annealing, supports "linear" and "cos"
        lr: learning rate for SWA
        avg_fn: function to compute custom average of parameters
    """

    epoch_start: int
    anneal_epochs: int
    anneal_strategy: str = "linear"
    lr: float = 0.05
    avg_fn: Optional[TSWA_avg_fn] = None


@dataclass
class TorchDynamoParams:
    """
    Dataclass to store parameters for torchdynamo.

    Args:
        backend: a string backend name in `torch._dynamo.list_backends()`
    """

    backend: str


@dataclass
class ActivationCheckpointParams:
    """
    Dataclass to store parameters for activation checkpointing.

    Args:
        checkpoint_impl: type of checkpointing implementation to use
        check_fn: A lambda function which will be passed to each child submodule and return ``True`` or ``False`` depending on whether the submodule should be wrapped.
    """

    checkpoint_impl: CheckpointImpl
    check_fn: Optional[Callable[[torch.nn.Module], bool]]


# pyre-ignore: Invalid type parameters [24]
TSelf = TypeVar("TSelf", bound="AutoUnit")
TData = TypeVar("TData")


class _ConfigureOptimizersCaller(ABCMeta):
    def __call__(self, *args, **kwargs):
        x = super().__call__(*args, **kwargs)
        x.optimizer = None
        x.lr_scheduler = None
        x.swa_model = None
        x.swa_scheduler = None

        if x.training:
            x.optimizer, x.lr_scheduler = x.configure_optimizers_and_lr_scheduler(
                x.module
            )

            if x.swa_params:
                if not x.swa_params.avg_fn:
                    # pyre-ignore: Unexpected keyword [28]
                    x.swa_model = AveragedModel(x.module, use_buffers=True)
                else:
                    # pyre-ignore: Unexpected keyword [28]
                    x.swa_model = AveragedModel(
                        x.module, avg_fn=x.swa_params.avg_fn, use_buffers=True
                    )

                x.swa_scheduler = SWALR(
                    optimizer=x.optimizer,
                    swa_lr=x.swa_params.lr,
                    anneal_epochs=x.swa_params.anneal_epochs,
                    anneal_strategy=x.swa_params.anneal_strategy,
                )

        return x


class AutoPredictUnit(PredictUnit[TPredictData]):
    def __init__(
        self,
        *,
        module: torch.nn.Module,
        device: Optional[torch.device] = None,
        strategy: Optional[Union[Strategy, str]] = None,
        precision: Optional[Union[str, torch.dtype]] = None,
        torchdynamo_params: Optional[TorchDynamoParams] = None,
    ) -> None:
        """
        AutoPredictUnit is a convenience for users who are running inference and would like to have certain features handled for them, such as:
        - Moving data to the correct device.
        - Running inference under a mixed precision context.
        - Handling data parallel replication, especially if the module cannot fit on a single device using FullyShardedDataParallel.
        - Profiling the data transfer to device and forward pass.
        - Interleaving moving the next batch to the device with running the module's forward pass on the current batch.

        Additionally, the AutoPredictUnit offers an optional hook ``on_predict_step_end`` to further post-process module outputs if desired.

        Then use with the :py:func:`~torchtnt.framework.predict` entry point.

        For more advanced customization, directly use the :class:`~torchtnt.framework.unit.PredictUnit` interface.

        Args:
            module: module to be used during prediction.
            device: the device to be used.
            precision: the precision to use in training, as either a string or a torch.dtype.
            strategy: the data parallelization strategy to be used. if a string, must be one of ``ddp`` or ``fsdp``.
            torchdynamo_params: params for TorchDynamo https://pytorch.org/docs/stable/dynamo/index.html

        Note:
            TorchDynamo support is only available in PyTorch 2.0 or higher.
        """
        if torchdynamo_params:
            _validate_torchdynamo_available()

        super().__init__()

        self.device: torch.device = device or init_from_env()
        self.precision: Optional[torch.dtype]
        if isinstance(precision, str):
            self.precision = _convert_precision_str_to_dtype(precision)
        else:
            self.precision = precision

        # create autocast context based on precision and device type
        self.maybe_autocast_precision = torch.autocast(
            device_type=self.device.type,
            dtype=self.precision,
            enabled=self.precision is not None,
        )
        if strategy:
            if isinstance(strategy, str):
                strategy = _convert_str_to_strategy(strategy)
            if isinstance(strategy, DDPStrategy):
                module = _prepare_ddp(module, strategy, self.device, torchdynamo_params)
            elif isinstance(strategy, FSDPStrategy):
                module = _prepare_fsdp(
                    module,
                    strategy,
                    self.device,
                    None,  # SWA params
                    self.precision,
                )
        else:
            module = module.to(self.device)
        if torchdynamo_params:
            module = _dynamo_wrapper(module, torchdynamo_params)
        self.module: torch.nn.Module = module

        # cuda stream to use for moving data to device
        self._prefetch_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.Stream() if self.device.type == "cuda" else None
        )
        # the next batch which has been prefetched and is ready to be used
        self._next_batch: Optional[TPredictData] = None

        # whether the next batch has been prefetched and is ready to be used
        self._prefetched: bool = False

    def predict_step(self, state: State, data: Iterator[TPredictData]) -> Any:
        batch = self._get_next_batch(state, data)

        with self.maybe_autocast_precision:
            with _get_timing_context(state, f"{self.__class__.__name__}.forward"):
                outputs = self.module(batch)

        step = get_current_progress(state).num_steps_completed
        with _get_timing_context(
            state, f"{self.__class__.__name__}.on_predict_step_end"
        ):
            self.on_predict_step_end(state, batch, step, outputs)
        return outputs

    def on_predict_step_end(
        self, state: State, data: TPredictData, step: int, outputs: Any
    ) -> None:
        """
        This will be called at the end of every ``predict_step`` before returning. The user can implement this method with code to update and log their metrics,
        or do anything else.

        Args:
            state: a State object which is passed from the ``predict_step``
            data: a batch of data which is passed from the ``predict_step``
            step: how many ``predict_step``s have been completed
            outputs: the outputs of the model forward pass
        """
        pass

    def move_data_to_device(
        self, state: State, data: TPredictData, non_blocking: bool
    ) -> TPredictData:
        """
        The user can override this method with custom code to copy data to device. This will be called at the start of every ``predict_step``.
        By default this uses the utility function :py:func:`~torchtnt.utils.copy_data_to_device`.

        If on GPU, this method will be called on a separate CUDA stream.

        Args:
            state: a State object which is passed from the ``predict_step``
            data: a batch of data which is passed from the ``predict_step``
            non_blocking: parameter to pass to ``torch.tensor.to``

        Returns:
            A batch of data which is on the device
        """
        return copy_data_to_device(data, self.device, non_blocking=non_blocking)

    def _get_next_batch(
        self, state: State, data: Iterator[TPredictData]
    ) -> TPredictData:
        if not self._prefetched:
            self._prefetch_next_batch(state, data)
            self._prefetched = True

        if self._prefetch_stream:
            with _get_timing_context(state, f"{self.__class__.__name__}.wait_stream"):
                # wait on the CUDA stream to complete the host to device copy
                torch.cuda.current_stream().wait_stream(self._prefetch_stream)

        # get the next batch which was stored by _prefetch_next_batch
        batch = self._next_batch
        if batch is None:
            self._prefetched = False
            raise StopIteration

        if self._prefetch_stream:
            with _get_timing_context(
                state, f"{self.__class__.__name__}.record_data_in_stream"
            ):
                # record the batch in the current stream
                record_data_in_stream(batch, torch.cuda.current_stream())

        # kick off prefetching the next batch
        self._prefetch_next_batch(state, data)
        return batch

    def _prefetch_next_batch(
        self, state: State, data_iter: Iterator[TPredictData]
    ) -> None:
        """Prefetch the next batch on a separate CUDA stream."""

        try:
            with _get_timing_context(
                state, f"{self.__class__.__name__}.next(data_iter)"
            ):
                next_batch = next(data_iter)
        except StopIteration:
            self._next_batch = None
            return

        non_blocking = (
            True if self.device.type == "cuda" and self._prefetched else False
        )

        # if on cpu, self._prefetch_stream is None so the torch.cuda.stream call is a no-op
        with torch.cuda.stream(self._prefetch_stream), _get_timing_context(
            state, f"{self.__class__.__name__}.move_data_to_device"
        ):
            self._next_batch = self.move_data_to_device(
                state, next_batch, non_blocking=non_blocking
            )


class AutoUnit(
    TrainUnit[TData],
    EvalUnit[TData],
    PredictUnit[Any],
    metaclass=_ConfigureOptimizersCaller,
):
    """
    The AutoUnit is a convenience for users who are training with stochastic gradient descent and would like to have model optimization
    and data parallel replication handled for them.
    The AutoUnit subclasses :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, and :class:`~torchtnt.framework.unit.PredictUnit`,
    and implements the ``train_step``, ``eval_step``, and ``predict_step`` methods for the user.

    For the ``train_step`` it runs:

    - forward pass and loss computation
    - backward pass
    - optimizer step

    For the ``eval_step`` it only runs forward and loss computation.

    For the ``predict_step`` it only runs forward.

    To benefit from the AutoUnit, the user must subclass it and implement the ``compute_loss`` and ``configure_optimizers_and_lr_scheduler`` methods.
    Additionally, the AutoUnit offers these optional hooks:

    - ``on_train_step_end``
    - ``on_eval_step_end``
    - ``on_predict_step_end``

    Then use with the :py:func:`~torchtnt.framework.train`, :py:func:`~torchtnt.framework.evaluate`, :py:func:`~torchtnt.framework.predict`, or :py:func:`~torchtnt.framework.fit` entry point as normal.

    For more advanced customization, directly use the :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, and :class:`~torchtnt.framework.unit.PredictUnit` interfaces.

    Args:
        module: module to be used during training.
        device: the device to be used.
        strategy: the data parallelization strategy to be used. if a string, must be one of ``ddp`` or ``fsdp``.
        step_lr_interval: whether to step lr_scheduler every step or every epoch. Defaults to every epoch.
        precision: the precision to use in training, as either a string or a torch.dtype.
        gradient_accumulation_steps: how many batches to accumulate gradients over.
        detect_anomaly: whether to enable anomaly detection for the autograd engine https://pytorch.org/docs/stable/autograd.html#anomaly-detection
        clip_grad_norm: max norm of the gradients for clipping https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
        clip_grad_value: max value of the gradients for clipping https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html
        swa_params: params for stochastic weight averaging https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging
        torchdynamo_params: params for TorchDynamo https://pytorch.org/docs/stable/dynamo/index.html
        activation_checkpoint_params: params for enabling activation checkpointing
        training: if True, the optimizer and optionally LR scheduler will be created after the class is initialized.

    Note:
        Stochastic Weight Averaging is currently not supported with the FSDP strategy.

    Note:
        TorchDynamo support is only available in PyTorch 2.0 or higher.

    """

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        device: Optional[torch.device] = None,
        strategy: Optional[Union[Strategy, str]] = None,
        step_lr_interval: Literal["step", "epoch"] = "epoch",
        precision: Optional[Union[str, torch.dtype]] = None,
        gradient_accumulation_steps: int = 1,
        detect_anomaly: Optional[bool] = None,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
        swa_params: Optional[SWAParams] = None,
        torchdynamo_params: Optional[TorchDynamoParams] = None,
        activation_checkpoint_params: Optional[ActivationCheckpointParams] = None,
        training: bool = True,
    ) -> None:
        super().__init__()

        if not gradient_accumulation_steps > 0:
            raise ValueError(
                f"gradient_accumulation_steps must be > 0. Got {gradient_accumulation_steps}"
            )
        if torchdynamo_params:
            _validate_torchdynamo_available()

        self.device: torch.device = device or init_from_env()
        self.precision: Optional[torch.dtype]
        if isinstance(precision, str):
            self.precision = _convert_precision_str_to_dtype(precision)
        else:
            self.precision = precision

        if strategy:
            if isinstance(strategy, str):
                strategy = _convert_str_to_strategy(strategy)
            if isinstance(strategy, DDPStrategy):
                module = _prepare_ddp(module, strategy, self.device, torchdynamo_params)
            elif isinstance(strategy, FSDPStrategy):
                module = _prepare_fsdp(
                    module, strategy, self.device, swa_params, self.precision
                )
        else:
            module = module.to(self.device)

        if activation_checkpoint_params:
            checkpoint_impl = activation_checkpoint_params.checkpoint_impl
            check_fn = activation_checkpoint_params.check_fn
            custom_checkpoint_wrapper = partial(
                checkpoint_wrapper,
                checkpoint_impl=checkpoint_impl,
            )
            apply_activation_checkpointing(
                module,
                checkpoint_wrapper_fn=custom_checkpoint_wrapper,
                check_fn=check_fn,
            )

        self.module: torch.nn.Module = module

        self.step_lr_interval = step_lr_interval

        self.grad_scaler: Optional[GradScaler] = None
        if self.precision:
            self.grad_scaler = _get_grad_scaler_from_precision(
                self.precision,
                self.module,
            )

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._num_optimizer_steps_completed: StatefulInt = StatefulInt(0)

        self.detect_anomaly = detect_anomaly
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        # create autocast context based on precision and device type
        self.maybe_autocast_precision = torch.autocast(
            device_type=self.device.type,
            dtype=self.precision,
            enabled=self.precision is not None,
        )

        self.swa_params: Optional[SWAParams] = swa_params

        if torchdynamo_params:
            # pyre-ignore
            self.compute_loss = _dynamo_wrapper(self.compute_loss, torchdynamo_params)
            self.module = _dynamo_wrapper(self.module, torchdynamo_params)

        self.training = training

        # cuda stream to use for moving data to device
        self._prefetch_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.Stream() if self.device.type == "cuda" else None
        )
        # the next batch which has been prefetched and is ready to be used
        self._next_batch: Optional[TData] = None
        # whether the next batch has been prefetched and is ready to be used
        self._prefetched: bool = False
        # whether the current batch is the last train batch
        self._is_last_train_batch: bool = False

    @abstractmethod
    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, Optional[TLRScheduler]]:
        """
        The user should implement this method with their optimizer and learning rate scheduler construction code. This will be called upon initialization of
        the AutoUnit.

        Args:
            module: the module with which to construct optimizer and lr_scheduler

        Returns:
            A tuple containing optimizer and optionally the learning rate scheduler
        """
        ...

    @abstractmethod
    def compute_loss(self, state: State, data: TData) -> Tuple[torch.Tensor, Any]:
        """
        The user should implement this method with their loss computation. This will be called every ``train_step``/``eval_step``.

        Args:
            state: a State object which is passed from the ``train_step``/``eval_step``
            data: a batch of data which is passed from the ``train_step``/``eval_step``

        Returns:
            Tuple containing the loss and the output of the model

        Note:
            The module's forward pass must be run as part of this method.
        """
        ...

    def move_data_to_device(
        self, state: State, data: TData, non_blocking: bool
    ) -> TData:
        """
        The user can override this method with custom code to copy data to device. This will be called at the start of every ``train_step``/``eval_step``/``predict_step``.
        By default this uses the utility function :py:func:`~torchtnt.utils.copy_data_to_device`.

        If on GPU, this method will be called on a separate CUDA stream.

        Args:
            state: a State object which is passed from the ``train_step``/``eval_step``/``predict_step``
            data: a batch of data which is passed from the ``train_step``/``eval_step``/``predict_step``
            non_blocking: parameter to pass to ``torch.tensor.to``

        Returns:
            A batch of data which is on the device
        """
        return copy_data_to_device(data, self.device, non_blocking=non_blocking)

    def _prefetch_next_batch(self, state: State, data_iter: Iterator[TData]) -> None:
        """Prefetch the next batch on a separate CUDA stream."""

        phase = state.active_phase.name.lower()
        try:
            with _get_timing_context(
                state, f"{self.__class__.__name__}.{phase}.next(data_iter)"
            ):
                next_batch = next(data_iter)
        except StopIteration:
            self._next_batch = None
            self._is_last_train_batch = True
            return

        non_blocking = (
            True
            if state.active_phase == ActivePhase.TRAIN
            and self.device.type == "cuda"
            and self._prefetched
            else False
        )

        # if on cpu, self._prefetch_stream is None so the torch.cuda.stream call is a no-op
        with torch.cuda.stream(self._prefetch_stream), _get_timing_context(
            state, f"{self.__class__.__name__}.{phase}.move_data_to_device"
        ):
            self._next_batch = self.move_data_to_device(
                state, next_batch, non_blocking=non_blocking
            )

    def _get_next_batch(self, state: State, data: Iterator[TData]) -> TData:
        if not self._prefetched:
            self._prefetch_next_batch(state, data)
            self._prefetched = True

        if self._prefetch_stream:
            with _get_timing_context(state, f"{self.__class__.__name__}.wait_stream"):
                # wait on the CUDA stream to complete the host to device copy
                torch.cuda.current_stream().wait_stream(self._prefetch_stream)

        # get the next batch which was stored by _prefetch_next_batch
        batch = self._next_batch
        if batch is None:
            self._prefetched = False
            self._is_last_train_batch = False
            raise StopIteration

        if self._prefetch_stream:
            with _get_timing_context(
                state, f"{self.__class__.__name__}.record_data_in_stream"
            ):
                # record the batch in the current stream
                record_data_in_stream(batch, torch.cuda.current_stream())

        # prefetch the next batch
        self._prefetch_next_batch(state, data)

        return batch

    def train_step(
        self, state: State, data: Iterator[TData]
    ) -> Tuple[torch.Tensor, Any]:
        train_state = none_throws(state.train_state)

        batch = self._get_next_batch(state, data)

        should_update_weights = (
            train_state.progress.num_steps_completed_in_epoch + 1
        ) % self.gradient_accumulation_steps == 0 or self._is_last_train_batch

        # for pyre, assign to local variable
        module = self.module

        # if using gradient accumulation with either DDP or FSDP, when in a step where we will not update the weights,
        # run forward and backward in no_sync context
        # https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync
        # https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync
        maybe_no_sync = (
            # pyre-ignore[29]
            module.no_sync()
            if not should_update_weights
            and (isinstance(module, DDP) or _is_fsdp_module(module))
            else contextlib.nullcontext()
        )

        # if detect_anomaly is true, run forward and backward pass in detect_anomaly context
        detect_anomaly = self.detect_anomaly
        maybe_detect_anomaly = (
            torch.autograd.set_detect_anomaly(detect_anomaly)
            if detect_anomaly is not None
            else contextlib.nullcontext()
        )

        grad_scaler = self.grad_scaler
        with maybe_no_sync, maybe_detect_anomaly:
            with self.maybe_autocast_precision:
                with _get_timing_context(
                    state, f"{self.__class__.__name__}.compute_loss"
                ):
                    # Run the forward pass and compute the loss
                    loss, outputs = self.compute_loss(state, batch)

            # normalize loss to account for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            if grad_scaler:
                scaled_loss = grad_scaler.scale(loss)
                with _get_timing_context(state, f"{self.__class__.__name__}.backward"):
                    scaled_loss.backward()
            else:
                with _get_timing_context(state, f"{self.__class__.__name__}.backward"):
                    loss.backward()

        if should_update_weights:
            # Run gradient clipping, optimizer step, and zero_grad
            # TODO try to use dynamo here
            clip_grad_norm = self.clip_grad_norm
            clip_grad_value = self.clip_grad_value
            if grad_scaler and (clip_grad_norm or clip_grad_value):
                # unscale the gradients of optimizer's assigned params in-place in preparation for gradient clipping
                with _get_timing_context(
                    state, f"{self.__class__.__name__}.grad_unscale"
                ):
                    grad_scaler.unscale_(self.optimizer)

            # gradient norm clipping
            if clip_grad_norm:
                if _is_fsdp_module(module):
                    if isinstance(module, FSDP):
                        with _get_timing_context(
                            state, f"{self.__class__.__name__}.clip_grad_norm"
                        ):
                            module.clip_grad_norm_(max_norm=clip_grad_norm)
                    else:
                        raise RuntimeError(
                            "Composable FSDP clip_grad_norm is not yet implemented: https://github.com/pytorch/pytorch/issues/97271"
                        )
                else:
                    with _get_timing_context(
                        state, f"{self.__class__.__name__}.clip_grad_norm"
                    ):
                        torch.nn.utils.clip_grad_norm_(
                            parameters=module.parameters(),
                            max_norm=clip_grad_norm,
                        )

            # gradient value clipping
            if clip_grad_value:
                with _get_timing_context(
                    state, f"{self.__class__.__name__}.clip_grad_value"
                ):
                    torch.nn.utils.clip_grad_value_(
                        parameters=module.parameters(),
                        clip_value=clip_grad_value,
                    )

            with _get_timing_context(
                state, f"{self.__class__.__name__}.optimizer_step"
            ):
                if grad_scaler:
                    grad_scaler.step(self.optimizer)
                    # update the scale for next iteration
                    grad_scaler.update()
                else:
                    self.optimizer.step()

            self._num_optimizer_steps_completed += 1

            # sets gradients to zero
            with _get_timing_context(
                state, f"{self.__class__.__name__}.optimizer_zero_grad"
            ):
                self.optimizer.zero_grad(set_to_none=True)

            # optionally step lr scheduler if SWA not in use
            train_state = none_throws(state.train_state)
            if (
                self.swa_params is None
                or train_state.progress.num_epochs_completed
                < self.swa_params.epoch_start
            ):
                lr_scheduler = self.lr_scheduler
                if lr_scheduler and self.step_lr_interval == "step":
                    with _get_timing_context(
                        state, f"{self.__class__.__name__}.lr_scheduler_step"
                    ):
                        lr_scheduler.step()

        step = get_current_progress(state).num_steps_completed
        # users can override this, by default this is a no-op
        with _get_timing_context(state, f"{self.__class__.__name__}.on_train_step_end"):
            self.on_train_step_end(state, batch, step, loss, outputs)
        return loss, outputs

    def on_train_step_end(
        self, state: State, data: TData, step: int, loss: torch.Tensor, outputs: Any
    ) -> None:
        """
        This will be called at the end of every ``train_step`` before returning. The user can implement this method with code to update and log their metrics,
        or do anything else.

        Args:
            state: a State object which is passed from the ``train_step``
            data: a batch of data which is passed from the ``train_step``
            step: how many ``train_step``s have been completed
            loss: the loss computed in the ``compute_loss`` function
            outputs: the outputs of the model forward pass
        """
        pass

    def on_train_epoch_end(self, state: State) -> None:
        """
        Note: if overriding ``on_train_epoch_end``, remember to call ``super().on_train_epoch_end()``
        """
        train_state = none_throws(state.train_state)

        if (
            self.swa_model
            and self.swa_params
            and train_state.progress.num_epochs_completed >= self.swa_params.epoch_start
        ):
            with _get_timing_context(
                state, f"{self.__class__.__name__}.stochastic_weight_avg_update"
            ):
                self.swa_model.update_parameters(self.module)
            with _get_timing_context(
                state, f"{self.__class__.__name__}.stochastic_weight_avg_step"
            ):
                none_throws(self.swa_scheduler).step()
        elif self.lr_scheduler and self.step_lr_interval == "epoch":
            # optionally step lr scheduler
            with _get_timing_context(
                state, f"{self.__class__.__name__}.lr_scheduler_step"
            ):
                self.lr_scheduler.step()

    def on_train_end(self, state: State) -> None:
        """
        Note that if using SWA and implementing `on_train_end()`, must call `super().on_train_end()`.
        """
        swa_model = self.swa_model
        if swa_model:
            with _get_timing_context(
                state,
                f"{self.__class__.__name__}.stochastic_weight_avg_transfer_weights",
            ):
                transfer_weights(swa_model, self.module)
            with _get_timing_context(
                state,
                f"{self.__class__.__name__}.stochastic_weight_avg_transfer_batch_norm_stats",
            ):
                transfer_batch_norm_stats(swa_model, self.module)

    def eval_step(self, state: State, data: TData) -> Tuple[torch.Tensor, Any]:
        with _get_timing_context(
            state, f"{self.__class__.__name__}.move_data_to_device"
        ):
            data = self.move_data_to_device(state, data, non_blocking=False)

        with self.maybe_autocast_precision:
            # users must override this
            with _get_timing_context(state, f"{self.__class__.__name__}.compute_loss"):
                loss, outputs = self.compute_loss(state, data)

        step = get_current_progress(state).num_steps_completed
        # users can override this, by default this is a no-op
        with _get_timing_context(state, f"{self.__class__.__name__}.on_eval_step_end"):
            self.on_eval_step_end(state, data, step, loss, outputs)
        return loss, outputs

    def on_eval_step_end(
        self, state: State, data: TData, step: int, loss: torch.Tensor, outputs: Any
    ) -> None:
        """
        This will be called at the end of every ``eval_step`` before returning. The user can implement this method with code to update and log their metrics,
        or do anything else.

        Args:
            state: a State object which is passed from the ``eval_step``
            data: a batch of data which is passed from the ``eval_step``
            step: how many steps have been completed (``train_step``s when running fit and ``eval_step``s when running evaluation)
            loss: the loss computed in the ``compute_loss`` function
            outputs: the outputs of the model forward pass
        """
        pass

    def predict_step(self, state: State, data: Any) -> Any:
        with _get_timing_context(
            state, f"{self.__class__.__name__}.move_data_to_device"
        ):
            data = self.move_data_to_device(state, data, non_blocking=False)

        with self.maybe_autocast_precision:
            with _get_timing_context(state, f"{self.__class__.__name__}.forward"):
                outputs = self.module(data)

        step = get_current_progress(state).num_steps_completed
        # users can override this, by default this is a no-op
        with _get_timing_context(
            state, f"{self.__class__.__name__}.on_predict_step_end"
        ):
            self.on_predict_step_end(state, data, step, outputs)
        return outputs

    def on_predict_step_end(
        self, state: State, data: TData, step: int, outputs: Any
    ) -> None:
        """
        This will be called at the end of every ``predict_step`` before returning. The user can implement this method with code to update and log their metrics,
        or do anything else.

        Args:
            state: a State object which is passed from the ``predict_step``
            data: a batch of data which is passed from the ``predict_step``
            step: how many ``predict_step``s have been completed
            outputs: the outputs of the model forward pass
        """
        pass

    @property
    def num_optimizer_steps_completed(self) -> int:
        return self._num_optimizer_steps_completed.val


def _validate_torchdynamo_available() -> None:
    if not is_torch_version_ge_1_13_1():
        raise RuntimeError(
            "TorchDynamo support is available only in PyTorch 2.0 or higher. "
            "Please install PyTorch 2.0 or higher to continue: https://pytorch.org/get-started/locally/"
        )


def _convert_precision_str_to_dtype(precision: str) -> Optional[torch.dtype]:
    """
    Converts precision as a string to a torch.dtype

    Args:
        precision: string containing the precision

    Raises:
        ValueError if an invalid precision string is passed.

    """
    string_to_dtype_mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": None,
    }
    if precision not in string_to_dtype_mapping.keys():
        raise ValueError(
            f"Precision {precision} not supported. Please use one of {list(string_to_dtype_mapping.keys())}"
        )
    return string_to_dtype_mapping[precision]


def _convert_str_to_strategy(strategy: str) -> Union[DDPStrategy, FSDPStrategy]:
    """
    Converts strategy as a string to a default instance of the Strategy dataclass.

    Args:
        strategy: string specifying the distributed strategy to use

    Raises:
        ValueError if an invalid strategy string is passed.

    """
    string_to_strategy_mapping = {
        "ddp": DDPStrategy(),
        "fsdp": FSDPStrategy(),
    }

    if strategy not in string_to_strategy_mapping:
        raise ValueError(
            f"Strategy {strategy} not supported. Please use one of {list(string_to_strategy_mapping.keys())}"
        )
    return string_to_strategy_mapping[strategy]


def _get_grad_scaler_from_precision(
    precision: torch.dtype, module: torch.nn.Module
) -> Optional[GradScaler]:
    if precision == torch.float16:
        if _is_fsdp_module(module):
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            return ShardedGradScaler()
        else:
            return GradScaler()
    return None


# pyre-ignore
def _dynamo_wrapper(fn: Callable, torchdynamo_params: TorchDynamoParams):
    backend = torchdynamo_params.backend
    try:
        return torch.compile(fn, backend=backend)
    except KeyError as e:
        raise RuntimeError(
            f"Torchdynamo backend {torchdynamo_params.backend} is not supported."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"The following error encountered when calling torch.compile for dynamo: {e}"
        ) from e


def _prepare_ddp(
    module: torch.nn.Module,
    strategy: DDPStrategy,
    device: torch.device,
    torchdynamo_params: Optional[TorchDynamoParams],
) -> DDP:
    # wrap module in DDP
    device_ids = None
    if device.type == "cuda":
        device_ids = [device.index]
    params_dict = asdict(strategy)
    # remove ddp comm hook variables from params dict
    del params_dict["comm_state"]
    del params_dict["comm_hook"]
    module = module.to(device)

    # remove sync batch norm from params dict before converting module
    del params_dict["sync_batchnorm"]
    if strategy.sync_batchnorm:
        if device.type == "cuda":
            module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        else:
            rank_zero_warn(
                f"SyncBatchNorm layers only work with GPU modules. Skipping the conversion because the device type is {device.type}."
            )

    module = DDP(module, device_ids=device_ids, **params_dict)
    if torchdynamo_params:
        # TODO: Add support for dynamo and DDP
        rank_zero_warn(
            "Torchdynamo params has been set with DDP - Note that performance will likely be slower and we recommend using only one."
        )
    if strategy.comm_hook:
        module.register_comm_hook(state=strategy.comm_state, hook=strategy.comm_hook)
    return module


def _prepare_fsdp(
    module: torch.nn.Module,
    strategy: FSDPStrategy,
    device: torch.device,
    swa_params: Optional[SWAParams],
    precision: Optional[torch.dtype],
) -> FSDP:
    if not is_torch_version_geq_1_12():
        raise RuntimeError(
            "Please install PyTorch 1.12 or higher to use FSDP: https://pytorch.org/get-started/locally/"
        )
    elif swa_params:
        raise RuntimeError(
            "Stochastic Weight Averaging is currently not supported with the FSDP strategy"
        )
    mixed_precision = None
    if precision:
        mixed_precision = MixedPrecision(
            param_dtype=precision,
            reduce_dtype=precision,
            buffer_dtype=precision,
        )

    params_dict = asdict(strategy)

    # extract params to set state dict type
    state_dict_type = params_dict.pop("state_dict_type")
    state_dict_config = params_dict.pop("state_dict_config")
    optim_state_dict_config = params_dict.pop("optim_state_dict_config")

    # wrap module in FSDP
    module = FSDP(
        module,
        device_id=device,
        mixed_precision=mixed_precision,
        **params_dict,
    )

    if state_dict_type:
        FSDP.set_state_dict_type(
            module, state_dict_type, state_dict_config, optim_state_dict_config
        )
    return module
