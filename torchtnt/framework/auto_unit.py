# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import contextlib
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
from pyre_extensions import none_throws
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import SWALR
from torchtnt.framework.state import ActivePhase, EntryPoint, State
from torchtnt.framework.unit import EvalUnit, PredictUnit, TPredictData, TrainUnit
from torchtnt.framework.utils import _step_requires_iterator, get_timing_context
from torchtnt.utils.device import copy_data_to_device, record_data_in_stream
from torchtnt.utils.env import init_from_env
from torchtnt.utils.lr_scheduler import TLRScheduler
from torchtnt.utils.precision import convert_precision_str_to_dtype
from torchtnt.utils.prepare_module import (
    _is_fsdp_module,
    ActivationCheckpointParams,
    FSDPStrategy,
    prepare_fsdp,
    prepare_module,
    Strategy,
    TorchCompileParams,
)
from torchtnt.utils.swa import AveragedModel
from torchtnt.utils.version import is_torch_version_ge_1_13_1
from typing_extensions import Literal


TData = TypeVar("TData")

TSWA_avg_fn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
# pyre-ignore Invalid type parameters [24]: Generic type `Callable` expects 2 type parameters.
TSWA_multi_avg_fn = Callable


@dataclass
class SWALRParams:
    """
    Dataclass to store parameters for SWALR learning rate scheduler.
    See https://github.com/pytorch/pytorch/blob/main/torch/optim/swa_utils.py#L279
    for more details.

    Args:
        anneal_steps_or_epochs: number of steps or epochs to anneal the SWA Scheduler
        anneal_strategy: method for annealing, supports "linear" and "cos"
        swa_lrs: the learning rate value for all param groups together or separately for each group

        Note: Whether steps or epochs is used based on what `step_lr_interval` is set on the AutoUnit.
    """

    anneal_steps_or_epochs: int
    anneal_strategy: str = "linear"
    swa_lrs: Union[List[float], float] = 0.05


@dataclass
class SWAParams:
    """
    Dataclass to store parameters for stochastic weight averaging.

    Args:
        warmup_steps_or_epochs: number of steps or epochs before starting SWA
        step_or_epoch_update_freq: number of steps or epochs between each SWA update
        avg_fn: function to compute custom average of parameters
        multi_avg_fn: function used to update parameters inplace
        swalr_params: params for SWA learning rate scheduler

        Note: Whether steps or epochs is used based on what `step_lr_interval` is set on the AutoUnit.

        Note: Only one of avg_fn, multi_avg_fn should be specified

    """

    warmup_steps_or_epochs: int
    step_or_epoch_update_freq: int
    avg_fn: Optional[TSWA_avg_fn] = None
    multi_avg_fn: Optional[TSWA_multi_avg_fn] = None
    swalr_params: Optional[SWALRParams] = None


class _ConfigureOptimizersCaller(ABCMeta):
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __call__(self, *args, **kwargs):
        x = super().__call__(*args, **kwargs)
        x.optimizer = None
        x.lr_scheduler = None
        x.swa_scheduler = None

        if x.training:
            x.optimizer, x.lr_scheduler = x.configure_optimizers_and_lr_scheduler(
                x.module
            )

            if x.swa_params and x.swa_params.swalr_params:
                swalr_params = x.swa_params.swalr_params
                x.swa_scheduler = SWALR(
                    optimizer=x.optimizer,
                    swa_lr=swalr_params.swa_lrs,
                    anneal_epochs=swalr_params.anneal_steps_or_epochs,
                    anneal_strategy=swalr_params.anneal_strategy,
                )

        return x


class _AutoUnitMixin(Generic[TData]):
    """
    A mixin to share initialization of shared attributes and introduce prefetching.
    """

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        device: Optional[torch.device] = None,
        precision: Optional[Union[str, torch.dtype]] = None,
        detect_anomaly: Optional[bool] = None,
        torch_compile_params: Optional[TorchCompileParams] = None,
    ) -> None:
        super().__init__()
        if torch_compile_params:
            _validate_torch_compile_available()

        self.device: torch.device = device or init_from_env()
        self.precision: Optional[torch.dtype] = (
            convert_precision_str_to_dtype(precision)
            if isinstance(precision, str)
            else precision
        )

        self.detect_anomaly = detect_anomaly

        # create autocast context based on precision and device type
        self.maybe_autocast_precision = torch.autocast(
            device_type=self.device.type,
            dtype=self.precision,
            enabled=self.precision is not None,
        )

        # cuda stream to use for moving data to device
        self._prefetch_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.Stream() if self.device.type == "cuda" else None
        )
        # dict mapping phase to whether the next batch which has been prefetched for that phase and is ready to be used
        self._phase_to_next_batch: dict[ActivePhase, Optional[TData]] = {
            ActivePhase.TRAIN: None,
            ActivePhase.EVALUATE: None,
            ActivePhase.PREDICT: None,
        }

        # dict mapping phase to whether the next batch for that phase has been prefetched and is ready to be used
        self._phase_to_prefetched: dict[ActivePhase, bool] = {
            ActivePhase.TRAIN: False,
            ActivePhase.EVALUATE: False,
            ActivePhase.PREDICT: False,
        }
        # whether the current batch is the last train batch
        self._is_last_batch: bool = False

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
        active_phase = state.active_phase
        phase = state.active_phase.name.lower()
        try:
            with get_timing_context(
                state, f"{self.__class__.__name__}.{phase}.next(data_iter)"
            ):
                next_batch = next(data_iter)
        except StopIteration:
            self._phase_to_next_batch[active_phase] = None
            self._is_last_batch = True
            return

        non_blocking = bool(
            self.device.type == "cuda" and self._phase_to_prefetched[active_phase]
        )

        # if on cpu, self._prefetch_stream is None so the torch.cuda.stream call is a no-op
        with torch.cuda.stream(self._prefetch_stream), get_timing_context(
            state, f"{self.__class__.__name__}.{phase}.move_data_to_device"
        ):
            self._phase_to_next_batch[active_phase] = self.move_data_to_device(
                state, next_batch, non_blocking=non_blocking
            )

    def _get_next_batch(self, state: State, data: Iterator[TData]) -> TData:
        active_phase = state.active_phase
        if not self._phase_to_prefetched[active_phase]:
            self._prefetch_next_batch(state, data)
            self._phase_to_prefetched[active_phase] = True

        if self._prefetch_stream:
            with get_timing_context(state, f"{self.__class__.__name__}.wait_stream"):
                # wait on the CUDA stream to complete the host to device copy
                torch.cuda.current_stream().wait_stream(self._prefetch_stream)

        # get the next batch which was stored by _prefetch_next_batch
        batch = self._phase_to_next_batch[active_phase]
        if batch is None:
            self._phase_to_prefetched[active_phase] = False
            self._is_last_batch = False
            raise StopIteration

        if self._prefetch_stream:
            with get_timing_context(
                state, f"{self.__class__.__name__}.record_data_in_stream"
            ):
                # record the batch in the current stream
                record_data_in_stream(batch, torch.cuda.current_stream())

        # prefetch the next batch
        self._prefetch_next_batch(state, data)

        return batch


class AutoPredictUnit(_AutoUnitMixin[TPredictData], PredictUnit[TPredictData]):
    def __init__(
        self,
        *,
        module: torch.nn.Module,
        device: Optional[torch.device] = None,
        strategy: Optional[Union[Strategy, str]] = None,
        precision: Optional[Union[str, torch.dtype]] = None,
        torch_compile_params: Optional[TorchCompileParams] = None,
        detect_anomaly: Optional[bool] = None,
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
            torch_compile_params: params for Torch compile https://pytorch.org/docs/stable/generated/torch.compile.html
            detect_anomaly: whether to enable anomaly detection for the autograd engine https://pytorch.org/docs/stable/autograd.html#anomaly-detection

        Note:
            Torch compile support is only available in PyTorch 2.0 or higher.
        """
        super().__init__(
            module=module,
            device=device,
            precision=precision,
            torch_compile_params=torch_compile_params,
            detect_anomaly=detect_anomaly,
        )
        self.module: torch.nn.Module = prepare_module(
            module,
            self.device,
            strategy=strategy,
            torch_compile_params=torch_compile_params,
        )

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def predict_step(self, state: State, data: TPredictData) -> Any:
        # if detect_anomaly is true, run forward pass under detect_anomaly context
        detect_anomaly = self.detect_anomaly
        maybe_detect_anomaly = (
            torch.autograd.set_detect_anomaly(detect_anomaly)
            if detect_anomaly is not None
            else contextlib.nullcontext()
        )

        with self.maybe_autocast_precision, maybe_detect_anomaly:
            with get_timing_context(state, f"{self.__class__.__name__}.forward"):
                outputs = self.module(data)

        step = self.predict_progress.num_steps_completed
        self.on_predict_step_end(state, data, step, outputs)
        return outputs

    def on_predict_step_end(
        self,
        state: State,
        data: TPredictData,
        step: int,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        outputs: Any,
    ) -> None:
        """
        This will be called at the end of every ``predict_step`` before returning. The user can implement this method with code to update and log their metrics,
        or do anything else.

        Args:
            state: a State object which is passed from the ``predict_step``
            data: a batch of data which is passed from the ``predict_step``
            step: how many ``predict_step`` s have been completed
            outputs: the outputs of the model forward pass
        """
        pass

    def get_next_predict_batch(
        self, state: State, data_iter: Iterator[TPredictData]
    ) -> Union[Iterator[TPredictData], TPredictData]:
        # Override the default behavior from PredictUnit in order to enable prefetching if possible.
        pass_data_iter_to_step = _step_requires_iterator(self.predict_step)
        if pass_data_iter_to_step:
            return data_iter
        return self._get_next_batch(state, data_iter)


class AutoUnit(
    _AutoUnitMixin[TData],
    TrainUnit[TData],
    EvalUnit[TData],
    PredictUnit[TData],
    metaclass=_ConfigureOptimizersCaller,
):
    """
    The AutoUnit is a convenience for users who are training with stochastic gradient descent and would like to have model optimization
    and data parallel replication handled for them.
    The AutoUnit subclasses :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, and
    :class:`~torchtnt.framework.unit.PredictUnit` and implements the ``train_step``, ``eval_step``, and ``predict_step`` methods for the user.

    For the ``train_step`` it runs:

    - forward pass and loss computation
    - backward pass
    - optimizer step

    For the ``eval_step`` it only runs forward and loss computation.

    For the ``predict_step`` it only runs forward computation.

    To benefit from the AutoUnit, the user must subclass it and implement the ``compute_loss`` and ``configure_optimizers_and_lr_scheduler`` methods.
    Additionally, the AutoUnit offers these optional hooks:

    - ``on_train_step_end``
    - ``on_eval_step_end``
    - ``on_predict_step_end``

    Then use with the :py:func:`~torchtnt.framework.train`, :py:func:`~torchtnt.framework.evaluate`, :py:func:`~torchtnt.framework.fit`, or
    :py:func:`~torchtnt.framework.predict` entry point as normal.

    For more advanced customization, directly use the :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`,
    and :class:`~torchtnt.framework.unit.PredictUnit` interfaces.

    Args:
        module: module to be used during training/evaluation.
        device: the device to be used.
        strategy: the data parallelization strategy to be used. if a string, must be one of ``ddp`` or ``fsdp``.
        step_lr_interval: whether to step lr_scheduler every step or every epoch. Defaults to every epoch.
        precision: the precision to use in training/evaluation, as either a string or a torch.dtype.
        gradient_accumulation_steps: how many batches to accumulate gradients over.
        detect_anomaly: whether to enable anomaly detection for the autograd engine https://pytorch.org/docs/stable/autograd.html#anomaly-detection
        clip_grad_norm: max norm of the gradients for clipping https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
        clip_grad_value: max value of the gradients for clipping https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html
        swa_params: params for stochastic weight averaging https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging
        torch_compile_params: params for Torch compile https://pytorch.org/docs/stable/generated/torch.compile.html
        activation_checkpoint_params: params for enabling activation checkpointing
        training: if True, the optimizer and optionally LR scheduler will be created after the class is initialized.

    Note:
        If :class:`~torchtnt.utils.prepare_module.FSDPStrategy` and SWAParams are passed in, the swa model will be sharded with the same FSDP parameters.

    Note:
        Torch compile support is only available in PyTorch 2.0 or higher.

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
        torch_compile_params: Optional[TorchCompileParams] = None,
        activation_checkpoint_params: Optional[ActivationCheckpointParams] = None,
        training: bool = True,
    ) -> None:
        super().__init__(
            module=module,
            device=device,
            precision=precision,
            detect_anomaly=detect_anomaly,
            torch_compile_params=torch_compile_params,
        )

        if not gradient_accumulation_steps > 0:
            raise ValueError(
                f"gradient_accumulation_steps must be > 0. Got {gradient_accumulation_steps}"
            )

        self.swa_params: Optional[SWAParams] = swa_params
        self.swa_model: Optional[AveragedModel] = None
        if swa_params and training:
            module_for_swa = module
            skip_deepcopy = False
            if isinstance(strategy, FSDPStrategy):
                # must use exact same FSDPStrategy as original module
                # so each rank can computes EMA for its own local shard
                # since models are sharded identically if FSDP params are the same
                module_for_swa = prepare_fsdp(deepcopy(module), self.device, strategy)
                skip_deepcopy = True

            self.swa_model = AveragedModel(
                module_for_swa,
                device=device,
                avg_fn=swa_params.avg_fn,
                multi_avg_fn=swa_params.multi_avg_fn,
                use_buffers=True,
                skip_deepcopy=skip_deepcopy,
            )

        self.module: torch.nn.Module = prepare_module(
            module,
            self.device,
            strategy=strategy,
            torch_compile_params=torch_compile_params,
            activation_checkpoint_params=activation_checkpoint_params,
        )

        self.grad_scaler: Optional[GradScaler] = None
        if self.precision:
            self.grad_scaler = _get_grad_scaler_from_precision(
                self.precision,
                self.module,
            )

        self.step_lr_interval = step_lr_interval

        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        # create autocast context based on precision and device type

        self.training = training

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[TLRScheduler] = None
        self.swa_scheduler: Optional[SWALR] = None

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
    # pyre-fixme[3]: Return annotation cannot contain `Any`.
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

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def train_step(self, state: State, data: TData) -> Tuple[torch.Tensor, Any]:
        should_update_weights = (
            self.train_progress.num_steps_completed_in_epoch + 1
        ) % self.gradient_accumulation_steps == 0 or self._is_last_batch

        # for pyre, assign to local variable
        module = self.module

        # if using gradient accumulation with either DDP or FSDP, when in a step where we will not update the weights,
        # run forward and backward in no_sync context
        # https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync
        # https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync
        maybe_no_sync = (
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
                with get_timing_context(
                    state, f"{self.__class__.__name__}.compute_loss"
                ):
                    # Run the forward pass and compute the loss
                    loss, outputs = self.compute_loss(state, data)

            # normalize loss to account for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            if grad_scaler:
                scaled_loss = grad_scaler.scale(loss)
                with get_timing_context(state, f"{self.__class__.__name__}.backward"):
                    scaled_loss.backward()
            else:
                with get_timing_context(state, f"{self.__class__.__name__}.backward"):
                    loss.backward()

        if should_update_weights:
            self._update_weights(state)

        step = self.train_progress.num_steps_completed
        self.on_train_step_end(state, data, step, loss, outputs)
        return loss, outputs

    def on_train_step_end(
        self,
        state: State,
        data: TData,
        step: int,
        loss: torch.Tensor,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        outputs: Any,
    ) -> None:
        """
        This will be called at the end of every ``train_step`` before returning. The user can implement this method with code to update and log their metrics,
        or do anything else.

        Args:
            state: a State object which is passed from the ``train_step``
            data: a batch of data which is passed from the ``train_step``
            step: how many ``train_step`` s have been completed
            loss: the loss computed in the ``compute_loss`` function
            outputs: the outputs of the model forward pass
        """
        pass

    def on_train_epoch_end(self, state: State) -> None:
        """
        Note: if overriding ``on_train_epoch_end``, remember to call ``super().on_train_epoch_end()``
        """
        if self.step_lr_interval == "epoch":
            # number of epochs is incremented before calling this, so we're offsetting by 1
            self._update_lr_and_swa(state, self.train_progress.num_epochs_completed - 1)

        self._is_last_batch = False

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def eval_step(self, state: State, data: TData) -> Tuple[torch.Tensor, Any]:
        with self.maybe_autocast_precision:
            # users must override this
            with get_timing_context(state, f"{self.__class__.__name__}.compute_loss"):
                loss, outputs = self.compute_loss(state, data)

        if state.entry_point == EntryPoint.FIT:
            step = self.train_progress.num_steps_completed
        else:
            step = self.eval_progress.num_steps_completed

        self.on_eval_step_end(state, data, step, loss, outputs)
        return loss, outputs

    def on_eval_step_end(
        self,
        state: State,
        data: TData,
        step: int,
        loss: torch.Tensor,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        outputs: Any,
    ) -> None:
        """
        This will be called at the end of every ``eval_step`` before returning. The user can implement this method with code to update and log their metrics,
        or do anything else.

        Args:
            state: a State object which is passed from the ``eval_step``
            data: a batch of data which is passed from the ``eval_step``
            step: how many steps have been completed (``train_step`` s when running fit and ``eval_step`` s when running evaluation)
            loss: the loss computed in the ``compute_loss`` function
            outputs: the outputs of the model forward pass
        """
        pass

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def predict_step(self, state: State, data: TData) -> Any:
        with self.maybe_autocast_precision:
            with get_timing_context(state, f"{self.__class__.__name__}.forward"):
                outputs = self.module(data)

        step = self.predict_progress.num_steps_completed
        # users can override this, by default this is a no-op
        with get_timing_context(
            state, f"{self.__class__.__name__}.on_predict_step_end"
        ):
            self.on_predict_step_end(state, data, step, outputs)
        return outputs

    def on_predict_step_end(
        self,
        state: State,
        data: TData,
        step: int,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        outputs: Any,
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

    def _update_weights(self, state: State) -> None:
        module = self.module
        optimizer = none_throws(self.optimizer)
        grad_scaler = self.grad_scaler
        # Run gradient clipping, optimizer step, and zero_grad
        clip_grad_norm = self.clip_grad_norm
        clip_grad_value = self.clip_grad_value
        if grad_scaler and (clip_grad_norm or clip_grad_value):
            # unscale the gradients of optimizer's assigned params in-place in preparation for gradient clipping
            with get_timing_context(state, f"{self.__class__.__name__}.grad_unscale"):
                grad_scaler.unscale_(optimizer)

        # gradient norm clipping
        if clip_grad_norm:
            if _is_fsdp_module(module):
                if isinstance(module, FSDP):
                    with get_timing_context(
                        state, f"{self.__class__.__name__}.clip_grad_norm"
                    ):
                        module.clip_grad_norm_(max_norm=clip_grad_norm)
                else:
                    raise RuntimeError(
                        "Composable FSDP clip_grad_norm is not yet implemented: https://github.com/pytorch/pytorch/issues/97271"
                    )
            else:
                with get_timing_context(
                    state, f"{self.__class__.__name__}.clip_grad_norm"
                ):
                    torch.nn.utils.clip_grad_norm_(
                        parameters=module.parameters(),
                        max_norm=clip_grad_norm,
                    )

        # gradient value clipping
        if clip_grad_value:
            with get_timing_context(
                state, f"{self.__class__.__name__}.clip_grad_value"
            ):
                torch.nn.utils.clip_grad_value_(
                    parameters=module.parameters(),
                    clip_value=clip_grad_value,
                )

        with get_timing_context(state, f"{self.__class__.__name__}.optimizer_step"):
            if grad_scaler:
                grad_scaler.step(optimizer)
                # update the scale for next iteration
                grad_scaler.update()
            else:
                optimizer.step()

        # sets gradients to zero
        with get_timing_context(
            state, f"{self.__class__.__name__}.optimizer_zero_grad"
        ):
            optimizer.zero_grad(set_to_none=True)

        if self.step_lr_interval == "step":
            self._update_lr_and_swa(state, self.train_progress.num_steps_completed)

    def get_next_train_batch(
        self, state: State, data_iter: Iterator[TData]
    ) -> Union[Iterator[TData], TData]:
        # Override the default behavior from PredictUnit in order to enable prefetching if possible.
        pass_data_iter_to_step = _step_requires_iterator(self.train_step)
        if pass_data_iter_to_step:
            return data_iter
        return self._get_next_batch(state, data_iter)

    def get_next_eval_batch(
        self, state: State, data_iter: Iterator[TData]
    ) -> Union[Iterator[TData], TData]:
        # Override the default behavior from PredictUnit in order to enable prefetching if possible.
        pass_data_iter_to_step = _step_requires_iterator(self.eval_step)
        if pass_data_iter_to_step:
            return data_iter
        return self._get_next_batch(state, data_iter)

    def get_next_predict_batch(
        self, state: State, data_iter: Iterator[TData]
    ) -> Union[Iterator[TData], TData]:
        # Override the default behavior from PredictUnit in order to enable prefetching if possible.
        pass_data_iter_to_step = _step_requires_iterator(self.predict_step)
        if pass_data_iter_to_step:
            return data_iter
        return self._get_next_batch(state, data_iter)

    def _should_update_swa(self) -> bool:
        if not self.swa_params:
            return False

        swa_params = none_throws(self.swa_params)
        if self.step_lr_interval == "step":
            current_progress = self.train_progress.num_steps_completed
        else:
            # since num_epochs_completed is incremented prior to updating swa
            current_progress = self.train_progress.num_epochs_completed - 1

        if current_progress >= swa_params.warmup_steps_or_epochs:
            progress_since = current_progress - swa_params.warmup_steps_or_epochs
            update_freq = swa_params.step_or_epoch_update_freq
            return progress_since % update_freq == 0
        return False

    def _update_swa(self, state: State) -> None:
        with get_timing_context(
            state, f"{self.__class__.__name__}.stochastic_weight_avg_update"
        ):
            none_throws(self.swa_model).update_parameters(self.module)

    def _update_lr_and_swa(self, state: State, number_of_steps_or_epochs: int) -> None:
        if self._should_update_swa():
            self._update_swa(state)

        if self.swa_scheduler and (
            number_of_steps_or_epochs
            >= none_throws(self.swa_params).warmup_steps_or_epochs
        ):
            with get_timing_context(
                state, f"{self.__class__.__name__}.swa_lr_scheduler_step"
            ):
                none_throws(self.swa_scheduler).step()
        else:
            # optionally step lr scheduler if SWA not in use
            lr_scheduler = self.lr_scheduler
            if lr_scheduler:
                with get_timing_context(
                    state, f"{self.__class__.__name__}.lr_scheduler_step"
                ):
                    lr_scheduler.step()


def _validate_torch_compile_available() -> None:
    if not is_torch_version_ge_1_13_1():
        raise RuntimeError(
            "Torch compile support is available only in PyTorch 2.0 or higher. "
            "Please install PyTorch 2.0 or higher to continue: https://pytorch.org/get-started/locally/"
        )


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
