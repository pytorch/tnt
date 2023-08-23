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
from dataclasses import asdict
from typing import Any, Iterator, Optional, Tuple, TypeVar, Union

import torch
from pyre_extensions import none_throws
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, SWALR
from torchtnt.framework.state import ActivePhase, EntryPoint, State
from torchtnt.framework.unit import EvalUnit, PredictUnit, TPredictData, TrainUnit
from torchtnt.framework.utils import _is_fsdp_module, get_timing_context
from torchtnt.utils.device import copy_data_to_device, record_data_in_stream
from torchtnt.utils.env import init_from_env
from torchtnt.utils.lr_scheduler import TLRScheduler
from torchtnt.utils.misc import transfer_batch_norm_stats, transfer_weights
from torchtnt.utils.precision import convert_precision_str_to_dtype
from torchtnt.utils.prepare_module import (
    ActivationCheckpointParams,
    convert_str_to_strategy,
    DDPStrategy,
    FSDPStrategy,
    prepare_ddp,
    prepare_fsdp,
    prepare_module,
    Strategy,
    SWAParams,
    TorchCompileParams,
)
from torchtnt.utils.rank_zero_log import rank_zero_warn
from torchtnt.utils.version import is_torch_version_ge_1_13_1
from typing_extensions import Literal


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
        if torch_compile_params:
            _validate_torch_compile_available()

        super().__init__()

        self.device: torch.device = device or init_from_env()
        self.precision: Optional[torch.dtype] = (
            convert_precision_str_to_dtype(precision)
            if isinstance(precision, str)
            else precision
        )

        # create autocast context based on precision and device type
        self.maybe_autocast_precision = torch.autocast(
            device_type=self.device.type,
            dtype=self.precision,
            enabled=self.precision is not None,
        )
        if strategy:
            if isinstance(strategy, str):
                strategy = convert_str_to_strategy(strategy)
            if isinstance(strategy, DDPStrategy):
                if torch_compile_params and strategy.static_graph is True:
                    # https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860
                    raise RuntimeError(
                        "Torch compile requires DDPStrategy's static_graph to be False"
                    )
                module = prepare_ddp(module, self.device, strategy)
            elif isinstance(strategy, FSDPStrategy):
                if torch_compile_params and strategy.use_orig_params is False:
                    # as stated here https://pytorch.org/get-started/pytorch-2.0/
                    rank_zero_warn(
                        "We recommend setting FSDPStrategy's use_orig_params to True when using torch compile."
                    )
                module = prepare_fsdp(
                    module,
                    self.device,
                    strategy,
                )
        else:
            module = module.to(self.device)
        if torch_compile_params:
            try:
                # use in-place compile to avoid altering the state_dict keys
                module.compile(**asdict(torch_compile_params))
            except AttributeError:
                rank_zero_warn(
                    "Please install pytorch nightlies to use in-place compile to avoid altering the state_dict keys when checkpointing."
                )
                torch.compile(module, **asdict(torch_compile_params))
        self.module: torch.nn.Module = module

        # cuda stream to use for moving data to device
        self._prefetch_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.Stream() if self.device.type == "cuda" else None
        )
        # the next batch which has been prefetched and is ready to be used
        self._next_batch: Optional[TPredictData] = None

        # whether the next batch has been prefetched and is ready to be used
        self._prefetched: bool = False

        self.detect_anomaly = detect_anomaly

    def predict_step(self, state: State, data: Iterator[TPredictData]) -> Any:
        batch = self._get_next_batch(state, data)

        # if detect_anomaly is true, run forward pass under detect_anomaly context
        detect_anomaly = self.detect_anomaly
        maybe_detect_anomaly = (
            torch.autograd.set_detect_anomaly(detect_anomaly)
            if detect_anomaly is not None
            else contextlib.nullcontext()
        )

        with self.maybe_autocast_precision, maybe_detect_anomaly:
            with get_timing_context(state, f"{self.__class__.__name__}.forward"):
                outputs = self.module(batch)

        step = self.predict_progress.num_steps_completed
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
            step: how many ``predict_step`` s have been completed
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
            with get_timing_context(state, f"{self.__class__.__name__}.wait_stream"):
                # wait on the CUDA stream to complete the host to device copy
                torch.cuda.current_stream().wait_stream(self._prefetch_stream)

        # get the next batch which was stored by _prefetch_next_batch
        batch = self._next_batch
        if batch is None:
            self._prefetched = False
            raise StopIteration

        if self._prefetch_stream:
            with get_timing_context(
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
            with get_timing_context(
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
        with torch.cuda.stream(self._prefetch_stream), get_timing_context(
            state, f"{self.__class__.__name__}.move_data_to_device"
        ):
            self._next_batch = self.move_data_to_device(
                state, next_batch, non_blocking=non_blocking
            )


class AutoUnit(
    TrainUnit[TData],
    EvalUnit[TData],
    metaclass=_ConfigureOptimizersCaller,
):
    """
    The AutoUnit is a convenience for users who are training with stochastic gradient descent and would like to have model optimization
    and data parallel replication handled for them.
    The AutoUnit subclasses :class:`~torchtnt.framework.unit.TrainUnit` and :class:`~torchtnt.framework.unit.EvalUnit`,
    and implements the ``train_step`` and ``eval_step`` methods for the user.

    For the ``train_step`` it runs:

    - forward pass and loss computation
    - backward pass
    - optimizer step

    For the ``eval_step`` it only runs forward and loss computation.

    To benefit from the AutoUnit, the user must subclass it and implement the ``compute_loss`` and ``configure_optimizers_and_lr_scheduler`` methods.
    Additionally, the AutoUnit offers these optional hooks:

    - ``on_train_step_end``
    - ``on_eval_step_end``

    Then use with the :py:func:`~torchtnt.framework.train`, :py:func:`~torchtnt.framework.evaluate`, or :py:func:`~torchtnt.framework.fit` entry point as normal.

    For more advanced customization, directly use the :class:`~torchtnt.framework.unit.TrainUnit` and :class:`~torchtnt.framework.unit.EvalUnit` interfaces.

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
        Stochastic Weight Averaging is currently not supported with the FSDP strategy.

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
        super().__init__()

        if not gradient_accumulation_steps > 0:
            raise ValueError(
                f"gradient_accumulation_steps must be > 0. Got {gradient_accumulation_steps}"
            )
        if torch_compile_params:
            _validate_torch_compile_available()

        self.device: torch.device = device or init_from_env()
        self.precision: Optional[torch.dtype] = (
            convert_precision_str_to_dtype(precision)
            if isinstance(precision, str)
            else precision
        )

        self.module: torch.nn.Module = prepare_module(
            module,
            self.device,
            strategy,
            swa_params,
            torch_compile_params,
            activation_checkpoint_params,
        )

        self.grad_scaler: Optional[GradScaler] = None
        if self.precision:
            self.grad_scaler = _get_grad_scaler_from_precision(
                self.precision,
                module,
            )

        self.step_lr_interval = step_lr_interval

        self.gradient_accumulation_steps = gradient_accumulation_steps

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
        The user can override this method with custom code to copy data to device. This will be called at the start of every ``train_step``/``eval_step``.
        By default this uses the utility function :py:func:`~torchtnt.utils.copy_data_to_device`.

        If on GPU, this method will be called on a separate CUDA stream.

        Args:
            state: a State object which is passed from the ``train_step``/``eval_step``
            data: a batch of data which is passed from the ``train_step``/``eval_step``
            non_blocking: parameter to pass to ``torch.tensor.to``

        Returns:
            A batch of data which is on the device
        """
        return copy_data_to_device(data, self.device, non_blocking=non_blocking)

    def _prefetch_next_batch(self, state: State, data_iter: Iterator[TData]) -> None:
        """Prefetch the next batch on a separate CUDA stream."""

        phase = state.active_phase.name.lower()
        try:
            with get_timing_context(
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
        with torch.cuda.stream(self._prefetch_stream), get_timing_context(
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
            with get_timing_context(state, f"{self.__class__.__name__}.wait_stream"):
                # wait on the CUDA stream to complete the host to device copy
                torch.cuda.current_stream().wait_stream(self._prefetch_stream)

        # get the next batch which was stored by _prefetch_next_batch
        batch = self._next_batch
        if batch is None:
            self._prefetched = False
            self._is_last_train_batch = False
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

    def train_step(
        self, state: State, data: Iterator[TData]
    ) -> Tuple[torch.Tensor, Any]:
        # In auto unit they will not be exclusive since data fetching is done as
        # part of the training step
        with none_throws(state.train_state).iteration_timer.time("data_wait_time"):
            batch = self._get_next_batch(state, data)

        should_update_weights = (
            self.train_progress.num_steps_completed_in_epoch + 1
        ) % self.gradient_accumulation_steps == 0 or self._is_last_train_batch

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
                    loss, outputs = self.compute_loss(state, batch)

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
            # Run gradient clipping, optimizer step, and zero_grad
            clip_grad_norm = self.clip_grad_norm
            clip_grad_value = self.clip_grad_value
            if grad_scaler and (clip_grad_norm or clip_grad_value):
                # unscale the gradients of optimizer's assigned params in-place in preparation for gradient clipping
                with get_timing_context(
                    state, f"{self.__class__.__name__}.grad_unscale"
                ):
                    grad_scaler.unscale_(self.optimizer)

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
                    grad_scaler.step(self.optimizer)
                    # update the scale for next iteration
                    grad_scaler.update()
                else:
                    self.optimizer.step()

            # sets gradients to zero
            with get_timing_context(
                state, f"{self.__class__.__name__}.optimizer_zero_grad"
            ):
                self.optimizer.zero_grad(set_to_none=True)

            # optionally step lr scheduler if SWA not in use
            if (
                self.swa_params is None
                or self.train_progress.num_epochs_completed
                < self.swa_params.epoch_start
            ):
                lr_scheduler = self.lr_scheduler
                if lr_scheduler and self.step_lr_interval == "step":
                    with get_timing_context(
                        state, f"{self.__class__.__name__}.lr_scheduler_step"
                    ):
                        lr_scheduler.step()

        step = self.train_progress.num_steps_completed
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
            step: how many ``train_step`` s have been completed
            loss: the loss computed in the ``compute_loss`` function
            outputs: the outputs of the model forward pass
        """
        pass

    def on_train_epoch_end(self, state: State) -> None:
        """
        Note: if overriding ``on_train_epoch_end``, remember to call ``super().on_train_epoch_end()``
        """
        if (
            self.swa_model
            and self.swa_params
            and self.train_progress.num_epochs_completed >= self.swa_params.epoch_start
        ):
            with get_timing_context(
                state, f"{self.__class__.__name__}.stochastic_weight_avg_update"
            ):
                self.swa_model.update_parameters(self.module)
            with get_timing_context(
                state, f"{self.__class__.__name__}.stochastic_weight_avg_step"
            ):
                none_throws(self.swa_scheduler).step()
        elif self.lr_scheduler and self.step_lr_interval == "epoch":
            # optionally step lr scheduler
            with get_timing_context(
                state, f"{self.__class__.__name__}.lr_scheduler_step"
            ):
                self.lr_scheduler.step()

    def on_train_end(self, state: State) -> None:
        """
        Note that if using SWA and implementing `on_train_end()`, must call `super().on_train_end()`.
        """
        swa_model = self.swa_model
        if swa_model:
            with get_timing_context(
                state,
                f"{self.__class__.__name__}.stochastic_weight_avg_transfer_weights",
            ):
                transfer_weights(swa_model, self.module)
            with get_timing_context(
                state,
                f"{self.__class__.__name__}.stochastic_weight_avg_transfer_batch_norm_stats",
            ):
                transfer_batch_norm_stats(swa_model, self.module)

    def eval_step(self, state: State, data: TData) -> Tuple[torch.Tensor, Any]:
        with get_timing_context(
            state, f"{self.__class__.__name__}.move_data_to_device"
        ):
            data = self.move_data_to_device(state, data, non_blocking=False)

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
        self, state: State, data: TData, step: int, loss: torch.Tensor, outputs: Any
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
