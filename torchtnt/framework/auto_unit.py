# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ignore errors due to `Any` type
# pyre-ignore-all-errors[2]
# pyre-ignore-all-errors[3]

import contextlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Tuple, TypeVar, Union

import torch
from pyre_extensions import none_throws
from torch.cuda.amp import GradScaler
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, SWALR
from torchtnt.framework.state import EntryPoint, State
from torchtnt.framework.unit import EvalUnit, PredictUnit, TrainUnit
from torchtnt.utils import (
    copy_data_to_device,
    init_from_env,
    is_torch_version_geq_1_12,
    TLRScheduler,
    transfer_batch_norm_stats,
    transfer_weights,
)
from torchtnt.utils.rank_zero_log import rank_zero_warn
from torchtnt.utils.version import is_torch_version_ge_1_13_1
from typing_extensions import Literal

TSWA_avg_fn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


@dataclass
class Strategy:
    """Dataclass representing the parallelization strategy for the AutoParallelUnit"""

    pass


@dataclass
class DDPStrategy(Strategy):
    """Dataclass representing the `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ strategy"""

    output_device: Optional[Union[int, torch.device]] = None
    dim: int = 0
    broadcast_buffers: bool = True
    process_group: Optional[ProcessGroup] = None
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    check_reduction: bool = False
    gradient_as_bucket_view: bool = False
    static_graph: bool = False


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


# pyre-ignore: Invalid type parameters [24]
TSelf = TypeVar("TSelf", bound="AutoUnit")
TData = TypeVar("TData")


class AutoUnit(TrainUnit[TData], EvalUnit[TData], PredictUnit[Any], ABC):
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

    To benefit from the AutoUnit, the user must subclass it and implement the ``compute_loss`` method, and optionally the ``update_metrics`` and ``log_metrics`` methods.
    Then use with the :py:func:`~torchtnt.framework.train`, :py:func:`~torchtnt.framework.evaluate`, :py:func:`~torchtnt.framework.predict`, or :py:func:`~torchtnt.framework.fit` entry point as normal.

    For more advanced customization, directly use the :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, and :class:`~torchtnt.framework.unit.PredictUnit` interfaces.

    Args:
        module: module to be used during training.
        device: the device to be used.
        strategy: the data parallelization strategy to be used
        step_lr_interval: whether to step lr_scheduler every step or every epoch. Defaults to every epoch.
        log_frequency_steps: how often to log in terms of steps (parameter updates) during training.
        precision: the precision to use in training, as either a string or a torch.dtype.
        gradient_accumulation_steps: how many batches to accumulate gradients over.
        detect_anomaly: whether to enable anomaly detection for the autograd engine https://pytorch.org/docs/stable/autograd.html#anomaly-detection
        clip_grad_norm: max norm of the gradients for clipping https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
        clip_grad_value: max value of the gradients for clipping https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html
        swa_params: params for stochastic weight averaging https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging
        torchdynamo_params: params for TorchDynamo https://pytorch.org/docs/master/dynamo/

            Note:
                TorchDynamo support is only available in PyTorch 2.0 or higher.
    """

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        device: Optional[torch.device] = None,
        strategy: Optional[Strategy] = None,
        step_lr_interval: Literal["step", "epoch"] = "epoch",
        log_frequency_steps: int = 1000,
        precision: Optional[Union[str, torch.dtype]] = None,
        gradient_accumulation_steps: int = 1,
        detect_anomaly: bool = False,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
        swa_params: Optional[SWAParams] = None,
        torchdynamo_params: Optional[TorchDynamoParams] = None,
    ) -> None:
        super().__init__()
        self.device: torch.device = device or init_from_env()

        if strategy:
            if isinstance(strategy, DDPStrategy):
                # move module to device
                module = module.to(self.device)
                # wrap module in DDP
                device_ids = None
                if self.device.type == "cuda":
                    device_ids = [self.device.index]
                module = DDP(module, device_ids=device_ids, **asdict(strategy))
                if torchdynamo_params:
                    # TODO: Add support for dynamo and DDP
                    rank_zero_warn(
                        "Torchdynamo params has been set with DDP - Note that performance will likely be slower and we recommend using only one."
                    )

        else:
            # move module to device
            module = module.to(self.device)

        self.module: torch.nn.Module = module

        self.step_lr_interval = step_lr_interval
        if not log_frequency_steps > 0:
            raise ValueError(
                f"log_frequency_steps must be > 0. Got {log_frequency_steps}"
            )
        self.log_frequency_steps: int = log_frequency_steps

        if not precision:
            self.precision: Optional[torch.dtype] = None
            self.grad_scaler: Optional[GradScaler] = None
        else:
            if isinstance(precision, str):
                self.precision: Optional[torch.dtype] = _convert_precision_str_to_dtype(
                    precision
                )
            else:
                self.precision = precision

            self.grad_scaler = _get_grad_scaler_from_precision(
                # pyre-ignore
                self.precision,
                self.module,
            )

        if not gradient_accumulation_steps > 0:
            raise ValueError(
                f"gradient_accumulation_steps must be > 0. Got {gradient_accumulation_steps}"
            )
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._num_optimizer_steps_completed: int = 0

        self.detect_anomaly = detect_anomaly
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        # create autocast context based on precision and device type
        self.maybe_autocast_precision = torch.autocast(
            device_type=self.device.type,
            dtype=self.precision,
            enabled=self.precision is not None,
        )

        optimizer, lr_scheduler = self.configure_optimizers_and_lr_scheduler(
            self.module
        )
        self.optimizer: torch.optim.optimizer.Optimizer = optimizer
        self.lr_scheduler: TLRScheduler = lr_scheduler

        self.swa_model: Optional[AveragedModel] = None
        self.swa_scheduler: Optional[SWALR] = None
        self.swa_params: Optional[SWAParams] = swa_params
        if swa_params:
            if not swa_params.avg_fn:
                # pyre-ignore: Unexpected keyword [28]
                self.swa_model = AveragedModel(self.module, use_buffers=True)
            else:
                # pyre-ignore: Unexpected keyword [28]
                self.swa_model = AveragedModel(
                    self.module, avg_fn=swa_params.avg_fn, use_buffers=True
                )

            self.swa_scheduler = SWALR(
                optimizer=self.optimizer,
                swa_lr=swa_params.lr,
                anneal_epochs=swa_params.anneal_epochs,
                anneal_strategy=swa_params.anneal_strategy,
            )

        if torchdynamo_params:
            if not is_torch_version_ge_1_13_1():
                raise RuntimeError(
                    "TorchDynamo support is available only in PyTorch 2.0 or higher. "
                    "Please install PyTorch 2.0 or higher to continue: https://pytorch.org/get-started/locally/"
                )
            # pyre-ignore
            self.compute_loss = _dynamo_wrapper(self.compute_loss, torchdynamo_params)
            # pyre-ignore
            self._forward_and_backward = _dynamo_wrapper(
                self._forward_and_backward, torchdynamo_params
            )
            self.module = _dynamo_wrapper(self.module, torchdynamo_params)

        # TODO: Make AutoTrainUnit work when data type is Iterator

    @abstractmethod
    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
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
        """
        ...

    def update_metrics(
        self, state: State, data: TData, loss: torch.Tensor, outputs: Any
    ) -> None:
        """
        The user should implement this method with code to update metrics. This will be called every ``train_step``/``eval_step``.

        Args:
            state: a State object which is passed from the ``train_step``/``eval_step``
            data: a batch of data which is passed from the ``train_step``/``eval_step``
            outputs: the outputs of the model forward pass
        """
        pass

    def log_metrics(
        self, state: State, step: int, interval: Literal["step", "epoch"]
    ) -> None:
        """
        The user should implement this method with their code to log metrics. This will be called:

        - every ``train_step`` based on ``log_frequency_steps`` and how many parameter updates have been run on the model
        - in ``on_train_epoch_end`` and ``on_eval_epoch_end``

        Args:
            state: a State object which is passed from ``train_step``/``on_train_epoch_end``/``on_eval_epoch_end``
            step: how many steps have been completed (i.e. how many parameter updates have been run on the model)
            interval: whether ``log_metrics`` is called at the end of a step or at the end of an epoch
        """
        pass

    def move_data_to_device(self, state: State, data: TData) -> TData:
        """
        The user can override this method with custom code to copy data to device. This will be called at the start of every ``train_step``/``eval_step``/``predict_step``.
        By default this uses the utility function :py:func:`~torchtnt.utils.copy_data_to_device`.

        Args:
            state: a State object which is passed from the ``train_step``/``eval_step``/``predict_step``
            data: a batch of data which is passed from the ``train_step``/``eval_step``/``predict_step``

        Returns:
            A batch of data which is on the device
        """
        return copy_data_to_device(data, self.device)

    def train_step(self, state: State, data: TData) -> Tuple[torch.Tensor, Any]:
        data = self.move_data_to_device(state, data)

        train_state = none_throws(state.train_state)
        should_update_weights = (
            train_state.progress.num_steps_completed_in_epoch + 1
        ) % self.gradient_accumulation_steps == 0 or train_state.is_last_batch

        loss, outputs = self._forward_and_backward(state, data, should_update_weights)

        # users can override this, by default this is a no-op
        self.update_metrics(state, data, loss, outputs)

        if should_update_weights:
            # TODO try to use dynamo here
            self._run_optimizer_lr_scheduler_step(state)

            # log metrics only after an optimizer step
            if self.num_optimizer_steps_completed % self.log_frequency_steps == 0:
                self.log_metrics(state, self.num_optimizer_steps_completed - 1, "step")
        return loss, outputs

    def _forward_and_backward(
        self, state: State, data: TData, should_update_weights: bool
    ):
        # if using gradient accumulation and DDP or FSDP, when in a step where we will not update the weights,
        # run forward and backward in no_sync context
        # https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync
        # https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync
        maybe_no_sync = (
            self.module.no_sync()
            if not should_update_weights and isinstance(self.module, (DDP, FSDP))
            else contextlib.nullcontext()
        )

        # if detect_anomaly is true, run forward and backward pass in detect_anomaly context
        with maybe_no_sync, torch.autograd.set_detect_anomaly(self.detect_anomaly):
            with self.maybe_autocast_precision:
                # users must override this
                loss, outputs = self.compute_loss(state, data)

            # normalize loss to account for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            grad_scaler = self.grad_scaler
            if grad_scaler:
                loss = grad_scaler.scale(loss)
            loss.backward()
        return loss, outputs

    def _run_optimizer_lr_scheduler_step(self, state: State) -> None:
        """Runs the optimizer step, sets gradients to zero, runs lr scheduler step, and calls `log_metrics`"""
        # optimizer step
        grad_scaler = self.grad_scaler
        clip_grad_norm = self.clip_grad_norm
        clip_grad_value = self.clip_grad_value
        if grad_scaler and (clip_grad_norm or clip_grad_value):
            # unscale the gradients of optimizer's assigned params in-place in preparation for gradient clipping
            grad_scaler.unscale_(self.optimizer)

        # gradient norm clipping
        if clip_grad_norm:
            if isinstance(self.module, FSDP):
                self.module.clip_grad_norm_(max_norm=clip_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.module.parameters(),
                    max_norm=clip_grad_norm,
                )
        # gradient value clipping
        if clip_grad_value:
            torch.nn.utils.clip_grad_value_(
                parameters=self.module.parameters(),
                clip_value=clip_grad_value,
            )

        # optimizer step
        if grad_scaler:
            grad_scaler.step(self.optimizer)
            # update the scale for next iteration
            grad_scaler.update()
        else:
            self.optimizer.step()

        self._num_optimizer_steps_completed += 1

        # sets gradients to zero
        self.optimizer.zero_grad(set_to_none=True)

        # optionally step lr scheduler if SWA not in use
        train_state = none_throws(state.train_state)
        if (
            self.swa_params is None
            or train_state.progress.num_epochs_completed < self.swa_params.epoch_start
        ):
            lr_scheduler = self.lr_scheduler
            if lr_scheduler and self.step_lr_interval == "step":
                lr_scheduler.step()

    def on_train_epoch_end(self, state: State) -> None:
        # note: if user wants to override on_train_epoch_end themselves, they should remember to call up to this method via super().on_train_epoch_end()

        train_state = none_throws(state.train_state)

        if (
            self.swa_model
            and self.swa_params
            and train_state.progress.num_epochs_completed >= self.swa_params.epoch_start
        ):
            self.swa_model.update_parameters(self.module)
            none_throws(self.swa_scheduler).step()
        elif self.lr_scheduler and self.step_lr_interval == "epoch":
            # optionally step lr scheduler
            self.lr_scheduler.step()

        # users can override this, by default this is a no-op
        self.log_metrics(state, self.num_optimizer_steps_completed, "epoch")

    def on_train_end(self, state: State) -> None:
        """
        Note that if using SWA and implementing `on_train_end()`, must call `super().on_train_end()`.
        """
        swa_model = self.swa_model
        if swa_model:
            transfer_weights(swa_model, self.module)
            transfer_batch_norm_stats(swa_model, self.module)

    def eval_step(self, state: State, data: TData) -> Tuple[torch.Tensor, Any]:
        data = self.move_data_to_device(state, data)

        with self.maybe_autocast_precision:
            # users must override this
            loss, outputs = self.compute_loss(state, data)

        # users can override this, by default this is a no-op
        self.update_metrics(state, data, loss, outputs)
        return loss, outputs

    def on_eval_epoch_end(self, state: State) -> None:
        # note: if user wants to override on_eval_epoch_end themselves, they should remember to call up to this method via super().on_eval_epoch_end()
        if state.entry_point == EntryPoint.FIT:
            # if in fit, use the number of optimizer steps completed
            # users can override this, by default this is a no-op
            self.log_metrics(state, self.num_optimizer_steps_completed, "epoch")
        else:
            eval_state = none_throws(state.eval_state)

            # if in evaluate, use the number of eval steps completed
            # users can override this, by default this is a no-op
            self.log_metrics(state, eval_state.progress.num_steps_completed, "epoch")

    def predict_step(self, state: State, data: Any) -> Any:
        data = self.move_data_to_device(state, data)

        with self.maybe_autocast_precision:
            outputs = self.module(data)
        return outputs

    @property
    def num_optimizer_steps_completed(self) -> int:
        return self._num_optimizer_steps_completed


def _convert_precision_str_to_dtype(precision: str) -> torch.dtype:
    """
    Converts precision as a string to a torch.dtype

    Args:
        precision: string containing the precision

    Raises:
        ValueError if an invalid precision string is passed.

    """
    string_to_dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in string_to_dtype_mapping.keys():
        raise ValueError(
            f"Precision {precision} not supported. Please use one of `fp16` or `bf16`"
        )
    return string_to_dtype_mapping[precision]


def _get_grad_scaler_from_precision(
    precision: torch.dtype, module: torch.nn.Module
) -> Optional[GradScaler]:
    if precision == torch.float16:
        if isinstance(module, FSDP):
            if not is_torch_version_geq_1_12():
                raise RuntimeError(
                    "Using float16 precision with torch.distributed.fsdp.FullyShardedDataParallel requires "
                    "torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler from PyTorch 1.12. "
                    "Please install PyTorch 1.12 or higher to continue: https://pytorch.org/get-started/locally/"
                )
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
