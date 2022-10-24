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
from typing import Any, Optional, Tuple, Union

import torch
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtnt.runner.state import State
from torchtnt.runner.unit import TrainUnit, TTrainData
from torchtnt.utils import copy_data_to_device, get_device_from_env
from typing_extensions import Literal


class AutoTrainUnit(TrainUnit[TTrainData], ABC):
    """
    The AutoTrainUnit is a convenience for users who are training with stochastic gradient descent and would like to have model optimization
    handled for them. The AutoTrainUnit subclasses TrainUnit, and runs the train_step for the user, specifically: forward pass, loss computation,
    backward pass, and optimizer step. To benefit from the AutoTrainUnit, the user must subclass it and implement the `compute_loss` method, and
    optionally the `update_metrics` and `log_metrics` methods. Then use with the `train` or `fit` entry point as normal.

    For more advanced customization, the basic TrainUnit interface may be a better fit.

    Args:
        module: module to be used during training.
        optimizer: optimizer to be used during training.
        lr_scheduler: lr_scheduler to be used during training.
        step_lr_interval: whether to step lr_scheduler every step or every epoch. Defaults to every epoch.
        device: the device to be used.
        log_frequency_steps: how often to log in terms of steps (parameter updates)
        precision: the precision to use in training, as either a string or a torch.dtype.
        gradient_accumulation_steps: how often to accumulate gradients (every gradient_accumulation_steps)
        detect_anomaly: whether to enable anomaly detection for the autograd engine https://pytorch.org/docs/stable/autograd.html#anomaly-detection

    Attributes:
        module: module to be used during training.
        optimizer: optimizer to be used during training.
        lr_scheduler: lr_scheduler to be used during training.
        step_lr_interval: whether to step lr_scheduler every step or every epoch. Defaults to every epoch.
        device: the device to be used.
        log_frequency_steps: how often to log in terms of steps (parameter updates)
        precision: the precision to use in training, as a torch.dtype.
        grad_scaler: a torch.cuda.amp.GradScaler, if using fp16 precision
        gradient_accumulation_steps: how often to accumulate gradients (every gradient_accumulation_steps)
        num_optimizer_steps_completed: number of optimizer steps (weight updates) completed
        detect_anomaly: whether to enable anomaly detection for the autograd engine https://pytorch.org/docs/stable/autograd.html#anomaly-detection
    """

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        step_lr_interval: Literal["step", "epoch"] = "epoch",
        device: Optional[torch.device] = None,
        log_frequency_steps: int = 1000,
        precision: Optional[Union[str, torch.dtype]] = None,
        gradient_accumulation_steps: int = 1,
        detect_anomaly: bool = False,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step_lr_interval = step_lr_interval
        self.device: torch.device = device or get_device_from_env()
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

        # TODO: Make AutoTrainUnit work when data type is Iterator

    @abstractmethod
    def compute_loss(self, state: State, data: TTrainData) -> Tuple[torch.Tensor, Any]:
        """
        The user should implement this method with their loss computation. This will be called every `train_step`.

        Args:
            state: a State object which is passed from the `train_step`
            data: a batch of data which is passed from the `train_step`

        Returns:
            Tuple containing the loss and the output of the model
        """
        ...

    def update_metrics(
        self, state: State, data: TTrainData, loss: torch.Tensor, outputs: Any
    ) -> None:
        """
        The user should implement this method with code to update metrics. This will be called every `train_step`.

        Args:
            state: a State object which is passed from the `train_step`
            data: a batch of data which is passed from the `train_step`
            outputs: the outputs of the model forward pass
        """
        pass

    def log_metrics(
        self, state: State, step: int, interval: Literal["step", "epoch"]
    ) -> None:
        """
        The user should implement this method with their code to log metrics. This will be called based on `log_frequency_steps`
        and how many parameter updates have been run on the model.

        Args:
            state: a State object which is passed from the `train_step`
            step: how many steps have been completed (i.e. how many parameter updates have been run on the model)
            interval: whether `log_metrics` is called at the end of a step or at the end of an epoch
        """
        pass

    def train_step(self, state: State, data: TTrainData) -> Tuple[torch.Tensor, Any]:
        data = copy_data_to_device(data, self.device)
        assert state.train_state

        should_update_weights = (
            state.train_state.progress.num_steps_completed_in_epoch + 1
        ) % self.gradient_accumulation_steps == 0

        maybe_autocast_precision = torch.autocast(
            device_type=self.device.type,
            dtype=self.precision,
            enabled=self.precision is not None,
        )
        # if using gradient accumulation and DDP or FSDP, when in a step where we will not update the weights,
        # run forward and backward in no_sync context
        # https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync
        # https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync
        maybe_no_sync = (
            self.module.no_sync()  # pyre-ignore
            if not should_update_weights and isinstance(self.module, (DDP, FSDP))
            else contextlib.nullcontext()
        )

        # if detect_anomaly is true, run forward and backward pass in detect_anomaly context
        with maybe_no_sync, torch.autograd.set_detect_anomaly(self.detect_anomaly):
            with maybe_autocast_precision:
                # users must override this
                loss, outputs = self.compute_loss(state, data)

            # normalize loss to account for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            grad_scaler = self.grad_scaler
            if grad_scaler:
                loss = grad_scaler.scale(loss)
            loss.backward()

        # users can override this, by default this is a no-op
        self.update_metrics(state, data, loss, outputs)

        if should_update_weights:
            self._run_optimizer_lr_scheduler_step(state)

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

        # optionally step lr scheduler
        lr_scheduler = self.lr_scheduler
        if lr_scheduler and self.step_lr_interval == "step":
            lr_scheduler.step()

        # call `log_metrics`
        if self.num_optimizer_steps_completed % self.log_frequency_steps == 0:
            # users can override this, by default this is a no-op
            self.log_metrics(state, self.num_optimizer_steps_completed - 1, "step")

    def on_train_epoch_end(self, state: State) -> None:
        # note: if user wants to override on_train_epoch_end themselves, they should remember to call up to this method via super().on_train_epoch_end()
        assert state.train_state
        train_state = state.train_state

        # in the case that the number of training steps is not evenly divisible by gradient_accumulation_steps, we must update the weights one last
        # time for the last step
        should_update_weights_for_last_step = (
            train_state.progress.num_steps_completed_in_epoch
            % self.gradient_accumulation_steps
            != 0
        )

        if should_update_weights_for_last_step:
            self._run_optimizer_lr_scheduler_step(state)

        # optionally step lr scheduler
        if self.lr_scheduler and self.step_lr_interval == "epoch":
            self.lr_scheduler.step()

        # users can override this, by default this is a no-op
        self.log_metrics(state, train_state.progress.num_steps_completed, "epoch")

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
            return ShardedGradScaler()
        else:
            return GradScaler()
    return None
