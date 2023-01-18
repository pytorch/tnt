# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import torch
from torch.distributed import ProcessGroup
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtnt.framework.auto_unit import AutoUnit, TData
from torchtnt.utils import init_from_env, TLRScheduler


class AutoDDPUnit(AutoUnit[TData], ABC):
    """
    The AutoDDPUnit is a convenience for users who are training with stochastic gradient descent and would like to have model optimization
    *and* data parallel replication with `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ handled for them.
    The AutoDDPUnit subclasses :class:`~torchtnt.runner.auto_unit.AutoUnit` to add DDP features on top.

    For more advanced customization, the basic :class:`~torchtnt.runner.unit.TrainUnit`/:class:`~torchtnt.runner.unit.EvalUnit`/:class:`~torchtnt.runner.unit.PredictUnit` interface may be a better fit.

    Args:
        module: module to be used during training.
        device: the device to be used.

        output_device: will be passed to constructor argument of `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ with the same name.
        dim: will be passed to constructor argument of `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ with the same name.
        broadcast_buffers: will be passed to constructor argument of `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ with the same name.
        process_group: will be passed to constructor argument of `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ with the same name.
        bucket_cap_mb: will be passed to constructor argument of `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ with the same name.
        find_unused_parameters: will be passed to constructor argument of `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ with the same name.
        check_reduction: will be passed to constructor argument of `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ with the same name.
        gradient_as_bucket_view: will be passed to constructor argument of `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ with the same name.
        static_graph: will be passed to constructor argument of `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ with the same name.

        kwargs: other keyword arguments to pass to the AutoUnit.
    """

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        device: Optional[torch.device] = None,
        # optional ddp constructor args
        output_device: Optional[Union[int, torch.device]] = None,
        dim: int = 0,
        broadcast_buffers: bool = True,
        process_group: Optional[ProcessGroup] = None,
        bucket_cap_mb: int = 25,
        find_unused_parameters: bool = False,
        check_reduction: bool = False,
        gradient_as_bucket_view: bool = False,
        static_graph: bool = False,
        # kwargs to be passed to AutoUnit
        **kwargs: Any,
    ) -> None:
        device = device or init_from_env()
        module = module.to(device)

        device_ids = None
        if device.type == "cuda":
            device_ids = [device.index]
        module = DDP(
            module,
            device_ids=device_ids,
            output_device=output_device,
            dim=dim,
            broadcast_buffers=broadcast_buffers,
            process_group=process_group,
            bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=find_unused_parameters,
            check_reduction=check_reduction,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )

        optimizer, lr_scheduler = self.configure_optimizers_and_lr_scheduler(module)

        super().__init__(
            module=module,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            **kwargs,
        )

    @abstractmethod
    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        """
        The user should implement this method with their optimizer and learning rate scheduler construction code. This will be called upon initialization of
        the AutoDDPUnit.

        Args:
            module: the module with which to construct optimizer and lr_scheduler

        Returns:
            Either an optimizer, or a tuple containing optimizer and optional lr scheduler
        """
        ...
