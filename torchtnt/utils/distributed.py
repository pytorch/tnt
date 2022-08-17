#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from typing_extensions import Literal


class PGWrapper:
    """
    A wrapper around ProcessGroup that allows collectives to be issued in a
    consistent fashion regardless of the following scenarios:

        pg is None, distributed is initialized:     use WORLD as pg
        pg is None, distributed is not initialized: single process app
        pg is not None:                             use pg
    """

    def __init__(self, pg: Optional[dist.ProcessGroup]) -> None:
        if pg is None and dist.is_initialized():
            self.pg: Optional[dist.ProcessGroup] = dist.group.WORLD
        else:
            self.pg: Optional[dist.ProcessGroup] = pg

    def get_rank(self) -> int:
        if self.pg is None:
            return 0
        return dist.get_rank(group=self.pg)

    def get_world_size(self) -> int:
        if self.pg is None:
            return 1
        return dist.get_world_size(group=self.pg)

    def barrier(self) -> None:
        if self.pg is None:
            return
        dist.barrier(group=self.pg)

    # pyre-ignore[2]: Parameter must have a type that does not contain `Any`
    def broadcast_object_list(self, obj_list: List[Any], src: int = 0) -> None:
        if self.pg is None:
            return
        dist.broadcast_object_list(obj_list, src=src, group=self.pg)

    # pyre-ignore[2]: Parameter must have a type that does not contain `Any`
    def all_gather_object(self, obj_list: List[Any], obj: Any) -> None:
        if self.pg is None:
            obj_list[0] = obj
            return
        dist.all_gather_object(obj_list, obj, group=self.pg)

    def scatter_object_list(
        self,
        # pyre-ignore[2]: Parameter must have a type that does not contain `Any`
        output_list: List[Any],
        # pyre-ignore[2]: Parameter must have a type that does not contain `Any`
        input_list: Optional[List[Any]],
        src: int = 0,
    ) -> None:
        rank = self.get_rank()
        world_size = self.get_world_size()
        if rank == src:
            if input_list is None:
                raise RuntimeError(
                    "The src rank's input_list for scatter_object_list must not be None."
                )
            if len(input_list) != world_size:
                raise RuntimeError(
                    f"The length of input_list {len(input_list)} for scatter_object_list "
                    f"must be the same as the process group's world size ({world_size})."
                )
        else:
            input_list = [None] * world_size

        if self.pg is None:
            output_list[0] = input_list[0]
            return

        # scatter_object_list does not yet support NCCL backend
        if dist.get_backend(self.pg) == "nccl":
            self.broadcast_object_list(obj_list=input_list, src=src)
            output_list[0] = input_list[rank]
            return

        dist.scatter_object_list(output_list, input_list, src=src, group=self.pg)


def get_global_rank() -> int:
    """
    Get rank using torch.distributed if available. Otherwise, the RANK env var instead if initialized.
    Returns 0 if neither condition is met.
    """
    if dist.is_initialized():
        return dist.get_rank()

    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(os.environ["RANK"])

    return 0


def get_process_group_backend_from_device(device: torch.device) -> str:
    """Function that gets the defaut process group backend from the device."""
    return "nccl" if device.type == "cuda" else "gloo"


def _simple_all_gather_tensors(
    result: Tensor, group: torch.distributed.group, world_size: int
) -> List[Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result, group)
    return gathered_result


def all_gather_tensors(
    result: Tensor, group: Optional[torch.distributed.group] = None
) -> List[Tensor]:
    """Function to gather tensors from several distributed processes onto a list that is broadcasted to all processes.
    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    # if torch.distributed is not available or not initialized
    # return single-item list containing the result
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [result]

    # if group is None, fallback to the default process group
    if group is None:
        group = torch.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)
    torch.distributed.barrier(group=group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        # pyre-fixme[6]: For 2nd param expected `group` but got `Union[None, group,
        #  ProcessGroup]`.
        return _simple_all_gather_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        # pyre-fixme[6]: For 2nd param expected `group` but got `Union[None, group,
        #  ProcessGroup]`.
        return _simple_all_gather_tensors(result, group, world_size)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result


TReturn = TypeVar("TReturn")


def rank_zero_fn(fn: Callable[..., TReturn]) -> Callable[..., Optional[TReturn]]:
    """Function that can be used as a decorator to enable a function to be called on global rank 0 only.

    Note:
        This decorator should be used judiciously. it should never be used on functions that need synchronization.
        It should be used very carefully with functions that mutate local state as well

    Example:

        >>> from torchtnt.utilities.distributed import rank_zero_fn
        >>> @rank_zero_fn
        ... def foo():
        ...     return 1
        ...
        >>> x = foo() # x is 1 if global rank is 0 else x is None

    Args:
        fn: the desired function to be executed on rank 0 only

    Return:
        wrapped_fn: the wrapped function that executes only if the global rank is  0

    """

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[TReturn]:
        if get_global_rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


class _BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    """
    The only difference between :class:`torch.nn.BatchNorm1d`, :class:`torch.nn.BatchNorm2d`,
    :class:`torch.nn.BatchNorm3d`, etc is this method that is overwritten by the sub-class.
    This method is used when calling forward as a sanity check.
    When using :function:`revert_sync_batchnorm` this sanity check is lost.
    """

    def _check_input_dim(self, input: Tensor) -> None:
        return


def revert_sync_batchnorm(
    module: torch.nn.Module, device: Optional[Union[str, torch.device]] = None
) -> torch.nn.Module:
    """
    Helper function to convert all :class:`torch.nn.SyncBatchNorm` layers in the module to
    :attr:`BatchNorm*D` layers. This function reverts :meth:`torch.nn.SyncBatchNorm.convert_sync_batchnorm`.

    Args:
        module (nn.Module): module containing one or more :class:`torch.nn.SyncBatchNorm` layers
        device (optional): device in which the :attr:`BatchNorm*D` should be created,
                default is cpu

    Returns:
        The original :attr:`module` with the converted :attr:`BatchNorm*D`
        layers. If the original :attr:`module` is a :class:`torch.nn.SyncBatchNorm` layer,
        a new :attr:`BatchNorm*D` layer object will be returned
        instead. Note that the :attr:`BatchNorm*D` layers returned will not have input dimension information.

    Example::

        >>> # Network with nn.BatchNorm layer
        >>> module = torch.nn.Sequential(
        >>>            torch.nn.Linear(20, 100),
        >>>            torch.nn.BatchNorm1d(100),
        >>>          ).cuda()
        >>> sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        >>> reverted_module = revert_sync_batchnorm(sync_bn_module, torch.device("cuda"))

    """
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        module_output = _BatchNormXd(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
            device,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            # pyre-ignore[16]: `_BatchNormXd` has no attribute `qconfig`.
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child, device))
    del module
    return module_output


def sync_bool(
    val: bool,
    pg: Optional[dist.ProcessGroup] = None,
    coherence_mode: Union[Literal["any", "all", "rank_zero"], int, float] = "any",
) -> bool:
    """Utility to synchronize a boolean value across members of a provided process group.

    In the case ``torch.distributed`` is not available or initialized, the input ``val`` is returned.

    Args:
        val (bool): boolean value to synchronize
        pg: process group to use for synchronization. If not specified, the default process group is used.
        coherence_mode Union[str, int, float]: the manner in which the boolean value should be synchronized. 5 options are currently supported:
            1. any (default): If any rank provides a True value, all ranks should receive True.
            2. all: Only if all ranks provide a True value should all ranks receive True.
            3. rank_zero: Makes rank 0 process's value the source of truth and broadcasts the result to all other processes.
            4. If an integer N is provided, return True only if at least N processes provide a True value.
            5. If a float F is provided, return True only if at least this ratio of processes provide a True value. The ratio provided should be in the range [0, 1].

    Returns:
        The synchronized boolean value.

    Example::

        >>> val = True
        >>> # synced_val is True iff all ranks provide a True value to the function
        >>> synced_val = sync_bool(val, coherence_mode="all")
        >>> if synced_val:
        >>>     print("success")

    """
    if not dist.is_available() or not dist.is_initialized():
        return val

    pg = pg or dist.group.WORLD
    device = torch.device(
        torch.cuda.current_device() if dist.get_backend(pg) == "nccl" else "cpu"
    )
    pg_wrapper = PGWrapper(pg)

    dtype = torch.uint8
    if pg_wrapper.get_world_size() > 256:
        dtype = torch.int

    indicator = (
        torch.ones(1, device=device, dtype=dtype)
        if val
        else torch.zeros(1, device=device, dtype=dtype)
    )

    if coherence_mode == "rank_zero":
        # Broadcast from rank 0 to all other ranks
        dist.broadcast(indicator, src=0, group=pg)
        return bool(indicator[0].item())
    elif coherence_mode == "any":
        # sum up the indicators across all the ranks.
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return indicator.item() > 0
    elif coherence_mode == "all":
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return indicator.item() == pg_wrapper.get_world_size()
    elif isinstance(coherence_mode, int):
        # if >= int(coherence_mode) processes signal to stop, all processes stop
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return indicator.item() >= coherence_mode
    elif isinstance(coherence_mode, float):
        dist.all_reduce(indicator, op=dist.ReduceOp.SUM)
        return (indicator.item() / pg_wrapper.get_world_size()) >= coherence_mode
    else:
        raise TypeError(
            f'Invalid value for `coherence_mode` provided: Expected type int, float, or one of ("any", "all", "rank_zero"), but received {coherence_mode}.'
        )
