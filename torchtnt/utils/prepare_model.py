# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.api import OptimStateDictConfig, StateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtnt.utils.rank_zero_log import rank_zero_warn
from torchtnt.utils.version import is_torch_version_geq_1_12


@dataclass
class Strategy:
    """Dataclass representing a parallelization strategy"""

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


def prepare_ddp(
    module: torch.nn.Module,
    device: torch.device,
    strategy: Optional[DDPStrategy] = None,
) -> DDP:
    """
    Utility to move a module to device and wrap in `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_.

    Args:
        module: module to be wrapped in DDP
        strategy: an instance of DDPStrategy which defines the settings of DDP APIs
        device: device to which module will be moved

    Examples::
        strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
        module = nn.Linear(1, 1)
        device = torch.device("cuda")
        ddp_module = prepare_ddp(module, strategy, device)
    """
    strategy = strategy if strategy is not None else DDPStrategy()
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
    if strategy.comm_hook:
        module.register_comm_hook(state=strategy.comm_state, hook=strategy.comm_hook)
    return module


def prepare_fsdp(
    module: torch.nn.Module,
    device: torch.device,
    strategy: Optional[FSDPStrategy] = None,
    precision: Optional[torch.dtype] = None,
) -> FSDP:
    """
    Utility to move a module to device and wrap in `FullyShardedDataParallel <https://pytorch.org/docs/stable/fsdp.html>`_.

    Args:
        module: module to be wrapped in FSDP
        strategy: an instance of FSDPStrategy which defines the settings of FSDP APIs
        device: device to which module will be moved
        precision: precision to use when wrapping in FSDP

    Examples::
        strategy = FSDPStrategy(limit_all_gathers=True)
        module = nn.Linear(1, 1)
        device = torch.device("cuda")
        fsdp_module = prepare_fsdp(module, strategy, device)
    """
    if not is_torch_version_geq_1_12():
        raise RuntimeError(
            "Please install PyTorch 1.12 or higher to use FSDP: https://pytorch.org/get-started/locally/"
        )
    strategy = strategy if strategy is not None else FSDPStrategy()
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
