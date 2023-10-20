# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, Optional, Union

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
from torchtnt.utils.version import is_torch_version_geq_1_12, is_torch_version_geq_2_0

if is_torch_version_geq_2_0():
    from torch.distributed._composable_state import _get_module_state
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
        CheckpointImpl,
    )
    from torch.distributed.fsdp._common_utils import _FSDPState


@dataclass
class Strategy:
    """Dataclass representing a parallelization strategy"""

    pass


@dataclass
class NOOPStrategy(Strategy):
    """
    Dataclass representing a no-op strategy. Nothing is applied to the module, and no device transfer occurs
    Use this strategy if applying custom wrapping to module prior to passing it into class:`~torchtnt.framework.auto_unit.AutoUnit`
    or into :py:func:`~torchtnt.utils.prepare_module.prepare_module`
    """

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
    backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE
    mixed_precision: Optional[MixedPrecision] = None
    ignored_modules: Optional[Iterable[torch.nn.Module]] = None
    param_init_fn: Optional[Callable[[torch.nn.Module], None]] = None
    sync_module_states: bool = False
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    use_orig_params: bool = False

    # FSDP set_state_dict_type params: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type
    # for setting type of state dict for checkpointing
    state_dict_type: Optional[StateDictType] = None
    state_dict_config: Optional[StateDictConfig] = None
    optim_state_dict_config: Optional[OptimStateDictConfig] = None


@dataclass
class TorchCompileParams:
    """
    Dataclass to store parameters for torch compile. See https://pytorch.org/docs/stable/generated/torch.compile.html for details.
    """

    fullgraph: bool = False
    dynamic: bool = False
    # pyre-ignore: Invalid type parameters [24]
    backend: Union[str, Callable] = "inductor"
    mode: Union[str, None] = None
    options: Optional[Dict[str, Union[str, int, bool]]] = None
    disable: bool = False


@dataclass
class ActivationCheckpointParams:
    """
    Dataclass to store parameters for activation checkpointing.

    Args:
        checkpoint_impl: type of checkpointing implementation to use
        check_fn: A lambda function which will be passed to each child submodule and return ``True`` or ``False`` depending on whether the submodule should be wrapped.
    """

    checkpoint_impl: "CheckpointImpl"
    check_fn: Optional[Callable[[torch.nn.Module], bool]]


def prepare_ddp(
    module: torch.nn.Module,
    device: torch.device,
    strategy: Optional[DDPStrategy] = None,
) -> DDP:
    """
    Utility to move a module to device and wrap in `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_.

    Args:
        module: module to be wrapped in DDP
        device: device to which module will be moved
        strategy: an instance of :class:`~torchtnt.utils.prepare_module.DDPStrategy` which defines the settings of DDP APIs

    Examples::
        strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
        module = nn.Linear(1, 1)
        device = torch.device("cuda")
        ddp_module = prepare_ddp(module, device, strategy)
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
) -> FSDP:
    """
    Utility to move a module to device and wrap in `FullyShardedDataParallel <https://pytorch.org/docs/stable/fsdp.html>`_.

    Args:
        module: module to be wrapped in FSDP
        device: device to which module will be moved
        strategy: an instance of :class:`~torchtnt.utils.prepare_module.FSDPStrategy` which defines the settings of FSDP APIs

    Examples::
        strategy = FSDPStrategy(limit_all_gathers=True)
        module = nn.Linear(1, 1)
        device = torch.device("cuda")
        fsdp_module = prepare_fsdp(module, device, strategy)
    """
    if not is_torch_version_geq_1_12():
        raise RuntimeError(
            "Please install PyTorch 1.12 or higher to use FSDP: https://pytorch.org/get-started/locally/"
        )
    strategy = strategy if strategy is not None else FSDPStrategy()

    # we use __dict__ and not asdict() here because asdict() is recursively applied on nested objects
    params_dict = strategy.__dict__.copy()

    # extract params to set state dict type
    state_dict_type = params_dict.pop("state_dict_type")
    state_dict_config = params_dict.pop("state_dict_config")
    optim_state_dict_config = params_dict.pop("optim_state_dict_config")

    # wrap module in FSDP
    module = FSDP(
        module,
        device_id=device,
        **params_dict,
    )

    if state_dict_type:
        FSDP.set_state_dict_type(
            module, state_dict_type, state_dict_config, optim_state_dict_config
        )
    return module


class FSDPOptimizerWrapper:
    """
    Wrapper for FSDP optimizer to call specific FSDP optimizer state checkpointing APIs.
    """

    def __init__(
        self, module: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        self.module = module
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        optim_state_dict = FSDP.optim_state_dict(self.module, self.optimizer)
        return optim_state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        optim_state_dict = FSDP.optim_state_dict_to_load(
            self.module, self.optimizer, state_dict
        )
        self.optimizer.load_state_dict(optim_state_dict)


def _is_fsdp_module(module: torch.nn.Module) -> bool:
    if isinstance(module, FSDP):
        return True

    if is_torch_version_geq_2_0():
        # Also check for composable FSDP API
        maybe_composable_state = _get_module_state(module)
        if maybe_composable_state is not None:
            return isinstance(maybe_composable_state, _FSDPState)

    return False


def prepare_module(
    module: torch.nn.Module,
    device: torch.device,
    *,
    strategy: Optional[Union[Strategy, str]] = None,
    torch_compile_params: Optional[TorchCompileParams] = None,
    activation_checkpoint_params: Optional[ActivationCheckpointParams] = None,
) -> torch.nn.Module:
    """
    Utility to move a module to device, set up parallelism, activation checkpointing and compile.

    Args:
        module: module to be used.
        device: device to which module will be moved.
        strategy: the data parallelization strategy to be used. if a string, must be one of ``ddp``, ``fsdp``, or ``noop``.
        torch_compile_params: params for Torch compile https://pytorch.org/docs/stable/generated/torch.compile.html.
        activation_checkpoint_params: params for enabling activation checkpointing.
    """

    if strategy:
        if isinstance(strategy, str):
            strategy = convert_str_to_strategy(strategy)
        if isinstance(strategy, DDPStrategy):
            if torch_compile_params and strategy.static_graph is True:
                # https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860
                raise RuntimeError(
                    "Torch compile requires DDPStrategy's static_graph to be False"
                )
            module = prepare_ddp(module, device, strategy)
        elif isinstance(strategy, FSDPStrategy):
            if torch_compile_params and strategy.use_orig_params is False:
                # as stated here https://pytorch.org/get-started/pytorch-2.0/
                raise RuntimeError(
                    "Torch compile requires FSDPStrategy's use_orig_params to be True, since AOTAutograd needs to be aware of the original parameters"
                )
            module = prepare_fsdp(module, device, strategy)
    else:
        module = module.to(device)

    if activation_checkpoint_params:
        if not is_torch_version_geq_2_0():
            raise RuntimeError("Activation checkpointing requires torch>=2.0")
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

    if torch_compile_params:
        try:
            # use in-place compile to avoid altering the state_dict keys
            module.compile(**asdict(torch_compile_params))
        except AttributeError:
            rank_zero_warn(
                "Please install PyTorch nightlies to use in-place compile to avoid altering the state_dict keys when checkpointing."
            )
            torch.compile(module, **asdict(torch_compile_params))

    return module


def convert_str_to_strategy(
    strategy: str,
) -> Union[DDPStrategy, FSDPStrategy, NOOPStrategy]:
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
        "noop": NOOPStrategy(),
    }

    if strategy not in string_to_strategy_mapping:
        raise ValueError(
            f"Strategy {strategy} not supported. Please use one of {list(string_to_strategy_mapping.keys())}"
        )
    return string_to_strategy_mapping[strategy]
