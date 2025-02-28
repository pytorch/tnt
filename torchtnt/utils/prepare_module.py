# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import asdict, dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    cast,
    ContextManager,
    Dict,
    Iterable,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from torch.distributed._composable_state import _get_module_state
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh

try:
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        fully_shard,
        MixedPrecisionPolicy,
    )
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
except ImportError:

    def noop(*args: Any, **kwargs: Any) -> None:
        pass

    class NOOP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    fully_shard = noop
    MixedPrecisionPolicy = NOOP
    CPUOffloadPolicy = NOOP
    FSDPState = NOOP

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType as _StateDictType,
)
from torch.distributed.fsdp._common_utils import _FSDPState
from torch.distributed.fsdp.api import OptimStateDictConfig, StateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch as _BackwardPrefetch,
    CPUOffload,
    MixedPrecision as _MixedPrecision,
    ShardingStrategy as _ShardingStrategy,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtnt.utils.fsdp_utils import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)

from torchtnt.utils.rank_zero_log import rank_zero_info, rank_zero_warn
from torchtnt.utils.version import is_torch_version_geq


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
    sync_batchnorm: bool = False


@dataclass
class FSDPStrategy(Strategy):
    """Dataclass representing the `FullyShardedDataParallel <https://pytorch.org/docs/stable/fsdp.html>`_ strategy"""

    process_group: Optional[ProcessGroup] = None
    sharding_strategy: Optional[Union[str, _ShardingStrategy]] = None
    cpu_offload: Optional[CPUOffload] = None
    auto_wrap_policy: Optional[Callable[[torch.nn.Module, bool, int], bool]] = None
    backward_prefetch: Optional[Union[str, _BackwardPrefetch]] = (
        _BackwardPrefetch.BACKWARD_PRE
    )
    mixed_precision: Optional[Union[_MixedPrecision, MixedPrecision]] = None
    ignored_modules: Optional[Iterable[torch.nn.Module]] = None
    param_init_fn: Optional[Callable[[torch.nn.Module], None]] = None
    sync_module_states: bool = False
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    use_orig_params: bool = False

    # FSDP set_state_dict_type params: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type
    # for setting type of state dict for checkpointing
    state_dict_type: Optional[Union[str, _StateDictType]] = None
    state_dict_config: Optional[StateDictConfig] = None
    optim_state_dict_config: Optional[OptimStateDictConfig] = None

    def __post_init__(self) -> None:
        if isinstance(self.sharding_strategy, str):
            self.sharding_strategy = ShardingStrategy.to_native_sharding_strategy(
                self.sharding_strategy
            )

        if isinstance(self.backward_prefetch, str):
            self.backward_prefetch = BackwardPrefetch.to_native_backward_prefetch(
                self.backward_prefetch
            )

        if isinstance(self.state_dict_type, str):
            self.state_dict_type = StateDictType.to_native_state_dict_type(
                self.state_dict_type
            )

        if isinstance(self.mixed_precision, MixedPrecision):
            self.mixed_precision = self.mixed_precision.to_native_mixed_precision()


@dataclass
class FSDP2Strategy(Strategy):
    """
    Dataclass representing the `FSDP2 <https://pytorch.org/docs/2.6/distributed.fsdp.fully_shard.html>`_ strategy.
    For more details on the args, see the link.

    Args:
        modules_to_shard: A list of modules that should be sharded across devices. Options are 'all' to shard all submodules, or a list of module names/module types.
        reshard_after_forward: If True, reshards parameters after the forward pass to optimize memory usage.
        mp_policy: Controls mixed precision policy. If only dtype is provided, it will be used to cast all relevant parts of model. If None, no mixed precision is used
        cpu_offload: If True, enables CPU offloading of model parameters to reduce GPU memory usage.

    Note:
        It is recommended to specify specific modules to shard to avoid unnecessary sharding of all submodules, which has
        communication overhead.

    Example:
        >>> model
            TransformerDecoder(
                (tok_embeddings): Embedding(128256, 4096)
                (layers): ModuleList(
                    (0-31): 32 x TransformerSelfAttentionLayer(
                    (attn): MultiHeadAttention(
                        (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
                        (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
                        (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
                        (output_proj): Linear(in_features=4096, out_features=4096, bias=False)
                        (pos_embeddings): RotaryPositionalEmbeddings()
                    )
                    ...
                )
                (output): Linear(in_features=4096, out_features=128256, bias=False)
            )
        >>> # You can either specify the module to shard as a name ("Linear") or the module type (torch.nn.Linear)
        >>> strategy = FSDP2Strategy(modules_to_shard=["TransformerSelfAttentionLayer", "Linear"])
    """

    modules_to_shard: Union[
        Literal["all"],
        Iterable[Union[str, Type[torch.nn.Module]]],
    ] = "all"
    reshard_after_forward: Union[bool, int] = True
    mp_policy: Optional[Union[torch.dtype, MixedPrecisionPolicy]] = None
    cpu_offload: bool = False


@dataclass
class TorchCompileParams:
    """
    Dataclass to store parameters for torch compile. See https://pytorch.org/docs/stable/generated/torch.compile.html for details.
    """

    fullgraph: bool = False
    dynamic: bool = False
    # pyre-ignore: Invalid type parameters. Uses PyTorch types.
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
        auto_wrap_policy A policy to wrap model's submodules with AC. Note that if this is specified, it takes precedence over ``check_fn``.
    """

    checkpoint_impl: CheckpointImpl
    check_fn: Callable[[torch.nn.Module], bool] = lambda _: True
    auto_wrap_policy: Optional[Callable[[torch.nn.Module, bool, int], bool]] = None
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    context_fn: Optional[Callable[[], Tuple[ContextManager, ContextManager]]] = None


def prepare_ddp(
    module: torch.nn.Module,
    device: torch.device,
    strategy: Optional[DDPStrategy] = None,
) -> DDP:
    """
    Utility to move a module to device and wrap in `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_.

    Args:
        module: module to be wrapped in DDP. If module has params on meta device, they will be materialized on the device prior to DDP wrapping
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

    materialize_meta_params(module, device)

    # now move rest of module to device
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


def prepare_fsdp2(
    module: torch.nn.Module,
    device: torch.device,
    strategy: Optional[FSDP2Strategy] = None,
    process_group: Optional[ProcessGroup] = None,
) -> torch.nn.Module:
    """
    Utility to move a module to device and wrap in `FSDP2 <https://pytorch.org/docs/2.6/distributed.fsdp.fully_shard.html>`_

    Args:
        module: module to be wrapped in FSDP
        device: device to which module will be moved
        strategy: an instance of :class:`~torchtnt.utils.prepare_module.FSDP2Strategy` which defines the settings of FSDP APIs
    """
    strategy = strategy or FSDP2Strategy()

    # prepare kwargs for fully_shard api
    pg = process_group or dist.distributed_c10d._get_default_group()
    mesh = init_device_mesh(device.type, mesh_shape=(pg.size(),))
    fsdp_kwargs: Dict[str, Any] = {
        "mesh": mesh,  # TODO we only configure 1D mesh for now, look into supporting HSDP
        "reshard_after_forward": strategy.reshard_after_forward,
    }
    if strategy.cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
    if (mp_policy := strategy.mp_policy) is not None:
        if isinstance(mp_policy, MixedPrecisionPolicy):
            fsdp_kwargs["mp_policy"] = mp_policy
        else:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=mp_policy,
                reduce_dtype=mp_policy,
                output_dtype=mp_policy,
                cast_forward_inputs=True,
            )

    # parse out the modules_to_shard argument
    modules_to_shard = strategy.modules_to_shard

    shard_all = modules_to_shard == "all"
    shard_module_names: Set[str] = set()
    shard_module_types: Tuple[Type[torch.nn.Module], ...] = ()
    if not shard_all:
        assert (
            type(modules_to_shard) is not str
        ), f"modules_to_shard must be an iterable of modules or 'all', got {shard_all}"

        for item in modules_to_shard:
            if isinstance(item, str):
                shard_module_names.add(item)
            else:
                shard_module_types = shard_module_types + (item,)

    # apply the fsdp2 sharding bottoms up
    num_layers_sharded = 0
    for _, m in reversed(list(module.named_modules())):
        if shard_all:
            # fully_shard does not support containers that do not implement forward
            if not isinstance(m, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                fully_shard(m, **fsdp_kwargs)
                num_layers_sharded += 1
        elif (
            isinstance(m, shard_module_types) or type(m).__name__ in shard_module_names
        ):
            # if m exists in shard_module_types, then shard it
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError(
            "No layer modules were sharded with fsdp2. Please check if shard conditions are working as expected."
        )

    # shard the top level model, so that all params are moved off cpu to gpu
    if not _is_fsdp_module(module):
        fully_shard(module, **fsdp_kwargs)

    # materialized sharded meta weights to device
    materialize_meta_params(module, device)

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


class FSDP2OptimizerWrapper:
    """
    Wrapper for FSDP2 optimizer which uses distributed state dict APIs.
    """

    def __init__(
        self, module: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        self.module = module
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        return get_optimizer_state_dict(self.module, self.optimizer)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_optimizer_state_dict(self.module, self.optimizer, state_dict)


def _is_fsdp_module(module: torch.nn.Module) -> bool:
    if isinstance(module, FSDP):
        return True

    # Also check for composable FSDP API
    maybe_composable_state = _get_module_state(module)
    if maybe_composable_state is not None:
        return isinstance(maybe_composable_state, (_FSDPState, FSDPState))

    return False


def _is_fsdp2_module(module: torch.nn.Module) -> bool:
    maybe_composable_state = _get_module_state(module)
    if maybe_composable_state is not None:
        return isinstance(maybe_composable_state, FSDPState)

    return False


def prepare_module(
    module: torch.nn.Module,
    device: torch.device,
    *,
    strategy: Optional[Union[Strategy, str]] = None,
    torch_compile_params: Optional[TorchCompileParams] = None,
    activation_checkpoint_params: Optional[ActivationCheckpointParams] = None,
    enable_compiled_autograd: bool = False,
) -> torch.nn.Module:
    """
    Utility to move a module to device, set up parallelism, activation checkpointing and compile.

    Args:
        module: module to be used.
        device: device to which module will be moved.
        strategy: the data parallelization strategy to be used. if a string, must be one of ``ddp``, ``fsdp``, or ``noop``.
        torch_compile_params: params for Torch compile https://pytorch.org/docs/stable/generated/torch.compile.html.
        activation_checkpoint_params: params for enabling activation checkpointing.
        enable_compiled_autograd: if True, `compiled_autograd` will be used to compile the backward, this is an experimental flag.
    """

    if strategy:
        if not isinstance(strategy, str) and not isinstance(strategy, Strategy):
            raise ValueError(
                f"Unknown strategy received: {strategy}. Expect either str (one of 'ddp', 'fsdp', or 'noop') or Strategy dataclass"
            )

        if isinstance(strategy, str):
            strategy = convert_str_to_strategy(strategy)
        if isinstance(strategy, DDPStrategy):
            if (
                torch_compile_params
                and strategy.static_graph is True
                and not is_torch_version_geq("2.1.0")
            ):
                raise RuntimeError(
                    "Torch version >= 2.1.0 required for Torch compile + DDP with static graph"
                )

            if enable_compiled_autograd:
                if not torch_compile_params:
                    raise RuntimeError(
                        "Compiled autograd should only be used when the module is compiled."
                    )
                try:
                    from torch._dynamo.trace_rules import LEGACY_MOD_INLINELIST

                    LEGACY_MOD_INLINELIST.add("torch.nn.parallel.distributed")
                except ImportError:
                    pass
                # This has to be set before DDP wrapping
                torch._dynamo.config.optimize_ddp = "python_reducer"
            module = prepare_ddp(module, device, strategy)
        elif isinstance(strategy, FSDPStrategy):
            if torch_compile_params and strategy.use_orig_params is False:
                # as stated here https://pytorch.org/get-started/pytorch-2.0/
                raise RuntimeError(
                    "Torch compile requires FSDPStrategy's use_orig_params to be True, since AOTAutograd needs to be aware of the original parameters"
                )
            module = prepare_fsdp(module, device, strategy)
        elif isinstance(strategy, FSDP2Strategy):
            module = prepare_fsdp2(module, device, strategy)
    else:
        module = module.to(device)

    if activation_checkpoint_params:
        checkpoint_impl = activation_checkpoint_params.checkpoint_impl
        check_fn = activation_checkpoint_params.check_fn
        auto_wrap_policy = activation_checkpoint_params.auto_wrap_policy
        context_fn = activation_checkpoint_params.context_fn
        additional_params = {}
        if context_fn:
            additional_params["context_fn"] = context_fn
        custom_checkpoint_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=checkpoint_impl,
            **additional_params,
        )
        apply_activation_checkpointing(
            module,
            checkpoint_wrapper_fn=custom_checkpoint_wrapper,
            check_fn=check_fn,
            auto_wrap_policy=auto_wrap_policy,
        )

    if torch_compile_params:
        try:
            # use in-place compile to avoid altering the state_dict keys
            module.compile(**asdict(torch_compile_params))
        except AttributeError:
            rank_zero_warn(
                "Please install PyTorch nightlies to use in-place compile to avoid altering the state_dict keys when checkpointing."
            )
            return cast(
                torch.nn.Module, torch.compile(module, **asdict(torch_compile_params))
            )

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


def on_meta_device(module: torch.nn.Module) -> bool:
    try:
        return next(module.parameters(recurse=False)).device.type == "meta"
    except StopIteration:
        return False


def materialize_meta_params(module: torch.nn.Module, device: torch.device) -> None:
    """
    Materialize meta device parameters to the given device.

    Args:
        module: module to be used.
        device: device to which module will be moved.
    """
    for name, submodule in module.named_modules():
        if on_meta_device(submodule):
            rank_zero_info(f"{name} is on meta device, intializing on device {device}")
            submodule.to_empty(device=device, recurse=False)
