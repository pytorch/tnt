Distributed training
================================

The core TNT framework makes no assumptions about distributed training or devices, and expects the user to handle configuring distributed training on their own. As a convenience, the framework offers the :class:`~torchtnt.framework.auto_unit.AutoUnit` for users who prefer for this to be handled automatically. The framework-provided checkpointing callbacks handle distributed model checkpointing and loading.

If you are using the the :class:`~torchtnt.framework.unit.TrainUnit`/ :class:`~torchtnt.framework.unit.EvalUnit`/ :class:`~torchtnt.framework.unit.PredictUnit` interface, you are expected to initialize the CUDA device, if applicable, along with the global process group from torch.distributed. We offer a convenience function :py:func:`~torchtnt.utils.init_from_env` that works with TorchElastic to automatically handle these settings for you, which you should invoke at the beginning of your script.


Distributed Data Parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using the the :class:`~torchtnt.framework.unit.TrainUnit`/ :class:`~torchtnt.framework.unit.EvalUnit`/ :class:`~torchtnt.framework.unit.PredictUnit` interface, `DDP <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ can be simply be wrapped around your model like so:

.. code-block:: python

    device = init_from_env()
    module = nn.Linear(input_dim, 1)
    # move module to device
    module = module.to(device)
    # wrap module in DDP
    device_ids = [device.index]
    model = torch.nn.parallel.DistributedDataParallel(module, device_ids=device_ids)

We also offer :py:func:`~torchtnt.utils.prepare_module.prepare_ddp` which can assist in wrapping the model for you.

The :class:`~torchtnt.framework.auto_unit.AutoUnit` automatically wraps the module in DDP when either

1. The string ``ddp`` is passed in the strategy argument

    .. code-block:: python

        module = nn.Linear(input_dim, 1)
        my_auto_unit = MyAutoUnit(module=module, strategy="ddp")

2. The dataclass :class:`~torchtnt.utils.prepare_module.DDPStrategy` is passed in to the strategy argument. This is helpful when wanting to customize the settings in DDP

    .. code-block:: python

        module = nn.Linear(input_dim, 1)
        ddp_strategy = DDPStrategy(broadcast_buffers=False, check_reduction=True)
        my_auto_unit = MyAutoUnit(module=module, strategy=ddp_strategy)


Fully Sharded Data Parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If using one or more of or :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, or :class:`~torchtnt.framework.unit.PredictUnit`, `FSDP <https://pytorch.org/docs/stable/fsdp.html>`_ can be simply be wrapped around the model like so:

.. code-block:: python

    device = init_from_env()
    module = nn.Linear(input_dim, 1)
    # move module to device
    module = module.to(device)
    # wrap module in FSDP
    model = torch.distributed.fsdp.FullyShardedDataParallel(module, device_id=device)

We also offer :py:func:`~torchtnt.utils.prepare_module.prepare_fsdp` which can assist in wrapping the model for you.

The :class:`~torchtnt.framework.auto_unit.AutoUnit` automatically wraps the module in FSDP when either

1. The string ``fsdp`` is passed in the strategy argument

    .. code-block:: python

        module = nn.Linear(input_dim, 1)
        my_auto_unit = MyAutoUnit(module=module, strategy="fsdp")

2. The dataclass :class:`~torchtnt.utils.prepare_module.FSDPStrategy` is passed in to the strategy argument. This is helpful when wanting to customize the settings in FSDP

    .. code-block:: python

        module = nn.Linear(input_dim, 1)
        fsdp_strategy = FSDPStrategy(forward_prefetch=True, limit_all_gathers=True)
        my_auto_unit = MyAutoUnit(module=module, strategy=fsdp_strategy)
