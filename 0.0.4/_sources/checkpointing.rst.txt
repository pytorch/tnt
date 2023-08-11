Checkpointing
================================

TorchTNT offers checkpointing via the :class:`~torchtnt.framework.callbacks.TorchSnapshotSaver` which uses `TorchSnapshot <https://pytorch.org/torchsnapshot/main/>`_ under the hood.

.. code-block:: python

    module = nn.Linear(input_dim, 1)
    unit = MyUnit(module=module)
    tss = TorchSnapshotSaver(
        dirpath=your_dirpath_here,
        save_every_n_train_steps=100,
        save_every_n_epochs=2,
    )
    # loads latest checkpoint, if it exists
    if latest_checkpoint_dir:
        tss.restore_from_latest(your_dirpath_here, unit, train_dataloader=dataloader)
    train(
        unit,
        dataloader,
        callbacks=[tss]
    )

There is built-in support for saving and loading distributed models (DDP, FSDP).

The state dict type to be used for checkpointing FSDP modules can be specified in the :class:`~torchtnt.utils.prepare_module.FSDPStrategy`'s state_dict_type argument like so:

.. code-block:: python

    module = nn.Linear(input_dim, 1)
    fsdp_strategy = FSDPStrategy(
        # sets state dict type of FSDP module
        state_dict_type=STATE_DICT_TYPE.SHARDED_STATE_DICT
    )
    module = prepare_fsdp(module, strategy=fsdp_strategy)
    unit = MyUnit(module=module)
    tss = TorchSnapshotSaver(
        dirpath=your_dirpath_here,
        save_every_n_epochs=2,
    )
    train(
        unit,
        dataloader,
        # checkpointer callback will use state dict type specified in FSDPStrategy
        callbacks=[tss]
    )

Or you can manually set this using `FSDP.set_state_dict_type <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type>`_.

.. code-block:: python

    module = nn.Linear(input_dim, 1)
    module = FSDP(module, ....)
    FSDP.set_state_dict_type(module, StateDictType.SHARDED_STATE_DICT)
    unit = MyUnit(module=module, ...)
    tss = TorchSnapshotSaver(
        dirpath=your_dirpath_here,
        save_every_n_epochs=2,
    )
    train(
        unit,
        dataloader,
        callbacks=[tss]
    )
