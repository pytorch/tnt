# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from uuid import uuid4

import pytest
import torch

from torch.utils.data import Dataset
from torchtnt.data.dataloaders import StatefulDataLoader
from torchtnt.data.samplers import StatefulDistributedSampler


class SimpleDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


@pytest.fixture
def temp_dir():
    temp_dir = tempfile.TemporaryDirectory()
    yield temp_dir.name
    temp_dir.cleanup()


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("prefetch_factor", [2, 4])
@pytest.mark.parametrize("break_index", [0, 5])
@pytest.mark.parametrize("persistent_workers", [True, False])
def test_single_rank(
    temp_dir: str,
    shuffle: bool,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    break_index: int,
    persistent_workers: bool,
) -> None:
    # num_workers > 0 when using prefetch_factor or persistent_workers.
    # prefetch_factor is default 2 in DataLoader.
    if num_workers == 0 and (prefetch_factor != 2 or persistent_workers):
        return

    items = list((i, i * 10) for i in range(100))
    dataset = SimpleDataset(items)

    def create_sampler():
        return StatefulDistributedSampler(
            dataset, shuffle=shuffle, seed=1, rank=0, world_size=1
        )

    def create_loader():
        return StatefulDataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    sampler = create_sampler()
    loader = create_loader()
    save_file = os.path.join(temp_dir, str(uuid4()))

    seen_batches = []
    for i, batch in enumerate(loader):
        seen_batches.extend(list(zip(batch[0].tolist(), batch[1].tolist())))
        # simulate a pre-emption
        if i == break_index:
            break

    torch.save(loader.state_dict(), save_file)

    # recreate loader and sampler to simulate a new process (i.e. resuming from checkpoint)
    sampler = create_sampler()
    loader = create_loader()
    loader.load_state_dict(torch.load(save_file))

    for i, batch in enumerate(loader):
        seen_batches.extend(list(zip(batch[0].tolist(), batch[1].tolist())))

    if shuffle:
        seen_batches.sort(key=lambda x: x[0])

    assert seen_batches == items


@pytest.mark.parametrize("world_size", [1, 2, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("prefetch_factor", [2, 4])
@pytest.mark.parametrize("break_index", [0, 5])
@pytest.mark.parametrize("persistent_workers", [True, False])
def test_distributed(
    temp_dir: str,
    world_size: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    break_index: int,
    persistent_workers: bool,
) -> None:
    # num_workers > 0 when using prefetch_factor or persistent_workers.
    # prefetch_factor is default 2 in DataLoader.
    if num_workers == 0 and (prefetch_factor != 2 or persistent_workers):
        return

    items = list((i, i * 10) for i in range(100))

    dataset = SimpleDataset(items)
    save_file = os.path.join(temp_dir, str(uuid4()))

    def create_sampler(rank):
        StatefulDistributedSampler(
            dataset, shuffle=True, seed=1, rank=rank, world_size=world_size
        )

    def create_loader(sampler):
        return StatefulDataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    loaders = []
    for i in range(world_size):
        sampler = create_sampler(i)
        loader = create_loader(sampler)
        loaders.append(loader)

    seen_batches = []
    for loader in loaders:
        for i, batch in enumerate(loader):
            seen_batches.extend(list(zip(batch[0].tolist(), batch[1].tolist())))
            # simulate a pre-emption
            if i == break_index:
                break

    # simulate save on rank 0
    torch.save(loaders[0].state_dict(), save_file)
    loaders = []
    for i in range(world_size):
        sampler = create_sampler(i)
        loader = create_loader(sampler)
        loader.load_state_dict(torch.load(save_file))
        loaders.append(loader)

    for loader in loaders:
        for i, batch in enumerate(loader):
            seen_batches.extend(list(zip(batch[0].tolist(), batch[1].tolist())))

    seen_batches.sort(key=lambda x: x[0])
    assert seen_batches == items
