# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch.utils.data import Dataset, Sampler
from torchtnt.utils import distributed as dist_utils


class StatefulDistributedSampler(Sampler):
    """
    StatefulDistributedSampler is a sampler whose state can be saved and loaded.
    It is expected to be used with ``torchtnt.data.dataloaders.StatefulDataLoader``.

    It can be used in a single or multi-process setup.

    Args:
        dataset (Dataset): dataset to sample from
        seed (int): seed for random number generator
        rank (int): rank of the current process. If left empty, it will automatically be determined.
        shuffle (bool): whether to shuffle the dataset
        drop_last (bool): whether to drop the last incomplete batch
        replacement (bool): whether to sample with replacement. If True, `drop_last` must be False.
    """

    def __init__(
        self,
        dataset: Dataset,
        seed: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = True,
        replacement: bool = False,
    ):
        self._dataset = dataset

        if seed is None:
            seed = torch.randint(0, 2**32, (1,), dtype=torch.int64).item()
        self._seed = seed

        if rank is None:
            rank = dist_utils.get_global_rank()
        self._rank = rank

        if world_size is None:
            world_size = dist_utils.get_world_size()
        self._world_size = world_size

        self._shuffle = shuffle
        self._drop_last = drop_last
        self._replacement = replacement
        self._epoch = 0
        self._generator_state = None
        self._start_index = 0

    @property
    def epoch(self):
        return self._epoch

    @property
    def world_size(self):
        return self._world_size

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)
        if self._shuffle:
            indices = torch.randperm(len(self._dataset), generator=g).tolist()
        else:
            indices = list(range(len(self._dataset)))
        # TODO implement drop last / even number of replicas
        indices = indices[self._rank + self._start_index :: self._world_size]
        return iter(indices)

    def state_dict(self):
        return {"epoch": self._epoch, "seed": self._seed}

    def load_state_dict(self, state_dict):
        self._epoch = state_dict["epoch"]
        self._seed = state_dict["seed"]
        # set by StatefulDataloader when loading from checkpoint
        self._start_index = state_dict.get("start_index", 0)
