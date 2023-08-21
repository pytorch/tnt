# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import fsspec
import torch
from torch.utils.data import Dataset

"""
Adapted from https://github.com/karpathy/minGPT/blob/master/projects/chargpt/chargpt.py
"""


@dataclass
class DataConfig:
    # pyre-fixme[8]: Attribute has type `str`; used as `None`.
    path: str = None
    # pyre-fixme[8]: Attribute has type `int`; used as `None`.
    block_size: int = None
    # pyre-fixme[8]: Attribute has type `float`; used as `None`.
    train_split: float = None
    truncate: float = 1.0


class CharDataset(Dataset):
    # pyre-fixme[3]: Return type must be annotated.
    def __init__(self, data_cfg: DataConfig):
        print(data_cfg.path)
        data = fsspec.open(data_cfg.path).open().read().decode("utf-8")
        data = data[: int(len(data) * data_cfg.truncate)]

        chars = sorted(set(data))
        data_size, vocab_size = len(data), len(chars)
        print("Data has %d characters, %d unique." % (data_size, vocab_size))

        # pyre-fixme[4]: Attribute must be annotated.
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        # pyre-fixme[4]: Attribute must be annotated.
        self.itos = {i: ch for i, ch in enumerate(chars)}
        # pyre-fixme[4]: Attribute must be annotated.
        self.block_size = data_cfg.block_size
        # pyre-fixme[4]: Attribute must be annotated.
        self.vocab_size = vocab_size
        # pyre-fixme[4]: Attribute must be annotated.
        self.data = data

    # pyre-fixme[3]: Return type must be annotated.
    def __len__(self):
        return len(self.data) - self.block_size

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
