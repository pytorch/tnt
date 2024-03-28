#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Tuple

import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from torchtnt.utils.data.data_prefetcher import CudaDataPrefetcher

Batch = Tuple[torch.Tensor, torch.Tensor]


class DataPrefetcherTest(unittest.TestCase):
    def _generate_dataset(self, num_samples: int, input_dim: int) -> Dataset[Batch]:
        """Returns a dataset of random inputs and labels for binary classification."""
        data = torch.randn(num_samples, input_dim)
        labels = torch.randint(low=0, high=2, size=(num_samples,))
        return TensorDataset(data, labels)

    def test_cpu_device_data_prefetcher(self) -> None:
        device = torch.device("cpu")

        num_samples = 12
        batch_size = 4
        dataloader = torch.utils.data.DataLoader(
            self._generate_dataset(num_samples, 2), batch_size=batch_size
        )

        num_prefetch_batches = 2
        with self.assertRaisesRegex(ValueError, "expects a CUDA device"):
            _ = CudaDataPrefetcher(dataloader, device, num_prefetch_batches)
