#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest
from collections import Counter
from typing import Any, Dict, List, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

from torchtnt.data.iterators import (
    AllDatasetBatches,
    DataIterationStrategy,
    InOrder,
    MultiIterator,
    RandomizedBatchSampler,
    RoundRobin,
    StoppingMechanism,
)
from torchtnt.data.multi_dataloader import MultiDataloader


class RandomDataset(Dataset[torch.Tensor]):
    def __init__(self, size: int, length: int) -> None:
        self.len: int = length
        self.data: torch.Tensor = torch.randn(length, size)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.len


class CustomRandomIterator(MultiIterator):
    """A test iterator.

    This returns batch from a random selected dataloader, until
    all dataloaders are exhausted.
    """

    def __init__(
        self,
        individual_dataloaders: Dict[str, DataLoader],
        iteration_strategy: DataIterationStrategy,
    ) -> None:
        super().__init__(individual_dataloaders, iteration_strategy)
        self.individual_iterators: Mapping[str, Any] = {
            name: iter(dl) for name, dl in individual_dataloaders.items()
        }
        self.remaining_dataloaders: List[str] = list(self.individual_iterators.keys())

    def __next__(self) -> Dict[str, Any]:
        if len(self.remaining_dataloaders) == 0:
            raise StopIteration

        cur_dataloader = random.choice(self.remaining_dataloaders)
        try:
            return {cur_dataloader: next(self.individual_iterators[cur_dataloader])}
        except StopIteration:
            self.remaining_dataloaders.remove(cur_dataloader)

            if len(self.remaining_dataloaders) == 0:
                raise StopIteration

            return self.__next__()


class TestMultiDataloader(unittest.TestCase):
    def test_round_robin_smallest_dataset_exhausted(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}

        round_robin = RoundRobin(
            stopping_mechanism=StoppingMechanism.SMALLEST_DATASET_EXHAUSTED
        )
        multi_dataloader = iter(
            MultiDataloader(
                individual_dataloaders,
                round_robin,
            )
        )

        # default iteration order
        iteration_order = ["1", "2"]

        # Since first dataset is exhausted in the first iteration,
        # it goes through only one full cycle.
        for index in range(2):
            batch = next(multi_dataloader)
            self.assertTrue(len(batch) == 1)
            self.assertTrue(iteration_order[index % 2] in batch)

        # Raises StopIteration after the first cycle
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_round_robin_all_datasets_exhausted(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}

        round_robin = RoundRobin(iteration_order=["2", "1"])
        multi_dataloader = iter(
            MultiDataloader(
                individual_dataloaders,
                round_robin,
            )
        )

        # iteration order as defined in the strategy
        iteration_order = ["2", "1"]

        # Fetches 3 batches in total in the order ["2", "1", "2"]
        for index in range(3):
            batch = next(multi_dataloader)
            self.assertTrue(len(batch) == 1)
            self.assertTrue(iteration_order[index % 2] in batch)

        # StopIteration after both datasets have been exhausted
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_all_dataset_batches_all_datasets_exhausted(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}

        all_dataset_batches = AllDatasetBatches()

        multi_dataloader = iter(
            MultiDataloader(
                individual_dataloaders,
                all_dataset_batches,
            )
        )

        # batch should contain both "1" and "2" keys
        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 2)
        self.assertTrue(("1" in batch) and ("2" in batch))

        # Dataset "1" was finished. batch now contains only samples from "2"
        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 1)
        self.assertTrue("2" in batch)

        # StopIteration after both datasets have been exhausted
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_all_dataset_batches_restart(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}

        all_dataset_batches = AllDatasetBatches(
            StoppingMechanism.RESTART_UNTIL_ALL_DATASETS_EXHAUSTED
        )

        multi_dataloader = iter(
            MultiDataloader(
                individual_dataloaders,
                all_dataset_batches,
            )
        )

        # batch should contain both "1" and "2" keys
        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 2)
        self.assertTrue(("1" in batch) and ("2" in batch))

        # Dataset "1" was finished. batch now contains only samples from "2"
        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 2)
        self.assertTrue(("1" in batch) and ("2" in batch))

        # StopIteration after both datasets have been exhausted
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_all_dataset_batches_smallest_dataset_exhausted(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}

        all_dataset_batches = AllDatasetBatches(
            StoppingMechanism.SMALLEST_DATASET_EXHAUSTED
        )
        multi_dataloader = iter(
            MultiDataloader(
                individual_dataloaders,
                all_dataset_batches,
            )
        )

        # batch should contain both "1" and "2" keys
        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 2)
        self.assertTrue(("1" in batch) and ("2" in batch))

        # StopIteration should be raised since the dataset "1" has no more samples
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_custom_iterator(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}

        multi_dataloader = iter(
            MultiDataloader(
                individual_dataloaders, DataIterationStrategy(), CustomRandomIterator
            )
        )

        # The two dataloaders constitute 3 batches in total
        for _ in range(3):
            batch = next(multi_dataloader)
            self.assertTrue(len(batch) == 1)

        # StopIteration after both datasets have been exhausted
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_random_sampling_dataloader_wrap_around(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(size=32, length=8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(size=32, length=16), batch_size=8)

        individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}

        multi_dataloader = iter(
            MultiDataloader(
                individual_dataloaders,
                RandomizedBatchSampler(
                    weights={"1": 1, "2": 100},
                    stopping_mechanism=StoppingMechanism.WRAP_AROUND_UNTIL_KILLED,
                ),
            )
        )

        # The two dataloaders constitute 3 batches in total
        selected_datasets = [next(multi_dataloader).popitem()[0] for _ in range(25)]
        # check that 2 is significantly more common
        counts = Counter(selected_datasets)
        self.assertTrue(counts["2"] > 0.8 * len(selected_datasets))

    def test_random_sampling_dataloader_with_empty_data(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(size=32, length=0), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(size=32, length=64), batch_size=8)

        individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}

        multi_dataloader = iter(
            MultiDataloader(
                individual_dataloaders,
                RandomizedBatchSampler(
                    weights={"1": 1, "2": 100},
                    stopping_mechanism=StoppingMechanism.ALL_DATASETS_EXHAUSTED,
                ),
                ignore_empty_data=True,
            )
        )

        # The first dataloader is empty
        # It will constitute 25 batches in total
        selected_datasets = [next(multi_dataloader).popitem()[0] for _ in range(8)]
        # check that 2 is the only one
        counts = Counter(selected_datasets)
        self.assertEqual(counts["2"], len(selected_datasets))

        # Now ensure that, without the `ignore_empty_data` flag, an exception is raised.
        with self.assertRaises(ValueError):
            individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}
            MultiDataloader(
                individual_dataloaders,
                RandomizedBatchSampler(weights={"1": 1, "2": 100}),
            )

    def test_random_sampling_dataloader_smallest_dataset_exhausted(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(size=32, length=8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(size=32, length=16), batch_size=8)

        individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}

        multi_dataloader = MultiDataloader(
            individual_dataloaders,
            RandomizedBatchSampler(
                weights={"1": 1, "2": 100},
                stopping_mechanism=StoppingMechanism.SMALLEST_DATASET_EXHAUSTED,
            ),
        )

        num_trials = 25
        len_counts = Counter()
        key_counts = Counter()
        for _ in range(num_trials):
            batches = list(iter(multi_dataloader))
            len_counts[len(batches)] += 1
            key_counts.update(key for batch in batches for key in batch.keys())
        # Most trials should only return two batches
        self.assertTrue(len_counts[2] > 0.8 * num_trials)
        # Most batches overall should be from dataset "2"
        self.assertTrue(key_counts["2"] > 0.8 * sum(key_counts.values()))

    def test_random_sampling_dataloader_all_datasets_exhausted(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(size=32, length=8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(size=32, length=16), batch_size=8)

        individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}

        multi_dataloader = MultiDataloader(
            individual_dataloaders,
            RandomizedBatchSampler(
                weights={"1": 1, "2": 100},
            ),
        )

        num_trials = 25
        trials = [list(iter(multi_dataloader)) for _ in range(num_trials)]

        # All trials should return three batches
        self.assertTrue(all(len(data) == 3 for data in trials))

        # Most trials should start with dataset "2"
        key_counts = Counter(data[0].popitem()[0] for data in trials)
        self.assertTrue(key_counts["2"] > 0.8 * num_trials)

    def test_random_sampling_dataloader_restart_until_all_datasets_exhausted(
        self,
    ) -> None:
        dataloader_1 = DataLoader(RandomDataset(size=32, length=8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(size=32, length=256), batch_size=8)

        individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}

        multi_dataloader = iter(
            MultiDataloader(
                individual_dataloaders,
                RandomizedBatchSampler(
                    weights={"1": 100, "2": 1},
                    stopping_mechanism=StoppingMechanism.RESTART_UNTIL_ALL_DATASETS_EXHAUSTED,
                ),
            )
        )

        num_trials = 32
        selected_datasets = [
            next(multi_dataloader).popitem()[0] for _ in range(num_trials)
        ]
        # check that 1 is significantly more common, even if it is shorter
        counts = Counter(selected_datasets)
        self.assertTrue(counts["1"] > 0.8 * len(selected_datasets))

    def test_inorder(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}

        in_order = InOrder()

        multi_dataloader = iter(
            MultiDataloader(
                individual_dataloaders,
                in_order,
            )
        )

        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 1)
        self.assertTrue("1" in batch)

        for _ in range(2):
            batch = next(multi_dataloader)
            self.assertTrue(len(batch) == 1)
            self.assertTrue("2" in batch)

        # Raises StopIteration after all exhausted
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)

    def test_in_order_with_repetitions(self) -> None:
        dataloader_1 = DataLoader(RandomDataset(32, 8), batch_size=8)
        dataloader_2 = DataLoader(RandomDataset(32, 16), batch_size=8)

        individual_dataloaders = {"1": dataloader_1, "2": dataloader_2}

        in_order = InOrder(iteration_order=["2", "1", "2"])

        multi_dataloader = iter(
            MultiDataloader(
                individual_dataloaders,
                in_order,
            )
        )

        # first 2 batches from dataset 2
        for _ in range(2):
            batch = next(multi_dataloader)
            self.assertTrue(len(batch) == 1)
            self.assertTrue("2" in batch)

        # next batch from dataset 1
        batch = next(multi_dataloader)
        self.assertTrue(len(batch) == 1)
        self.assertTrue("1" in batch)

        # last 2 batches from dataset 2
        for _ in range(2):
            batch = next(multi_dataloader)
            self.assertTrue(len(batch) == 1)
            self.assertTrue("2" in batch)

        # Raises StopIteration after all exhausted
        with self.assertRaises(StopIteration):
            batch = next(multi_dataloader)
