from .dataset import Dataset
import numpy as np


class SplitDataset(Dataset):
    """
    Dataset to partition a given dataset.

    Partition a given `dataset`, according to the specified `partitions`. Use
    the method `select()` to select the current partition in use.

    The `partitions` is a dictionary where a key is a user-chosen string
    naming the partition, and value is a number representing the weight (as a
    number between 0 and 1) or the size (in number of samples) of the
    corresponding partition.

    Partioning is achieved linearly (no shuffling). See `ShuffleDataset` if you
    want to shuffle the dataset before partitioning.

    Args:
        dataset (Dataset): Dataset to be split.
        partitions (dict): Dictionary where key is a user-chosen string
            naming the partition, and value is a number representing the weight
            (as a number between 0 and 1) or the size (in number of samples)
            of the corresponding partition.
        initial_partition (str, optional): Initial parition to be selected.

    """

    def __init__(self, dataset, partitions, initial_partition=None):
        super(SplitDataset, self).__init__()

        self.dataset = dataset
        self.partitions = partitions

        # A few assertions
        assert isinstance(partitions, dict), 'partitions must be a dict'
        assert len(partitions) >= 2, \
            'SplitDataset should have at least two partitions'
        assert min(partitions.values()) >= 0, \
            'partition sizes cannot be negative'
        assert max(partitions.values()) > 0, 'all partitions cannot be empty'

        self.partition_names = sorted(list(self.partitions.keys()))
        self.partition_index = {partition: i for i, partition in
                                enumerate(self.partition_names)}

        self.partition_sizes = [self.partitions[parition] for parition in
                                self.partition_names]
        # if partition sizes are fractions, convert to sizes:
        if sum(self.partition_sizes) <= 1:
            self.partition_sizes = [round(x * len(dataset)) for x in
                                    self.partition_sizes]
        else:
            for x in self.partition_sizes:
                assert x == int(x), ('partition sizes should be integer'
                                     ' numbers, or sum up to <= 1 ')

        self.partition_cum_sizes = np.cumsum(self.partition_sizes)

        if initial_partition is not None:
            self.select(initial_partition)

    def select(self, partition):
        """
        Select the parition.

        Args:
            partition (str): Partition to be selected.
        """
        self.current_partition_idx = self.partition_index[partition]

    def __len__(self):
        try:
            return self.partition_sizes[self.current_partition_idx]
        except AttributeError:
            raise ValueError("Select a partition before accessing data.")

    def __getitem__(self, idx):
        super(SplitDataset, self).__getitem__(idx)
        try:
            if self.current_partition_idx == 0:
                return self.dataset[idx]
            else:
                offset = self.partition_cum_sizes[self.current_partition_idx - 1]
                return self.dataset[int(offset) + idx]
        except AttributeError:
            raise ValueError("Select a partition before accessing data.")
