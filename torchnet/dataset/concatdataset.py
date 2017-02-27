from .dataset import Dataset
import numpy as np


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.

    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Args:
        datasets (iterable): List of datasets to be concatenated
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()

        self.datasets = list(datasets)
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.cum_sizes = np.cumsum([len(x) for x in self.datasets])

    def __len__(self):
        return self.cum_sizes[-1]

    def __getitem__(self, idx):
        super(ConcatDataset, self).__getitem__(idx)
        dataset_index = self.cum_sizes.searchsorted(idx, 'right')

        if dataset_index == 0:
            dataset_idx = idx
        else:
            dataset_idx = idx - self.cum_sizes[dataset_index - 1]

        return self.datasets[dataset_index][dataset_idx]
