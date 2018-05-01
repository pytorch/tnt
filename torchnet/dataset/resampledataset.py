from .dataset import Dataset


class ResampleDataset(Dataset):
    """
    Dataset which resamples a given dataset.

    Given a `dataset`, creates a new dataset which will (re-)sample from this
    underlying dataset using the provided `sampler(dataset, idx)` function.

    If `size` is provided, then the newly created dataset will have the
    specified `size`, which might be different than the underlying dataset
    size. If `size` is not provided, then the new dataset will have the same
    size as the underlying one.

    Purpose: shuffling data, re-weighting samples, getting a subset of the
    data. Note that an important sub-class `ShuffleDataset` is provided for
    convenience.

    Args:
        dataset (Dataset): Dataset to be resampled.
        sampler (function, optional): Function used for sampling. `idx`th
            sample is returned by `dataset[sampler(dataset, idx)]`. By default
            `sampler(dataset, idx)` is the identity, simply returning `idx`.
            `sampler(dataset, idx)` must return an index in the range
            acceptable for the underlying `dataset`.
        size (int, optional): Desired size of the dataset after resampling. By
            default, the new dataset will have the same size as the underlying
            one.

    """

    def __init__(self, dataset, sampler=lambda ds, idx: idx, size=None):
        super(ResampleDataset, self).__init__()
        self.dataset = dataset
        self.sampler = sampler
        self.size = size

    def __len__(self):
        return (self.size and self.size > 0) and self.size or len(self.dataset)

    def __getitem__(self, idx):
        super(ResampleDataset, self).__getitem__(idx)
        idx = self.sampler(self.dataset, idx)

        if idx < 0 or idx >= len(self.dataset):
            raise IndexError('out of range')

        return self.dataset[idx]
