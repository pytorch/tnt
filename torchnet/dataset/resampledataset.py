from .dataset import Dataset


class ResampleDataset(Dataset):
    """
    Given a `dataset`, creates a new dataset which will (re-)sample from this
    underlying dataset using the provided `sampler(dataset, idx)` closure.
    If `size` is provided, then the newly created dataset will have the
    specified `size`, which might be different than the underlying dataset
    size.

    If `size` is not provided, then the new dataset will have the same size
    than the underlying one.

    By default `sampler(dataset, idx)` is the identity, simply `return`ing `idx`.
    `dataset` corresponds to the underlying dataset provided at construction, and
    `idx` may take a value between 1 to `size`. It must return an index in the range
    acceptable for the underlying dataset.

    Purpose: shuffling data, re-weighting samples, getting a subset of the
    data. Note that an important sub-class is ([tnt.ShuffleDataset](#ShuffleDataset)),
    provided for convenience.
    """
    def __init__(self, dataset, sampler, size=None):
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
