from .resampledataset import ResampleDataset
import torch


class ShuffleDataset(ResampleDataset):
    """
    `tnt.ShuffleDataset` is a sub-class of
    [tnt.ResampleDataset](#ResampleDataset) provided for convenience.
    It samples uniformly from the given `dataset` with, or without
    `replacement`. The chosen partition can be redrawn by calling
    [resample()](#ShuffleDataset.resample).
    If `replacement` is `true`, then the specified `size` may be larger than
    the underlying `dataset`.
    If `size` is not provided, then the new dataset size will be equal to the
    underlying `dataset` size.
    Purpose: the easiest way to shuffle a dataset!
    """
    def __init__(self, dataset, size=None, replacement=False):
        if size and not replacement and size > len(dataset):
            raise ValueError('size cannot be larger than underlying dataset \
                    size when sampling without replacement')
        super(ShuffleDataset, self).__init__(dataset,
                lambda dataset, idx: self.perm[idx], size)
        self.replacement = replacement
        self.resample()
    
    def resample(self):
        if self.replacement:
            self.perm = torch.LongTensor(len(self)).random_(len(self.dataset))
        else:
            self.perm = torch.randperm(len(self.dataset)).narrow(0, 0, len(self))

