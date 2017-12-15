from .resampledataset import ResampleDataset
import torch


class ShuffleDataset(ResampleDataset):
    """
    Dataset which shuffles a given dataset.

    `ShuffleDataset` is a sub-class of `ResampleDataset` provided for
    convenience. It samples uniformly from the given `dataset` with, or without
    `replacement`. The chosen partition can be redrawn by calling `resample()`

    If `replacement` is `true`, then the specified `size` may be larger than
    the underlying `dataset`.
    If `size` is not provided, then the new dataset size will be equal to the
    underlying `dataset` size.

    Purpose: the easiest way to shuffle a dataset!

    Args:
        dataset (Dataset): Dataset to be shuffled.
        size (int, optional): Desired size of the shuffled dataset. If
            `replacement` is `true`, then can be larger than the `len(dataset)`.
            By default, the new dataset will have the same size as `dataset`.
        replacement (bool, optional): True if uniform sampling is to be done
            with replacement. False otherwise. Defaults to false.

    Raises:
        ValueError: If `size` is larger than the size of the underlying dataset
            and `replacement` is False.
    """

    def __init__(self, dataset, size=None, replacement=False):
        if size and not replacement and size > len(dataset):
            raise ValueError('size cannot be larger than underlying dataset \
                    size when sampling without replacement')

        super(ShuffleDataset, self).__init__(dataset,
                                             lambda dataset, idx: self.perm[idx],
                                             size)
        self.replacement = replacement
        self.resample()

    def resample(self, seed=None):
        """Resample the dataset.

        Args:
            seed (int, optional): Seed for resampling. By default no seed is
            used.
        """
        if seed is not None:
            gen = torch.manual_seed(seed)
        else:
            gen = torch.default_generator

        if self.replacement:
            self.perm = torch.LongTensor(len(self)).random_(
                len(self.dataset), generator=gen)
        else:
            self.perm = torch.randperm(
                len(self.dataset), generator=gen).narrow(0, 0, len(self))
