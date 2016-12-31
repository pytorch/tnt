from .dataset import Dataset


class TransformDataset(Dataset):
    """
    Given a closure `transform()`, and a `dataset`, `tnt.TransformDataset`
    applies the closure in an on-the-fly manner when querying a sample with
    `tnt.Dataset.__getitem__()`.

    If key is provided, the closure is applied to the sample field specified
    by `key` (only). The closure must return the new corresponding field value.
    If key is not provided, the closure is applied on the full sample. The
    closure must return the new sample table.

    The size of the new dataset is equal to the size of the underlying
    `dataset`.
    Purpose: when performing pre-processing operations, it is convenient to be
    able to perform on-the-fly transformations to a dataset.
    """
    def __init__(self, dataset, transforms):
        super(TransformDataset, self).__init__()
        assert isinstance(transforms, dict) or callable(transforms), \
            'expected a dict of transforms or a function'
        if isinstance(transforms, dict):
            for k, v in transforms.items():
                assert callable(v), str(k) + ' is not a function'
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        super(TransformDataset, self).__getitem__(idx)
        z = self.dataset[idx]
        if isinstance(self.transforms, dict):
            for k, transform in self.transforms.items():
                z[k] = transform(z[k])
        else:
            z = self.transforms(z)
        return z
