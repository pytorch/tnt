from .dataset import Dataset


class TransformDataset(Dataset):
    """
    Dataset which transforms a given dataset with a given function.

    Given a function `transform`, and a `dataset`, `TransformDataset` applies
    the function in an on-the-fly manner when querying a sample with
    `__getitem__(idx)` and therefore returning `transform[dataset[idx]]`.

    `transform` can also be a dict with functions as values. In this case, it
    is assumed that `dataset[idx]` is a dict which has all the keys in
    `transform`. Then, `transform[key]` is applied to dataset[idx][key] for
    each key in `transform`

    The size of the new dataset is equal to the size of the underlying
    `dataset`.

    Purpose: when performing pre-processing operations, it is convenient to be
    able to perform on-the-fly transformations to a dataset.

    Args:
        dataset (Dataset): Dataset which has to be transformed.
        transforms (function/dict): Function or dict with function as values.
            These functions will be applied to data.
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
