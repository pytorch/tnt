from . import dataset
import torch

class TransformDataset(dataset.Dataset):
    def __init__(self, dataset, transforms):
        assert isinstance(transforms, dict), 'expected a dict of transforms'
        for k,v in transforms.items():
            assert callable(v), str(k) + ' is not a function'
        self.dataset = dataset
        self.transforms = transforms.copy()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        super(TransformDataset, self).__getitem__(idx)
        z = self.dataset[idx]
        for k, transform in self.transforms.items():
            z[k] = transform(z[k])
        return z

