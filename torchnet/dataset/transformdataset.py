from . import dataset
import torch

class TransformDataset(dataset.Dataset):
    def __init__(self, dataset, transforms):
        assert isinstance(transforms, dict) or callable(transforms), \
                'expected a dict of transforms or a function'
        if isinstance(transforms, dict):
            for k,v in transforms.items():
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

