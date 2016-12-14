from . import dataset
import torch

class ResampleDataset(dataset.Dataset):

    def __init__(self, dataset, sampler, size=None):
        super(ResampleDataset, self).__init__()
        self.dataset = dataset
        self.sampler = sampler
        self.size = size

    def __len__(self):
        return (self.size and self.size > 0) and self.size or len(self.dataset)

    def __getitem__(self, idx):
        super(ResampleDataset, self).__getitem__(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError('out of range')
        idx = self.sampler(self.dataset, idx)
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError('out of range')
        return self.dataset[idx]
