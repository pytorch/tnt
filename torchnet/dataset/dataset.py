import torchnet
from torch.utils.data import DataLoader


class Dataset(object):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("CustomRange index out of range")
        pass

    def batch(self, *args, **kwargs):
        return torchnet.dataset.BatchDataset(self, *args, **kwargs)

    def transform(self, *args, **kwargs):
        return torchnet.dataset.TransformDataset(self, *args, **kwargs)

    def shuffle(self, *args, **kwargs):
        return torchnet.dataset.ShuffleDataset(self, *args, **kwargs)

    def parallel(self, *args, **kwargs):
        return DataLoader(self, *args, **kwargs)

    def split(self, *args, **kwargs):
        return torchnet.dataset.SplitDataset(self, *args, **kwargs)
