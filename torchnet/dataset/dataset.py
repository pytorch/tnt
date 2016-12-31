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

    def parallel(self, batch_size=1, shuffle=False, sampler=None,
                 num_workers=0, collate_fn=lambda x: x[0],
                 pin_memory=False):
        return DataLoader(self, batch_size, shuffle, sampler, num_workers,
                          collate_fn, pin_memory)
