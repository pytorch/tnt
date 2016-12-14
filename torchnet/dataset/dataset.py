import torchnet

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
