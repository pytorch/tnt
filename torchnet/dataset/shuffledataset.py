from . import resampledataset
import torch

class ShuffleDataset(resampledataset.ResampleDataset):

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

