from . import dataset
import math
import torch

class BatchDataset(dataset.Dataset):
    def __init__(self, dataset, batchsize, perm, merge, policy, filter):
        self.dataset = dataset
        self.perm = perm
        self.batchsize = batchsize
        self.policy = policy
        self.filter = filter
        len(self)

    def __len__(self):
        if self.policy == 'include-last':
            return math.ceil(float(self.dataset.size()) / self.batchsize)
        elif self.policy == 'skip-last':
            return math.floor(float(self.dataset.size()) / self.batchsize)
        elif self.policy == 'divisible-only':
            assert len(self.dataset) % self.batchsize == 0, \
                    'dataset size is not divisible by batch size'
            return len(self.dataset) / self.batchsize
        else:
            assert False, 'invalid policy (include-last | skip-last | divisible-only expected)'

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self)
        samples = []
        maxidx = len(self.dataset)
        for i in range(0, self.batchsize):
            idx = idx * self.batchsize + i
            if idx >= maxidx:
                break
            idx = self.perm(idx, maxidx)
            sample = self.dataset.get(idx)
            if self.filter(sample):
                samples.append(sample)
        samples = self.makebatch(samples)
        return samples
        
