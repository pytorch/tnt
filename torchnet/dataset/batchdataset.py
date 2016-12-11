from . import dataset
import math
import torch
from .. import transform

class BatchDataset(dataset.Dataset):
    def __init__(self,
            dataset,
            batchsize,
            perm = lambda idx, size: idx,
            merge = None,
            policy = 'include-last',
            filter = lambda sample: True):
        self.dataset = dataset
        self.perm = perm
        self.batchsize = batchsize
        self.policy = policy
        self.filter = filter
        self.makebatch = transform.makebatch(merge)
        len(self)

    def __len__(self):
        if self.policy == 'include-last':
            return int(math.ceil(float(len(self.dataset) / self.batchsize)))
        elif self.policy == 'skip-last':
            return int(math.floor(float(len(sel.dataset) / self.batchsize)))
        elif self.policy == 'divisible-only':
            assert len(self.dataset) % self.batchsize == 0, \
                    'dataset size is not divisible by batch size'
            return len(self.dataset) / self.batchsize
        else:
            assert False, 'invalid policy (include-last | skip-last | divisible-only expected)'

    def __getitem__(self, idx):
        super(BatchDataset, self).__getitem__(idx)
        samples = []
        assert idx >= 0 and idx < len(self)
        samples = []
        maxidx = len(self.dataset)
        for i in range(0, self.batchsize):
            j = idx * self.batchsize + i
            if j >= maxidx:
                break
            j = self.perm(j, maxidx)
            sample = self.dataset[j]
            if self.filter(sample):
                samples.append(sample)
        samples = self.makebatch(samples)
        return samples
        
