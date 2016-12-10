import numpy as np
import torch
from . import meter

class ClassErrorMeter(meter.Meter):
    def __init__(self, topk = [1], accuracy = False):
        self.topk = np.sort(topk)
        self.accuracy = accuracy
        self.reset()

    def reset(self):
        self.sum = np.zeros(len(self.topk))
        self.n = 0

    def add(self, output, target):
        output = output.cpu().squeeze().numpy()
        target = target.cpu().squeeze().numpy()
        if np.ndim(output) == 1:
            output = output[np.newaxis]
        else:
            assert np.ndim(output) == 2, \
                    'wrong output size (1D or 2D expected)'
            assert np.ndim(target) == 1, \
                    'target and output do not match'
        assert target.shape[0] == output.shape[0], \
            'target and output do not match'
        topk = self.topk
        maxk = topk[-1]
        no = output.shape[0]

        pred = torch.from_numpy(output).topk(maxk, 1, True, True)[1].numpy()
        correct = pred == target.reshape(pred.shape)

        for k in topk:
            self.sum[k-1] += no - correct[:,0:k].sum()
        self.n += no

    def value(self, k = -1):
        if k != -1:
            assert k <= self.sum.shape[0] and k >= 1, \
                    'invalid k (this k was not provided at construction time)'
            return self.accuracy and (1-self.sum[k-1] / self.n) * 100 or self.sum[k-1]*100 / self.n
        else:
            return [self.value(k) for k in self.topk]
