import math
from . import meter
import torch


class MAEMeter(meter.Meter):
    def __init__(self):
        super(MAEMeter, self).__init__()
        self.reset()

    def reset(self):
        self.n = 0
        self.sesum = 0.0

    def add(self, output, target):
        if not torch.is_tensor(output) and not torch.is_tensor(target):
            output = torch.from_numpy(output)
            target = torch.from_numpy(target)
        self.n += output.numel()
        self.sesum += torch.sum(torch.abs(output - target))

    def value(self):
        return self.sesum / max(1, self.n)
