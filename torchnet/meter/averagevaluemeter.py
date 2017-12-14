import math
from . import meter
import numpy as np

class AverageValueMeter(meter.Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()

    def add(self, value, n = 1):
        self.val = val
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            return self.sum, np.inf
        else:
            self.mean = self.sum / self.n
            self.std = math.sqrt( (self.var - self.n * self.mean * self.mean) / (self.n - 1.0) )

    def value(self):
	return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
	self.val = 0.0
	self.mean = np.nan
	self.std = np.nan

