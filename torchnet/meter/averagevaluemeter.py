import math
from . import meter
import numpy as np


class AverageValueMeter(meter.Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()

    def add(self, value, n = 1):
        self.n += n
        if self.n == 1:
            self.m_oldM = value
            self.m_newM = value;
            self.m_S = 0.0
        else:
            self.m_newM = self.m_oldM + (value - self.m_oldM)/float(self.n);
            self.m_S += (value - self.m_oldM) * (value - self.m_newM);

            # set up for next iteration
            self.m_oldM = self.m_newM;

    def value(self):
        n = self.n
        if n == 0:
            mean, std = np.nan, np.nan
        elif n == 1:
            return self.m_newM, np.nan
        else:
            mean = self.m_newM
            variance = self.m_S/(self.n - 1.0)
            std = math.sqrt(variance)
        return mean, std

    def reset(self):
        self.n = 0
        self.m_oldM = 0.0
        self.m_newM = 0.0
        self.m_S = 0.0

