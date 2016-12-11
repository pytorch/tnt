import unittest
import torch
import torchnet.meter as meter
import numpy as np

class TestMeters(unittest.TestCase):
    def testAverageValueMeter(self):
        m = meter.AverageValueMeter()
        for i in range(1,10):
            m.add(i)
        mean, std = m.value()
        self.assertEqual(mean, 5.0)
        m.reset()
        mean, std = m.value()
        self.assert_(np.isnan(mean))

    def testClassErrorMeter(self):
        mtr = meter.ClassErrorMeter(topk = [1])
        output = torch.eye(3)
        target = torch.range(0,2)
        mtr.add(output, target)
        err = mtr.value()
        
        self.assertEqual(err, [0], "All should be correct")

        target[0] = 1
        target[1] = 0
        target[2] = 0
        mtr.add(output, target)
        err = mtr.value()
        self.assertEqual(err, [50.0], "Half should be correct")

    def testConfusionMeter(self):
        mtr = meter.ConfusionMeter(k = 3)

        output = torch.Tensor([[.8,0.1,0.1],[10,11,10],[0.2,0.2,.3]])
        target = torch.range(0,2)
        mtr.add(output, target)

        conf_mtrx = mtr.value()
        self.assertEqual(conf_mtrx.sum(), 3, "All should be correct")
        self.assertEqual(conf_mtrx.diagonal().sum(), 3, "All should be correct")

        target = torch.Tensor([1,0,0])
        mtr.add(output, target)

        self.assertEqual(conf_mtrx.sum(), 6, \
                "Six tests should give six values")
        self.assertEqual(conf_mtrx.diagonal().sum(), 3, \
                "Shouldn't have changed since all new values were false")
        self.assertEqual(conf_mtrx[0].sum(), 3, \
                "All top have gotten one guess")
        self.assertEqual(conf_mtrx[1].sum(), 2, \
                "Two first at the 2nd row have a guess")
        self.assertEqual(conf_mtrx[1][2], 0, \
                "The last one should be empty")
        self.assertEqual(conf_mtrx[2].sum(), 1, \
                "Bottom row has only the first test correct")
        self.assertEqual(conf_mtrx[2][2], 1, \
                "Bottom row has only the first test correct")

        mtr = meter.ConfusionMeter(k = 4, normalized = True)
        output = torch.Tensor([
            [.8,0.1,0.1,0],
            [10,11,10,0],
            [0.2,0.2,.3,0],
            [0,0,0,1],
            ])

        target = torch.Tensor([0,1,2,3])
        mtr.add(output, target)
        conf_mtrx = mtr.value()

        self.assertEqual(conf_mtrx.sum(), output.size(1), \
                "All should be correct")
        self.assertEqual(conf_mtrx.diagonal().sum(), output.size(1), \
                "All should be correct")

        target[0] = 1
        target[1] = 0
        target[2] = 0

        mtr.add(output, target)
        conf_mtrx = mtr.value()

        self.assertEqual(conf_mtrx.sum(), output.size(1), \
                "The normalization should sum all values to 1")
        for i,row in enumerate(conf_mtrx):
            self.assertEqual(row.sum(), 1, \
                    "Row no " + str(i) + " fails to sum to one in normalized mode")

if __name__ == '__main__':
    unittest.main()
