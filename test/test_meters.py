import unittest
import math
import torch
import torchnet.meter as meter
import numpy as np


class TestMeters(unittest.TestCase):

    def testAverageValueMeter(self):
        m = meter.AverageValueMeter()
        for i in range(1, 10):
            m.add(i)
        mean, std = m.value()
        self.assertEqual(mean, 5.0)
        m.reset()
        mean, std = m.value()

        self.assertTrue(np.isnan(mean))

    def testAverageValueMeter_np_2d(self):
        m = meter.AverageValueMeter()
        for i in range(1, 10):
            m.add(np.float32([[i, i + 1]]))
        mean, std = m.value()
        self.assertTrue(np.allclose(mean, [[5.0, 6.0]]))
        self.assertTrue(np.allclose(std, [[2.738613, 2.738613]]))
        m.reset()
        mean, std = m.value()

        self.assertTrue(np.isnan(mean))

    def testAverageValueMeter_torch_2d(self):
        m = meter.AverageValueMeter()
        for i in range(1, 10):
            m.add(torch.Tensor([[i, i + 1]]))
        mean, std = m.value()
        self.assertTrue(np.allclose(mean, [[5.0, 6.0]]))
        self.assertTrue(np.allclose(std, [[2.738613, 2.738613]]))
        m.reset()
        mean, std = m.value()

        self.assertTrue(np.isnan(mean))

    def testAverageValueMeter_n(self):
        """Test the case of adding more than 1 value.
        """
        m = meter.AverageValueMeter()
        for i in range(1, 11):
            m.add(i * i, n=i)
        mean, std = m.value()
        self.assertEqual(mean, 7.0)
        m.reset()
        mean, std = m.value()

        self.assertTrue(np.isnan(mean))

    def testAverageValueMeter_stable(self):
        """Test the case of near-zero variance.

        The test compares the results to numpy, and uses
        isclose() to allow for some small differences in
        the results, which are due to slightly different arithmetic
        operations and order.
        """
        def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
            return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

        m = meter.AverageValueMeter()
        samples = [0.7] * 10
        truth = np.array([])
        for sample in samples:
            truth = np.append(truth, sample)
            m.add(sample)
            mean, std = m.value()
            self.assertTrue(isclose(truth.mean(), mean))
            self.assertTrue(
                (math.isnan(std) and math.isnan(truth.std(ddof=1))) or
                # When there is one sample in the dataset, numpy returns NaN and
                # AverageValueMeter returns Inf.  We forgive AverageValueMeter :-)
                (math.isinf(std) and math.isnan(truth.std(ddof=1))) or
                isclose(std, truth.std(ddof=1), abs_tol=1e-07))

    def testClassErrorMeter(self):
        mtr = meter.ClassErrorMeter(topk=[1])
        output = torch.eye(3)
        if hasattr(torch, "arange"):
            target = torch.arange(0, 3)
        else:
            target = torch.range(0, 2)
        mtr.add(output, target)
        err = mtr.value()

        self.assertEqual(err, [0], "All should be correct")

        target[0] = 1
        target[1] = 0
        target[2] = 0
        mtr.add(output, target)
        err = mtr.value()
        self.assertEqual(err, [50.0], "Half should be correct")

    def testClassErrorMeteri_batch1(self):
        mtr = meter.ClassErrorMeter(topk=[1])
        output = torch.tensor([1, 0, 0])
        if hasattr(torch, "arange"):
            target = torch.arange(0, 1)
        else:
            target = torch.range(0, 0)
        mtr.add(output, target)
        err = mtr.value()
        self.assertEqual(err, [0], "All should be correct")

    def testConfusionMeter(self):
        mtr = meter.ConfusionMeter(k=3)

        output = torch.Tensor([[.8, 0.1, 0.1], [10, 11, 10], [0.2, 0.2, .3]])
        if hasattr(torch, "arange"):
            target = torch.arange(0, 3)
        else:
            target = torch.range(0, 2)
        mtr.add(output, target)

        conf_mtrx = mtr.value()
        self.assertEqual(conf_mtrx.sum(), 3, "All should be correct")
        self.assertEqual(conf_mtrx.diagonal().sum(),
                         3, "All should be correct")

        target = torch.Tensor([1, 0, 0])
        mtr.add(output, target)

        self.assertEqual(conf_mtrx.sum(), 6,
                         "Six tests should give six values")
        self.assertEqual(conf_mtrx.diagonal().sum(), 3,
                         "Shouldn't have changed since all new values were false")
        self.assertEqual(conf_mtrx[0].sum(), 3,
                         "All top have gotten one guess")
        self.assertEqual(conf_mtrx[1].sum(), 2,
                         "Two first at the 2nd row have a guess")
        self.assertEqual(conf_mtrx[1][2], 0,
                         "The last one should be empty")
        self.assertEqual(conf_mtrx[2].sum(), 1,
                         "Bottom row has only the first test correct")
        self.assertEqual(conf_mtrx[2][2], 1,
                         "Bottom row has only the first test correct")

        mtr = meter.ConfusionMeter(k=4, normalized=True)
        output = torch.Tensor([
            [.8, 0.1, 0.1, 0],
            [10, 11, 10, 0],
            [0.2, 0.2, .3, 0],
            [0, 0, 0, 1],
        ])

        target = torch.Tensor([0, 1, 2, 3])
        mtr.add(output, target)
        conf_mtrx = mtr.value()

        self.assertEqual(conf_mtrx.sum(), output.size(1),
                         "All should be correct")
        self.assertEqual(conf_mtrx.diagonal().sum(), output.size(1),
                         "All should be correct")

        target[0] = 1
        target[1] = 0
        target[2] = 0

        mtr.add(output, target)
        conf_mtrx = mtr.value()

        self.assertEqual(conf_mtrx.sum(), output.size(1),
                         "The normalization should sum all values to 1")
        for i, row in enumerate(conf_mtrx):
            self.assertEqual(row.sum(), 1,
                             "Row no " + str(i) + " fails to sum to one in normalized mode")

    def testMSEMeter(self):
        a = torch.ones(7)
        b = torch.zeros(7)

        mtr = meter.MSEMeter()
        mtr.add(a, b)
        self.assertEqual(1.0, mtr.value())

    def testMovingAverageValueMeter(self):
        mtr = meter.MovingAverageValueMeter(3)

        mtr.add(1)
        avg, var = mtr.value()

        self.assertEqual(avg, 1.0)
        self.assertEqual(var, 0.0)
        mtr.add(3)
        avg, var = mtr.value()
        self.assertEqual(avg, 2.0)
        self.assertEqual(var, math.sqrt(2))

        mtr.add(5)
        avg, var = mtr.value()
        self.assertEqual(avg, 3.0)
        self.assertEqual(var, 2.0)

        mtr.add(4)
        avg, var = mtr.value()
        self.assertEqual(avg, 4.0)
        self.assertEqual(var, 1.0)

        mtr.add(0)
        avg, var = mtr.value()
        self.assertEqual(avg, 3.0)
        self.assertEqual(var, math.sqrt(7))

    def testAUCMeter(self):
        mtr = meter.AUCMeter()

        test_size = 1000
        mtr.add(torch.rand(test_size), torch.zeros(test_size))
        mtr.add(torch.rand(test_size), torch.Tensor(test_size).fill_(1))

        val, tpr, fpr = mtr.value()
        self.assertTrue(math.fabs(val - 0.5) < 0.1, msg="AUC Meter fails")

        mtr.reset()
        mtr.add(torch.Tensor(test_size).fill_(0), torch.zeros(test_size))
        mtr.add(torch.Tensor(test_size).fill_(0.1), torch.zeros(test_size))
        mtr.add(torch.Tensor(test_size).fill_(0.2), torch.zeros(test_size))
        mtr.add(torch.Tensor(test_size).fill_(0.3), torch.zeros(test_size))
        mtr.add(torch.Tensor(test_size).fill_(0.4), torch.zeros(test_size))
        mtr.add(torch.Tensor(test_size).fill_(1),
                torch.Tensor(test_size).fill_(1))
        val, tpr, fpr = mtr.value()

        self.assertEqual(val, 1.0, msg="AUC Meter fails")

    def testAPMeter(self):
        mtr = meter.APMeter()

        target = torch.Tensor([0, 1, 0, 1])
        output = torch.Tensor([0.1, 0.2, 0.3, 4])
        weight = torch.Tensor([0.5, 1.0, 2.0, 0.1])
        mtr.add(output, target, weight)

        ap = mtr.value()
        val = (1 * 0.1 / 0.1 + 0 * 2.0 / 2.1 + 1.1 * 1 / 3.1 + 0 * 1 / 4) / 2.0
        self.assertTrue(
            math.fabs(ap[0] - val) < 0.01,
            msg='ap test1 failed'
        )

        mtr.reset()
        mtr.add(output, target)
        ap = mtr.value()
        val = (1 * 1.0 / 1.0 + 0 * 1.0 / 2.0 +
               2 * 1.0 / 3.0 + 0 * 1.0 / 4.0) / 2.0
        self.assertTrue(
            math.fabs(ap[0] - val) < 0.01, msg='ap test2 failed')

        target = torch.Tensor([0, 1, 0, 1])
        output = torch.Tensor([4, 3, 2, 1])
        weight = torch.Tensor([1, 2, 3, 4])

        mtr.reset()
        mtr.add(output, target, weight)
        ap = mtr.value()
        val = (0 * 1.0 / 1.0 + 1.0 * 2.0 / 3.0 +
               2.0 * 0 / 6.0 + 6.0 * 1.0 / 10.0) / 2.0
        self.assertTrue(math.fabs(ap[0] - val) < 0.01, msg='ap test3 failed')

        mtr.reset()
        mtr.add(output, target)
        ap = mtr.value()
        val = (0 * 1.0 + 1 * 1.0 / 2.0 + 0 * 1.0 / 3.0 + 2 * 1.0 / 4.0) / 2.0
        self.assertTrue(math.fabs(ap[0] - val) < 0.01, msg='ap test4 failed')

        target = torch.Tensor([0, 1, 0, 1])
        output = torch.Tensor([1, 4, 2, 3])
        weight = torch.Tensor([1, 2, 3, 4])
        mtr.reset()
        mtr.add(output, target, weight)
        ap = mtr.value()
        val = (4 * 1.0 / 4.0 + 6 * 1.0 / 6.0 + 0 *
               6.0 / 9.0 + 0 * 6.0 / 10.0) / 2.0
        self.assertTrue(math.fabs(ap[0] - val) < 0.01, msg='ap test5 failed')

        mtr.reset()
        mtr.add(output, target)
        ap = mtr.value()
        val = (1 * 1.0 + 2 * 1.0 / 2.0 + 0 * 1.0 / 3.0 + 0 * 1.0 / 4.0) / 2.0
        self.assertTrue(math.fabs(ap[0] - val) < 0.01, msg='ap test6 failed')

        target = torch.Tensor([0, 0, 0, 0])
        output = torch.Tensor([1, 4, 2, 3])
        weight = torch.Tensor([1.0, 0.1, 0.0, 0.5])
        mtr.reset()
        mtr.add(output, target, weight)

        ap = mtr.value()
        self.assertEqual(ap[0], 0.)

        mtr.reset()
        mtr.add(output, target)
        ap = mtr.value()
        self.assertEqual(ap[0], 0.)

        target = torch.Tensor([1, 1, 0])
        output = torch.Tensor([3, 1, 2])
        weight = torch.Tensor([1, 0.1, 3])
        mtr.reset()
        mtr.add(output, target, weight)
        ap = mtr.value()
        val = (1 * 1.0 / 1.0 + 1 * 0.0 / 4.0 + 1.1 / 4.1) / 2.0
        self.assertTrue(math.fabs(ap[0] - val) < 0.01, msg='ap test7 failed')

        mtr.reset()
        mtr.add(output, target)
        ap = mtr.value()
        val = (1 * 1.0 + 0 * 1.0 / 2.0 + 2 * 1.0 / 3.0) / 2.0
        self.assertTrue(math.fabs(ap[0] - val) < 0.01, msg='ap test8 failed')

        # Test multiple K's
        target = torch.Tensor([[0, 1, 0, 1], [0, 1, 0, 1]]).transpose(0, 1)
        output = torch.Tensor([[.1, .2, .3, 4], [4, 3, 2, 1]]).transpose(0, 1)
        weight = torch.Tensor([[1.0, 0.5, 2.0, 3.0]]).transpose(0, 1)
        mtr.reset()
        mtr.add(output, target, weight)
        ap = mtr.value()
        self.assertTrue(
            math.fabs(ap.sum() -
                      torch.Tensor([
                          (1 * 3.0 / 3.0 + 0 * 3.0 / 5.0 +
                           3.5 * 1 / 5.5 + 0 * 3.5 / 6.5) / 2.0,
                          (0 * 1.0 / 1.0 + 1 * 0.5 / 1.5 +
                              0 * 0.5 / 3.5 + 1 * 3.5 / 6.5) / 2.0
                      ]).sum()) < 0.01, msg='ap test9 failed')

        mtr.reset()
        mtr.add(output, target)
        ap = mtr.value()
        self.assertTrue(
            math.fabs(ap.sum() -
                      torch.Tensor([
                          (1 * 1.0 + 0 * 1.0 / 2.0 + 2 *
                           1.0 / 3 + 0 * 1.0 / 4.0) / 2.0,
                          (0 * 1.0 + 1 * 1.0 / 2.0 + 0 *
                              1.0 / 3.0 + 2.0 * 1.0 / 4.0) / 2.0
                      ]).sum()) < 0.01, msg='ap test10 failed')

        mtr.reset()
        output = torch.Tensor(5, 4).fill_(0.25)
        target = torch.ones(5, 4)
        mtr.add(output, target)
        output = torch.Tensor(1, 4).fill_(0.25)
        target = torch.ones(1, 4)
        mtr.add(output, target)
        self.assertEqual(mtr.value().size(0), 4, msg='ap test11 failed')

    def testmAPMeter(self):
        mtr = meter.mAPMeter()
        target = torch.Tensor([0, 1, 0, 1])
        output = torch.Tensor([0.1, 0.2, 0.3, 4])
        weight = torch.Tensor([0.5, 1.0, 2.0, 0.1])
        mtr.add(output, target)

        ap = mtr.value()
        val = (1 * 1.0 / 1.0 + 0 * 1.0 / 2.0 +
               2.0 * 1.0 / 3.0 + 0 * 1.0 / 4.0) / 2.0
        self.assertTrue(
            math.fabs(ap - val) < 0.01,
            msg='mAP test1 failed'
        )

        mtr.reset()
        mtr.add(output, target, weight)
        ap = mtr.value()
        val = (1 * 0.1 / 0.1 + 0 * 2.0 / 2.1 +
               1.1 * 1 / 3.1 + 0 * 1.0 / 4.0) / 2.0
        self.assertTrue(
            math.fabs(ap - val) < 0.01, msg='mAP test2 failed')

        # Test multiple K's
        target = torch.Tensor([[0, 1, 0, 1], [0, 1, 0, 1]]).transpose(0, 1)
        output = torch.Tensor([[.1, .2, .3, 4], [4, 3, 2, 1]]).transpose(0, 1)
        weight = torch.Tensor([[1.0, 0.5, 2.0, 3.0]]).transpose(0, 1)
        mtr.reset()
        mtr.add(output, target, weight)
        ap = mtr.value()
        self.assertTrue(
            math.fabs(ap -
                      torch.Tensor([
                          (1 * 3.0 / 3.0 + 0 * 3.0 / 5.0 +
                           3.5 * 1 / 5.5 + 0 * 3.5 / 6.5) / 2.0,
                          (0 * 1.0 / 1.0 + 1 * 0.5 / 1.5 +
                              0 * 0.5 / 3.5 + 1 * 3.5 / 6.5) / 2.0
                      ]).mean()) < 0.01, msg='mAP test3 failed')

        mtr.reset()
        mtr.add(output, target)
        ap = mtr.value()
        self.assertTrue(
            math.fabs(ap -
                      torch.Tensor([
                          (1 * 1.0 + 0 * 1.0 / 2.0 + 2 *
                           1.0 / 3.0 + 0 * 1.0 / 4.0) / 2.0,
                          (0 * 1.0 + 1 * 1.0 / 2.0 + 0 *
                              1.0 / 3.0 + 2 * 1.0 / 4.0) / 2.0
                      ]).mean()) < 0.01, msg='mAP test4 failed')


if __name__ == '__main__':
    unittest.main()
