import unittest
import torch
import torchnet.dataset as dataset
import numpy as np
import os
import tempfile

class TestDatasets(unittest.TestCase):
    def testListDataset(self):
        identity = lambda x: x
        h = [0,1,2]
        d = dataset.ListDataset(elem_list = h, load = identity)
        self.assertEqual(len(d), 3)
        self.assertEqual(d[0], 0)

        t = torch.LongTensor([0,1,2])
        d = dataset.ListDataset(elem_list = t, load = identity)
        self.assertEqual(len(d), 3)
        self.assertEqual(d[0], 0)

        a = np.asarray([0,1,2])
        d = dataset.ListDataset(elem_list = a, load = identity)
        self.assertEqual(len(d), 3)
        self.assertEqual(d[0], 0)

    def testListDataset_path(self):
        prefix = lambda x: 'bar/' + str(x)
        tbl = [0,1,2]
        d = dataset.ListDataset(tbl, prefix, 'foo')
        self.assertEqual(len(d), 3)
        self.assertEqual(d[2], 'bar/foo/2')

    def testListDataset_file(self):
        _, filename = tempfile.mkstemp()
        with open(filename, 'w') as f:
            for i in range(0,50):
                f.write(str(i) + '\n')

        identity = lambda x: x
        d = dataset.ListDataset(filename, identity, 'foo')
        self.assertEqual(len(d), 50)
        self.assertEqual(d[15], 'foo/15')

        os.remove(filename)

    def testTensorDataset(self):
        # dict input
        data = {
                # 'input': torch.range(0,7),
                'input': np.arange(0,8),
                'target': np.arange(0,8),
                }
        d = dataset.TensorDataset(data)
        self.assertEqual(len(d), 8)
        self.assertEqual(d[2], {'input': 2, 'target': 2})

        # tensor input
        a = torch.randn(8)
        d = dataset.TensorDataset(a)
        self.assertEqual(len(a), len(d))
        self.assertEqual(a[1], d[1])

        # list of tensors input
        d = dataset.TensorDataset([a])
        self.assertEqual(len(a), len(d))
        self.assertEqual(a[1], d[1][0])

    def testBatchDataset(self):
        t = torch.range(0,15).long()
        batchsize = 8
        d = dataset.ListDataset(t, lambda x: {'input': x})
        d = dataset.BatchDataset(d, batchsize)
        ex = d[0]['input']
        self.assertEqual(len(ex), batchsize)
        self.assertEqual(ex[-1], batchsize - 1)

    # def testTransformDataset(self):
    #     d = dataset.TransformDataset(dataset.TensorDataset()

    def testResampleDataset(self):
        tbl = dataset.TensorDataset(np.asarray([0,1,2]))
        d = dataset.ResampleDataset(tbl, lambda dataset, i: i % 2)
        self.assertEqual(len(d), 3)
        self.assertEqual(d[0], 0)
        self.assertEqual(d[2], 0)

    def testShuffleDataset(self):
        tbl = dataset.TensorDataset(np.asarray([0,1,2,3,4]))
        d = dataset.ShuffleDataset(tbl)
        self.assertEqual(len(d), 5)
        # TODO: every item should appear exactly once

    def testSplitDataset(self):
        h = [0, 1, 2, 3]
        listdataset = dataset.ListDataset(elem_list=h)
        splitdataset = dataset.SplitDataset(listdataset, {'train': 3, 'val': 1})

        splitdataset.select('train')
        self.assertEqual(len(splitdataset), 3)
        self.assertEqual(splitdataset[2], 2)

        splitdataset.select('val')
        self.assertEqual(len(splitdataset), 1)
        self.assertEqual(splitdataset[0], 3)

        # test fluent api
        splitdataset = listdataset.split({'train': 3, 'val': 1})
        splitdataset.select('train')
        self.assertEqual(len(splitdataset), 3)
        self.assertEqual(splitdataset[2], 2)

    def testSplitDataset_fractions(self):
        h = [0, 1, 2, 3]
        listdataset = dataset.ListDataset(elem_list=h)
        splitdataset = dataset.SplitDataset(listdataset, {'train': 0.75,
                                                          'val': 0.25})

        splitdataset.select('train')
        self.assertEqual(len(splitdataset), 3)
        self.assertEqual(splitdataset[2], 2)

        splitdataset.select('val')
        self.assertEqual(len(splitdataset), 1)
        self.assertEqual(splitdataset[0], 3)

    def testConcatDataset(self):
        l1 = dataset.ListDataset(elem_list=[0, 1, 2, 3])
        l2 = dataset.ListDataset(elem_list=[10, 11, 13])
        concatdataset = dataset.ConcatDataset([l1, l2])

        self.assertEqual(len(concatdataset), 7)
        self.assertEqual(concatdataset[0], 0)
        self.assertEqual(concatdataset[3], 3)
        self.assertEqual(concatdataset[4], 10)
        self.assertEqual(concatdataset[6], 13)



if __name__ == '__main__':
    unittest.main()
