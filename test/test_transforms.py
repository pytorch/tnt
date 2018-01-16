import torchnet.transform as transform
import unittest
import torch


class TestTransforms(unittest.TestCase):
    def testCompose(self):
        self.assertEqual(transform.compose([lambda x: x + 1, lambda x: x + 2, lambda x: x / 2])(1), 2)

    def testTableMergeKeys(self):
        x = {
            'sample1': {'input': 1, 'target': "a"},
            'sample2': {'input': 2, 'target': "b", 'flag': "hard"}
        }

        y = transform.tablemergekeys()(x)

        self.assertEqual(y['input'], {'sample1': 1, 'sample2': 2})
        self.assertEqual(y['target'], {'sample1': "a", 'sample2': "b"})
        self.assertEqual(y['flag'], {'sample2': "hard"})

    def testTableApply(self):
        x = {1: 1, 2: 2}
        y = transform.tableapply(lambda x: x + 1)(x)
        self.assertEqual(y, {1: 2, 2: 3})

    def testMakeBatch(self):
        x = [
            {'input': torch.randn(4), 'target': "a"},
            {'input': torch.randn(4), 'target': "b"},
        ]
        y = transform.makebatch()(x)
        self.assertEqual(y['input'].size(), torch.Size([2, 4]))
        self.assertEqual(y['target'], ["a", "b"])


if __name__ == '__main__':
    unittest.main()
