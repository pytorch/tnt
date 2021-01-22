from . import meter
import numpy as np


class MultiLabelConfusionMeter(meter.Meter):
    """
    The MultiLabelConfusionMeter constructs a confusion matrix for a multi-class multi-label
    classification problem. For single labeled classification problem, please use ConfusionMeter.

    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not

    """

    def __init__(self, k, normalized=False):
        super(MultiLabelConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, 2, 2), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix of shape K x 2 x 2 where K is the number of classes

        Args:
            predicted (tensor): Must be an N x K tensor of predicted classes obtained from
                the model for N examples and K classes. The values must be 0/1. 
                The i'th row and j'th column represents a j'th label prediction for sample i
                if its value is 1 else if the value is 0 then the j'th label is not predicted. 
            target (tensor): Must be an N x K tensor of classes which represents the ground truth. 
                The values must be 0/1. The i'th row and j'th column represents a j'th label ground truth 
                for sample i if its value is 1 else if the value is 0 then the j'th label is not a ground truth.

        """

        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        assert np.ndim(predicted) == 2, \
            'predicted tensor must be of dimension 2'

        assert np.ndim(target) == 2, \
            'target tensor must be of dimension 2'

        assert predicted.shape[1] == self.k, \
            'number of prediction classes does not match the original number of classes'

        assert target.shape[1] == self.k, \
            'number of target classes does not match the original number of classes'

        num_samples = predicted.shape[0]

        assert int(self.k * num_samples) == ((predicted.astype(np.int32) == 0).sum() + (predicted.astype(np.int32) == 1).sum()), \
            'predicted array must be binary'

        assert int(self.k * num_samples) == ((target.astype(np.int32) == 0).sum() + (target.astype(np.int32) == 1).sum()), \
            'target array must be binary'

        target_total = np.count_nonzero(target, axis=0)
        predictions_total = np.count_nonzero(predicted, axis=0)
        tp = np.count_nonzero(np.multiply(target, predicted), axis=0)
        fp = predictions_total - tp
        fn = target_total - tp
        tn = target.shape[0] - tp - fp - fn

        self.conf += np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)


    def value(self):
        """
        Returns:
            Confusion matrix M of shape K x 2 x 2, where 
            M[i, 0, 0] corresponds to count/rate of true negatives of class i,
            M[i, 0, 1] corresponds to count/rate of false positives of class i,
            M[i, 1, 0] corresponds to count/rate of false negatives of class i,
            M[i, 1, 1] corresponds to count/rate of true positives of class i.
        """
        if self.normalized:
            conf = self.conf.astype(np.float64)
            sums = conf.sum(axis=2).sum(axis=1)
            return (conf.reshape(conf.shape[0], 4) / sums[:, None]).reshape(conf.shape)
        else:
            return self.conf
