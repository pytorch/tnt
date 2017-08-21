from . import meter
import numpy as np
import numbers


class ConfusionMeter(meter.Meter):
    """
    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.
    """

    def __init__(self, k, normalized=False):
        """
        Args:
            k (int): number of classes in the classification problem
            normalized: Determines whether or not the confusion matrix
                is normalized or not
        """
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """
        Computes the confusion matrix of K x K size where K is no of classes
        Args:
            predicted: Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of integer
                values between 1 and K.
            target: Can be a N-tensor of integer values assumed to be integer
                values between 1 and K or N x K tensor, where targets are
                assumed to be provided as one-hot vectors

        """
        output = output.cpu().squeeze().numpy()
        target = target.cpu().squeeze().numpy()
        onehot = np.ndim(target) != 1
        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'
        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and K '

        assert not onehot or target.shape[1] == predicted.shape[1], \
            'target should be 1D Tensor or have size of predicted (one-hot)'
        if onehot:
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)

        for i in range(self.k):
            for j in range(self.k):
                self.conf[i, j] += np.sum((target == i) * (predicted == j))

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf
