from . import meter
import numpy as np
import numbers

class ConfusionMeter(meter.Meter):
    """
    <a name="ConfusionMeter">
    #### tnt.ConfusionMeter(@ARGP)
    @ARGT

    The `tnt.ConfusionMeter` constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use `tnt.MultiLabelConfusionMeter`.

    At initialization time, the `k` parameter that indicates the number
    of classes in the classification problem under consideration must be specified.
    Additionally, an optional parameter `normalized` (default = `false`) may be
    specified that determines whether or not the confusion matrix is normalized
    (that is, it contains percentages) or not (that is, it contains counts).

    The `add(output, target)` method takes as input an NxK tensor `output` that
    contains the output scores obtained from the model for N examples and K classes,
    and a corresponding N-tensor or NxK-tensor `target` that provides the targets
    for the N examples. When `target` is an N-tensor, the targets are assumed to be
    integer values between 1 and K. When target is an NxK-tensor, the targets are
    assumed to be provided as one-hot vectors (that is, vectors that contain only
    zeros and a single one at the location of the target value to be encoded).

    The `value()` method has no parameters and returns the confusion matrix in a
    KxK tensor. In the confusion matrix, rows correspond to ground-truth targets and
    columns correspond to predicted targets.
    """
    def __init__(self, k, normalized = False):
        self.conf = np.ndarray((k,k), dtype=np.int32)
        self.normalized = normalized
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, output, target):
        output = output.cpu().squeeze().numpy()
        target = target.cpu().squeeze().numpy()
        if np.ndim(output) == 1:
            output = output[None]
        onehot = np.ndim(target) != 1
        assert output.shape[0] == target.shape[0], \
                'number of targets and outputs do not match'
        assert output.shape[1] == self.conf.shape[0], \
                'number of outputs does not match size of confusion matrix'
        assert not onehot or target.shape[1] == output.shape[1], \
                'target should be 1D Tensor or have size of output (one-hot)'
        if onehot:
            assert (target >= 0).all() and (target <= 1).all(), \
                    'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                    'multi-label setting is not supported'
 
        pred = output.argmax(1)
        for i,n in enumerate(pred):
            pos = onehot and target[i].argmax(0) or int(target[i])
            self.conf[pos][n] += 1

    def value(self):
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:,None]
        else:
            return self.conf

