""" Run MNIST example and log to visdom
    Notes:
        - Visdom must be installed (pip works)
        - the Visdom server must be running at start!

    Example:
        $ python -m visdom.server -port 8097 &
        $ python mnist_with_visdom.py
"""
from tqdm import tqdm
import torch
import torch.optim
import torchnet as tnt
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import kaiming_normal
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.datasets.mnist import MNIST


def get_iterator(mode):
    ds = MNIST(root='./', download=True, train=mode)
    data = getattr(ds, 'train_data' if mode else 'test_data')
    labels = getattr(ds, 'train_labels' if mode else 'test_labels')
    tds = tnt.dataset.TensorDataset([data, labels])
    return tds.parallel(batch_size=128, num_workers=4, shuffle=mode)


def conv_init(ni, no, k):
    return kaiming_normal(torch.Tensor(no, ni, k, k))


def linear_init(ni, no):
    return kaiming_normal(torch.Tensor(no, ni))


def f(params, inputs, mode):
    o = inputs.view(inputs.size(0), 1, 28, 28)
    o = F.conv2d(o, params['conv0.weight'], params['conv0.bias'], stride=2)
    o = F.relu(o)
    o = F.conv2d(o, params['conv1.weight'], params['conv1.bias'], stride=2)
    o = F.relu(o)
    o = o.view(o.size(0), -1)
    o = F.linear(o, params['linear2.weight'], params['linear2.bias'])
    o = F.relu(o)
    o = F.linear(o, params['linear3.weight'], params['linear3.bias'])
    return o


def main():
    params = {
        'conv0.weight': conv_init(1, 50, 5), 'conv0.bias': torch.zeros(50),
        'conv1.weight': conv_init(50, 50, 5), 'conv1.bias': torch.zeros(50),
        'linear2.weight': linear_init(800, 512), 'linear2.bias': torch.zeros(512),
        'linear3.weight': linear_init(512, 10), 'linear3.bias': torch.zeros(10),
    }
    params = {k: Variable(v, requires_grad=True) for k, v in params.items()}

    optimizer = torch.optim.SGD(
        params.values(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    classerr = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(10, normalized=True)

    port = 8097
    train_loss_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Train Loss'})
    train_err_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Train Class Error'})
    test_loss_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Test Loss'})
    test_err_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Test Class Error'})
    confusion_logger = VisdomLogger('heatmap', port=port, opts={'title': 'Confusion matrix',
                                                                'columnnames': list(range(10)),
                                                                'rownames': list(range(10))})

    def h(sample):
        inputs = Variable(sample[0].float() / 255.0)
        targets = Variable(torch.LongTensor(sample[1]))
        o = f(params, inputs, sample[2])
        return F.cross_entropy(o, targets), o

    def reset_meters():
        classerr.reset()
        meter_loss.reset()
        confusion_meter.reset()

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        classerr.add(state['output'].data,
                     torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data,
                            torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])

    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        print('Training loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))
        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_err_logger.log(state['epoch'], classerr.value()[0])

        # do validation at the end of each epoch
        reset_meters()
        engine.test(h, get_iterator(False))
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_err_logger.log(state['epoch'], classerr.value()[0])
        confusion_logger.log(confusion_meter.value())
        print('Testing loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(h, get_iterator(True), maxepoch=10, optimizer=optimizer)


if __name__ == '__main__':
    main()
