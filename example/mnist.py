from tqdm import tqdm
import math
import torch
import torch.optim
import torchnet as tnt
from torchnet.engine import Engine
from torch.utils.serialization.read_lua_file import load_lua
from torch.autograd import Variable
import torch.nn.functional as F


def get_iterator(mnist, mode):
    ds = mnist.train if mode else mnist.test
    tds = tnt.dataset.TensorDataset([ds.data, ds.label])
    return tds.parallel(batch_size=128, num_workers=4, shuffle=mode)


def conv_init(ni, no, k):
    return torch.Tensor(no, ni, k, k).normal_(0, 2/math.sqrt(ni*k*k))


def linear_init(ni, no):
    return torch.Tensor(no, ni).normal_(0, 2/math.sqrt(ni))


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
    # mnist = require 'mnist'
    # torch.save('./example/mnist.t7',{train = mnist.traindataset(), test = mnist.testdataset()})
    mnist = load_lua('./example/mnist.t7')

    meter_loss = tnt.meter.AverageValueMeter()
    classerr = tnt.meter.ClassErrorMeter(accuracy=True)

    params = {
            'conv0.weight':     conv_init(1, 50, 5),
            'conv0.bias':       torch.zeros(50),
            'conv1.weight':     conv_init(50, 50, 5),
            'conv1.bias':       torch.zeros(50),
            'linear2.weight':   linear_init(800, 512),
            'linear2.bias':     torch.zeros(512),
            'linear3.weight':   linear_init(512, 10),
            'linear3.bias':     torch.zeros(10),
            }

    for k, v in params.items():
        params[k] = Variable(v, requires_grad=True)

    def h(sample):
        inputs = Variable(sample[0].float() / 255.0)
        targets = Variable(torch.LongTensor(sample[1]))
        o = f(params, inputs, sample[2])
        return F.cross_entropy(o, targets), o

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        classerr.add(state['output'].data,
                     torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])

    def on_start_epoch(state):
        classerr.reset()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        print classerr.value()

    optimizer = torch.optim.SGD(params.values(), lr=0.01, momentum=0.9,
                                weight_decay=0.0005)

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(h, get_iterator(mnist, True), 10, optimizer)
    engine.test(h, get_iterator(mnist, False))


if __name__ == '__main__':
    main()
