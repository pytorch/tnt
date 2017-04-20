from __future__ import print_function

import sys
sys.path.append('../')

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import transforms

from torchnet.engine import Engine, TrainValEngine
from torchnet import hooks

def get_iterator(mode):
    ds = MNIST(root='./', train=mode, download=True,
       transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ]))
    return DataLoader(ds, batch_size=128, shuffle=mode, num_workers=4)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight.data)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    net = Net()
    net.apply(weights_init)

    def h(sample):
        inputs = Variable(sample[0].float())
        targets = Variable(torch.LongTensor(sample[1]))
        o = net(inputs)
        return F.cross_entropy(o, targets), o

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    engine = Engine()
    hooks.add_hooks(engine, hooks.classification.train_val)
    hooks.add_hooks(engine, hooks.checkpoint.model_and_optim(net, optimizer, 'checkpoint.tar.pth'))

    metaengine = TrainValEngine()
    metaengine.train(engine, h, get_iterator, 10, optimizer)
    metaengine.test(engine, h, get_iterator(False))
    # To get output
    hooks.replace_hooks(engine, hooks.classification.prediction_simple)
    predictions = metaengine.predict(engine, h, get_iterator(False))
    print((predictions.size()))
