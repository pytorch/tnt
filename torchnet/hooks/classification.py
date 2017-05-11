import torch
from .. import meter
from tqdm import tqdm

train_val = {}

def on_start(state):
    state['meters'] = {
        'loss': meter.AverageValueMeter(),
        'classacc': meter.ClassErrorMeter(accuracy=True)
    }
    if state['train']:
        state['loggers'] = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': [],
        }
    else:
        state['loggers'] = {}
    if not state['train']:
        state['meters']['classacc'].reset()
        state['meters']['loss'].reset()

def on_start_epoch(state):
    state['meters']['classacc'].reset()
    state['meters']['loss'].reset()
    state['iterator'] = tqdm(state['iterator'])

def on_forward(state):
    state['meters']['classacc'].add(state['output'].data, torch.LongTensor(state['sample'][1]))
    state['meters']['loss'].add(state['loss'].data[0])

def on_sample(state):
    state['sample'].append(state['train'])

def on_end_epoch(state):
    loss = state['meters']['loss'].value()[0]
    acc = state['meters']['classacc'].value()[0]
    print('Training loss: %.4f, Training accuracy: %.2f%%' % (loss,acc))
    state['loggers']['train_loss'].append(loss)
    state['loggers']['train_acc'].append(acc)

def on_end(state):
    if not state['train']:
        loss = state['meters']['loss'].value()[0]
        acc = state['meters']['classacc'].value()[0]
        print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (loss,acc))
        state['loggers']['test_loss'] = loss
        state['loggers']['test_acc'] = acc

train_val['on_start'] = on_start
train_val['on_sample'] = on_sample
train_val['on_forward'] = on_forward
train_val['on_start_epoch'] = on_start_epoch
train_val['on_end_epoch'] = on_end_epoch
train_val['on_end'] = on_end

################################################################################

prediction_simple = {}

def on_start(state):
    state['predictions'] = []
    state['iterator'] = tqdm(state['iterator'])

def on_forward(state):
    _, argmax = state['output'].data.max(1)
    state['predictions'].append(argmax.squeeze())

def on_sample(state):
    state['sample'].append(state['train'])

def on_end(state):
    state['predictions'] = torch.cat(state['predictions'])
    state['return'] = state['predictions']

prediction_simple['on_start'] = on_start
prediction_simple['on_sample'] = on_sample
prediction_simple['on_forward'] = on_forward
prediction_simple['on_end'] = on_end

################################################################################
