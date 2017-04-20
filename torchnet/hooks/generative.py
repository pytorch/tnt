import torch
from .. import meter
from tqdm import tqdm

train_val = {}

def on_start(state):
    state['meters'] = {
        'loss': meter.AverageValueMeter(),
    }
    if state['train']:
        state['loggers'] = {
            'train_loss': [],
            'test_loss': [],
        }
    else:
        state['loggers'] = {}
    if not state['train']:
        state['meters']['loss'].reset()

def on_start_epoch(state):
    state['meters']['loss'].reset()
    state['iterator'] = tqdm(state['iterator'])

def on_forward(state):
    state['meters']['loss'].add(state['loss'].data[0])

def on_sample(state):
    state['sample'].append(state['train'])

def on_end_epoch(state):
    loss = state['meters']['loss'].value()[0]
    print('Training loss: %.4f' % loss)
    state['loggers']['train_loss'].append(loss)

def on_end(state):
    if not state['train']:
        loss = state['meters']['loss'].value()[0]
        print('Testing loss: %.4f' % loss)
        state['loggers']['test_loss'] = loss

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
