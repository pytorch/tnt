from tqdm import tqdm

class Engine(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, network, iterator, maxepoch, optimizer):
        state = {
                'network': network,
                'iterator': iterator,
                'maxepoch': maxepoch,
                'optimizer': optimizer,
                'epoch': 0,
                't': 0,
                'train': True,
                }

        self.hook('onStart', state)
        while state['epoch'] < state['maxepoch']:
            self.hook('onStartEpoch', state)
            for sample in iterator:
                state['sample'] = sample
                self.hook('onSample', state)

                def closure():
                    loss, output = state['network'](state['sample'])
                    state['output'] = output
                    state['loss'] = loss
                    loss.backward()
                    self.hook('onForward', state)
                    return loss

                state['optimizer'].zero_grad()
                state['optimizer'].step(closure)
                state['t'] += 1
            state['epoch'] += 1
            self.hook('onEndEpoch', state)
        self.hook('onEnd', state)
        return state

    def test(self, network, iterator):
        state = {
            'network': network,
            'iterator': iterator,
            't': 0,
            'train': False,
            }

        self.hook('onStart', state)
        for sample in iterator:
            state['sample'] = sample
            self.hook('onSample', state)
          
            def closure():
                loss, output = state['network'](state['sample'])
                state['output'] = output
                state['loss'] = loss
                self.hook('onForward', state)
                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None

            closure()
        self.hook('onEnd', state)
        return state

