class Engine(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, network, iterator, maxepoch, optimizer, step_interval=1):
        state = {
                'network': network,
                'iterator': iterator,
                'maxepoch': maxepoch,
                'optimizer': optimizer,
                'epoch': 0,
                't': 0,
                'train': True,
                }

        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)
            state['optimizer'].zero_grad()

            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_sample', state)

                def closure():
                    loss, output = state['network'](state['sample'])
                    state['output'] = output
                    state['loss'] = loss
                    loss.backward()
                    self.hook('on_forward', state)
                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None
                    return loss

                if step_interval == 1:
                    state['optimizer'].zero_grad()
                    state['optimizer'].step(closure)
                    self.hook('on_step', state)
                else:
                    closure()
                    if (state['t'] + 1) % step_interval == 0:
                        state['optimizer'].step()
                        state['optimizer'].zero_grad()
                        self.hook('on_step', state)

                state['t'] += 1

            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state

    def test(self, network, iterator):
        state = {
            'network': network,
            'iterator': iterator,
            't': 0,
            'train': False,
            }

        self.hook('on_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_sample', state)

            def closure():
                loss, output = state['network'](state['sample'])
                state['output'] = output
                state['loss'] = loss
                self.hook('on_forward', state)
                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None

            closure()
            state['t'] += 1
        self.hook('on_end', state)
        return state
