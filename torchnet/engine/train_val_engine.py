class TrainValEngine(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, engine, network, get_iterator, maxepoch, optimizer):
        state = {
                'engine': engine,
                'network': network,
                'get_iterator': get_iterator,
                'maxepoch': maxepoch,
                'optimizer': optimizer,
                'train': True,
                }

        on_end_epoch = engine.hooks['on_end_epoch']

        iterator_train = get_iterator(True)
        iterator_test = get_iterator(False)

        def new_on_end_epoch(state):
            on_end_epoch(state)
            teststate = engine.test(state['network'], iterator_test)
            for k,v in teststate['loggers'].items():
                state['loggers'][k].append(v)
                state['loggers'][k].append(v)

        engine.hooks['on_end_epoch'] = new_on_end_epoch

        self.hook('on_start', state)
        state['state'] = engine.train(state['network'], iterator_train, state['maxepoch'], state['optimizer'])
        self.hook('on_end', state)

        engine.hooks['on_end_epoch'] = on_end_epoch

        return state

    def test(self, engine, network, iterator):
        state = {
            'engine': engine,
            'network': network,
            'iterator': iterator,
            'train': False,
            }
        self.hook('on_start', state)
        state['state'] = engine.test(state['network'], state['iterator'])
        self.hook('on_end', state)
        return state

    def predict(self, engine, network, iterator):
        state = {
            'engine': engine,
            'network': network,
            'iterator': iterator,
            'train': False,
            }

        self.hook('on_start', state)
        teststate = engine.test(state['network'], state['iterator'])
        state['return'] = teststate['return']
        self.hook('on_end', state)
        return state['return']
