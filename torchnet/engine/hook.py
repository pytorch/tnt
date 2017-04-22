class StopEngine(Exception):
    pass

class Hook(object):
    def __init__(self):
        pass

    def on_start(self, state):
        pass

    def on_start_epoch(self, state):
        pass

    def on_sample(self, state):
        pass

    def on_forward(self, state):
        pass

    def on_end_epoch(self, state):
        pass

    def on_end(self, state):
        pass


class HooksList(Hook):
    def __init__(self, hooks=None):
        super(HooksList, self).__init__()
        self.hooks = hooks or None

    def append(self, hook):
        self.hooks.append(hook)

    def on_start(self, state):
        for hook in self.hooks:
            hook.on_start(state)

    def on_start_epoch(self, state):
        for hook in self.hooks:
            hook.on_start_epoch(state)

    def on_sample(self, state):
        for hook in self.hooks:
            hook.on_sample(state)

    def on_forward(self, state):
        for hook in self.hooks:
            hook.on_forward(state)

    def on_end_epoch(self, state):
        for hook in self.hooks:
            hook.on_end_epoch(state)

    def on_end(self, state):
        for hook in self.hooks:
            hook.on_end(state)

    def __iter__(self):
        return iter(self.hooks)


class LearningRateScheduler(Hook):
    """Learning rate scheduler.

    Parameters
    ----------
    schedule: function
        a function that takes an epoch index as input (integer, indexed from 0)
        and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        assert callable(schedule), 'schedule should be callable'
        self.schedule = schedule

    def on_start_epoch(self, state):
        optimizer = state['optimizer']
        epoch = state['epoch']
        lr = self.schedule(epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        state['optimizer'] = optimizer
