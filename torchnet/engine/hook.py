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
