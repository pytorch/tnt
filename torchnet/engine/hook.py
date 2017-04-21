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