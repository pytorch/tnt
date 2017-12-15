""" Logging values to various sinks """


class Logger(object):
    _fields = None

    @property
    def fields(self):
        assert self._fields is not None, "self.fields is not set!"
        return self._fields

    @fields.setter
    def fields(self, value):
        self._fields

    def __init__(self, fields=None):
        """ Automatically logs the variables in 'fields' """
        self.fields = fields

    def log(self, *args, **kwargs):
        pass

    def log_state(self, state_dict):
        pass
