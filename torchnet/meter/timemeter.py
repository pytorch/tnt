import time
from . import meter


class TimeMeter(meter.Meter):
    """
    <a name="TimeMeter">
    #### tnt.TimeMeter(@ARGP)
    @ARGT

    The `tnt.TimeMeter` is designed to measure the time between events and can be
    used to measure, for instance, the average processing time per batch of data.
    It is different from most other meters in terms of the methods it provides:

    The `tnt.TimeMeter` provides the following methods:

       * `reset()` resets the timer, setting the timer and unit counter to zero.
       * `value()` returns the time passed since the last `reset()`; divided by the counter value when `unit=true`.
    """

    def __init__(self, unit):
        super(TimeMeter, self).__init__()
        self.unit = unit
        self.reset()

    def reset(self):
        self.n = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time
