.. role:: hidden
    :class: hidden-section

torchnet.meter
======================

.. automodule:: torchnet.meter
.. currentmodule:: torchnet.meter

Meters provide a way to keep track of important statistics in an online manner.
TNT also provides convenient ways to visualize and manage meters via the :class:`torchnet.logger.MeterLogger` class.

.. autoclass:: torchnet.meter.meter.Meter
    :members:

Classification Meters
------------------------------


:hidden:`APMeter`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: APMeter
    :members:

:hidden:`mAPMeter`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: mAPMeter
    :members:

:hidden:`ClassErrorMeter`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ClassErrorMeter
    :members:

:hidden:`ConfusionMeter`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ConfusionMeter
    :members:


Regression/Loss Meters
------------------------------

:hidden:`AverageValueMeter`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: AverageValueMeter
    :members:

:hidden:`AUCMeter`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: AUCMeter
    :members:

:hidden:`MovingAverageValueMeter`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MovingAverageValueMeter
    :members:

:hidden:`MSEMeter`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MSEMeter
    :members:





Miscellaneous Meters
------------------------------

:hidden:`TimeMeter`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: TimeMeter
    :members:
