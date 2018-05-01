.. role:: hidden
    :class: hidden-section

torchnet.logger
========================

.. automodule:: torchnet.logger
.. currentmodule:: torchnet.logger

Loggers provide a way to monitor your models. For example, the :class:`MeterLogger` class
provides easy meter visualizetion with `Visdom <https://github.com/facebookresearch/visdom>`_ ,
as well as the ability to print and save meters with the :class:`ResultsWriter` class. 

For visualization libraries, the current loggers support ``Visdom``, although ``TensorboardX`` 
would also be simple to implement. 


MeterLogger
~~~~~~~~~~~~~~~~~

.. autoclass:: MeterLogger
    :members:
    :undoc-members:
    :show-inheritance:

VisdomLogger
~~~~~~~~~~~~~~~
.. automodule:: torchnet.logger.visdomlogger
    :members:
    :undoc-members:
    :show-inheritance:

