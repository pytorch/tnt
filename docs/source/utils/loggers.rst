Loggers
=============

.. automodule:: torchtnt.utils.loggers

.. autoclass:: torchtnt.utils.loggers.FileLogger
   :members:
   :undoc-members:

.. autoclass:: torchtnt.utils.loggers.MetricLogger
   :members:
   :undoc-members:

Built-in loggers
~~~~~~~~~~~~~~~~~~~~~

Loggers are wrappers around different publishers to simplify logging metrics. While users can create their own loggers by subclassing the loggers above, we also offer the following loggers out of the box:

.. currentmodule:: torchtnt.utils.loggers

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class_template.rst

    CSVLogger
    InMemoryLogger
    JSONLogger
    TensorBoardLogger
