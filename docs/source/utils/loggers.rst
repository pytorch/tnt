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

Loggers are wrappers around different publishers to simplify logging metrics. We offer the following loggers:

.. currentmodule:: torchtnt.utils.loggers

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class_template.rst

    CSVLogger
    InMemoryLogger
    JSONLogger
    TensorBoardLogger
