Callbacks
=======================

.. automodule:: torchtnt.framework.callback
   :members:
   :undoc-members:


Built-in callbacks
~~~~~~~~~~~~~~~~~~~~~

We offer several pre-written callbacks which are ready to be used out of the box:


.. currentmodule:: torchtnt.framework.callbacks

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class_template.rst

    BaseCSVWriter
    DataloaderProfiler
    EarlyStopping
    GarbageCollector
    Lambda
    LearningRateMonitor
    MemorySnapshot
    ModuleSummary
    PyTorchProfiler
    SlowRankDetector
    SystemResourcesMonitor
    TensorBoardParameterMonitor
    TimeLimitInterrupter
    IterationTimeLogger
    TorchSnapshotSaver
    TQDMProgressBar
    TrainProgressMonitor
