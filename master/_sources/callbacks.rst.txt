Callbacks
=======================

What is a callback and how is it used
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In TorchTNT a :class:`~torchtnt.framework.callback.Callback` is an optional extension that can be used to supplement your loop with additional functionality. Good candidates
for such logic are ones that can be re-used across units. A Callback is simply a class with various hooks that are called during loop execution.

Below is an example of a basic callback which prints a message at various points during execution.

.. code-block:: python

 from torchtnt.framework import Callback, State, TEvalUnit, TPredictUnit, TTrainUnit

 class PrintingCallback(Callback):
    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        print("Starting training")

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        print("Ending training")

    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        print("Starting evaluation")

    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        print("Ending evaluation")

    def on_predict_start(self, state: State, unit: TPredictUnit) -> None:
        print("Starting prediction")

    def on_predict_end(self, state: State, unit: TPredictUnit) -> None:
        print("Ending prediction")

To use a callback, instantiate the class and pass it in the ``callbacks`` parameter to the :py:func:`~torchtnt.framework.train`, :py:func:`~torchtnt.framework.evaluate`,
:py:func:`~torchtnt.framework.predict`, or :py:func:`~torchtnt.framework.fit` entry point.

.. code-block:: python

 printing_callback = PrintingCallback()
 train(train_unit, train_dataloader, callbacks=[printing_callback])


Built-in callbacks
~~~~~~~~~~~~~~~~~~~~~

We offer several pre-written callbacks which are ready to be used out of the box:


.. currentmodule:: torchtnt.framework.callbacks

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class_template.rst

    BaseCSVWriter
    GarbageCollector
    Lambda
    LearningRateMonitor
    ModuleSummary
    PyTorchProfiler
    TensorBoardParameterMonitor
    TorchSnapshotSaver
    TQDMProgressBar

Writing your own callback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To write your own callback, subclass the :class:`~torchtnt.framework.callback.Callback` class and add your own code into the hooks.

.. automodule:: torchtnt.framework.callback
   :members:
   :undoc-members:
