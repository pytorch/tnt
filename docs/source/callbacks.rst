Callbacks
=======================

In TorchTNT a :class:`~torchtnt.framework.callback.Callback` is an optional extension that can be used to supplement your loop with additional functionality. Good candidates
for such logic are ones that can be re-used across units. A Callback is simply a class with various hooks that are called during loop execution.

To benefit from a Callback, you must extend the Callback interface and implement the necessary hooks
to add your supplemental functionality. For a list of hooks which can be implemented see the :class:`~torchtnt.framework.callback.Callback` API reference.

Below is an example of a basic callback which can be included by the user to print the state of the loop at various points.

.. code-block:: python

 from torchtnt.framework.callback import Callback

 class PrintingCallback(Callback):
    def on_train_start(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        print("Starting training")

    def on_train_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        print("Ending training")

    def on_eval_start(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        print("Starting evaluation")

    def on_eval_end(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        print("Ending evaluation")

    def on_predict_start(self, state: State, unit: PredictUnit[TPredictData]) -> None:
        print("Starting prediction")

    def on_predict_end(self, state: State, unit: PredictUnit[TPredictData]) -> None:
        print("Ending prediction")

To include this in your loop, instantiate the class and pass it in the `callbacks` parameter to the :py:func:`~torchtnt.framework.train`, :py:func:`~torchtnt.framework.evaluate`,
:py:func:`~torchtnt.framework.predict`, or :py:func:`~torchtnt.framework.fit` entry point.

.. code-block:: python

 printing_callback = PrintingCallback()
 train(train_unit, train_dataloader, callbacks=[printing_callback])
