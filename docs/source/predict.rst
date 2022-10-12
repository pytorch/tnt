Predict
=======================
Here we lay out the steps needed to configure and run your prediction loop.

PredictUnit
~~~~~~~~~~~~~

In TorchTNT, :class:`~torchtnt.runner.unit.PredictUnit` is the interface that allows you to customize your prediction loop when run by :py:func:`~torchtnt.runner.predict`.
To use, you must create a class which subclasses :class:`~torchtnt.runner.unit.PredictUnit`.
You must implement the ``predict_step`` method on your class, and then you can optionally implement any of the hooks which allow you to control the behavior of the loop at different points.
Below is a simple example of a user's subclass of :class:`~torchtnt.runner.unit.PredictUnit` that implements a basic ``predict_step``.


.. code-block:: python

 from torchtnt.runner.unit import PredictUnit

 class MyPredictUnit(PredictUnit[Batch]):
     def __init__(
         self,
         module: torch.nn.Module,
     ):
         super().__init__()
         self.module = module

     def predict_step(self, state: State, data: Batch) -> None:
         inputs, targets = data
         outputs = self.module(inputs)

 predict_unit = MyPredictUnit(module=...)

Predict Entry Point
~~~~~~~~~~~~~~~~~~~~

To run your prediction loop, call the prediction loop entry point: :py:func:`~torchtnt.runner.predict`.

The ``predict`` entry point takes as arguments one PredictUnit, one iterable containing your data (can be *any* iterable, including PyTorch DataLoader, numpy, etc.), an optional list of callbacks
(described below), and several optional parameters to control run duration of the loop.

Below is an example of calling the :py:func:`~torchtnt.runner.predict` entry point with the ``PredictUnit`` created above.

.. code-block:: python

 from torchtnt.runner.predict import predict

 predict_unit = MyPredictUnit(module=..., optimizer=..., lr_scheduler=...)
 predict_dataloader = torch.utils.data.DataLoader(...)
 predict(predict_unit, predict_dataloader, max_steps_per_epoch=20)
