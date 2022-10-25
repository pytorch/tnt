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

 from torchtnt.runner import PredictUnit

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

The :py:func:`~torchtnt.runner.predict` entry point takes a :class:`~torchtnt.runner.PredictUnit` object, a :class:`~torchtnt.runner.State` object, and an optional list of callbacks.

The :class:`~torchtnt.runner.State` object can be initialized with :func:`~torchtnt.runner.init_predict_state`, which takes in a dataloader (can be *any* iterable, including PyTorch DataLoader, numpy, etc.) and some parameters to control the run duration of the loop.

Below is an example of calling the :py:func:`~torchtnt.runner.predict` entry point with ``MyPredictUnit`` created above.

.. code-block:: python

 from torchtnt.runner import init_predict_state, predict

 predict_unit = MyPredictUnit(module=..., optimizer=..., lr_scheduler=...)
 dataloader = torch.utils.data.DataLoader(...)
 state = init_predict_state(dataloader=dataloader, max_steps_per_epoch=20)
 predict(state, predict_unit)
