Evaluate
=======================
Here we lay out the steps needed to configure and run your evaluation loop.

EvalUnit
~~~~~~~~~~~~~

In TorchTNT, :class:`~torchtnt.runner.unit.EvalUnit` is the interface that allows you to customize your evaluation loop when run by :py:func:`~torchtnt.runner.evaluate`.
To use, you must create a class which subclasses :class:`~torchtnt.runner.unit.EvalUnit`.
You must implement the ``eval_step`` method on your class, and then you can optionally implement any of the hooks which allow you to control the behavior of the loop at different points.
Below is a simple example of a user's subclass of :class:`~torchtnt.runner.unit.EvalUnit` that implements a basic ``eval_step``.


.. code-block:: python

 from torchtnt.runner.unit import EvalUnit

 class MyEvalUnit(EvalUnit[Batch]):
     def __init__(
         self,
         module: torch.nn.Module,
     ):
         super().__init__()
         self.module = module

     def eval_step(self, state: State, data: Batch) -> None:
         inputs, targets = data
         outputs = self.module(inputs)
         loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)

 eval_unit = MyEvalUnit(module=...)

Evaluate Entry Point
~~~~~~~~~~~~~~~~~~~~

To run your evaluation loop, call :py:func:`~torchtnt.runner.evaluate`.

The :py:func:`~torchtnt.runner.evaluate` entry point takes as arguments one EvalUnit, one iterable containing your data (can be *any* iterable, including PyTorch DataLoader, numpy, etc.), an optional list of callbacks
(described below), and several optional parameters to control run duration of the loop.

Below is an example of calling the :py:func:`~torchtnt.runner.evaluate` entry point with the ``EvalUnit`` created above.

.. code-block:: python

 from torchtnt.runner.evaluate import evaluate

 eval_unit = MyEvalUnit(module=..., optimizer=..., lr_scheduler=...)
 eval_dataloader = torch.utils.data.DataLoader(...)
 evaluate(eval_unit, eval_dataloader, max_steps_per_epoch=20)
