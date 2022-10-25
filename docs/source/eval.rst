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

 from torchtnt.runner import EvalUnit

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

The :py:func:`~torchtnt.runner.evaluate` entry point takes a :class:`~torchtnt.runner.EvalUnit` object, a :class:`~torchtnt.runner.State` object, and an optional list of callbacks.

The :class:`~torchtnt.runner.State` object can be initialized with :func:`~torchtnt.runner.init_eval_state`, which takes in a dataloader (can be *any* iterable, including PyTorch DataLoader, numpy, etc.) and some parameters to control the run duration of the loop.

Below is an example of calling the :py:func:`~torchtnt.runner.evaluate` entry point with ``MyEvalUnit`` created above.

.. code-block:: python

 from torchtnt.runner import evaluate, init_eval_state

 eval_unit = MyEvalUnit(module=..., optimizer=..., lr_scheduler=...)
 dataloader = torch.utils.data.DataLoader(...)
 state = init_eval_state(dataloader=dataloader, max_steps_per_epoch=20)
 evaluate(state, eval_unit)
