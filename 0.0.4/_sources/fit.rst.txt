Fit
=======================
Here we lay out the steps needed to configure and run your fit loop. The fit loop interleaves training with evaluation,
to give you immediate feedback about how your model is performing while training.

TrainUnit and EvalUnit
~~~~~~~~~~~~~~~~~~~~~~~

In TorchTNT the Unit interface allows you to customize your fit loop when run by :py:func:`~torchtnt.framework.fit`.
For fit, you must create a class which subclasses :class:`~torchtnt.framework.unit.TrainUnit` and :class:`~torchtnt.framework.unit.EvalUnit`.
You must implement the ``train_step`` and ``eval_step`` methods on your class, and then you can optionally implement any of the hooks which allow you to control the behavior of the loop at different points.
Below is a simple example of a user's fit Unit that implements a basic ``train_step``, ``eval_step``, and the ``on_train_epoch_end`` hook.


.. code-block:: python

 from torchtnt.framework import TrainUnit, EvalUnit

 class MyFitUnit(TrainUnit[Batch]), EvalUnit[Batch]:
     def __init__(
         self,
         module: torch.nn.Module,
         optimizer: torch.optim.Optimizer,
         lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
     ):
         super().__init__()
         self.module = module
         self.optimizer = optimizer
         self.lr_scheduler = lr_scheduler

     def train_step(self, state: State, data: Batch) -> None:
         inputs, targets = data
         outputs = self.module(inputs)
         loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
         loss.backward()

         self.optimizer.step()
         self.optimizer.zero_grad()

    def eval_step(self, state: State, data: Batch) -> None:
         inputs, targets = data
         outputs = self.module(inputs)
         loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)

     def on_train_epoch_end(self, state: State) -> None:
        # step the learning rate scheduler
        self.lr_scheduler.step()

 fit_unit = MyFitUnit(module=..., optimizer=..., lr_scheduler=...)

Fit Entry Point
~~~~~~~~~~~~~~~~~~~~

To run your fit loop, call the fit loop entry point: :py:func:`~torchtnt.framework.fit`.

The :py:func:`~torchtnt.framework.fit` entry point takes an object subclassing both :class:`~torchtnt.framework.TrainUnit` and :class:`~torchtnt.framework.EvalUnit`, a :class:`~torchtnt.framework.state.State` object, and an optional list of callbacks.

The :class:`~torchtnt.framework.state.State` object can be initialized with :func:`~torchtnt.framework.init_fit_state`, which takes in a dataloader (can be *any* iterable, including PyTorch DataLoader, numpy, etc.) and some parameters to control the run duration of the loop.

Below is an example of calling the :py:func:`~torchtnt.framework.fit` entry point with the TrainUnit/EvalUnit created above.

.. code-block:: python

 from torchtnt.framework import fit, init_fit_state

 fit_unit = MyFitUnit(module=..., optimizer=..., lr_scheduler=...)
 train_dataloader = torch.utils.data.DataLoader(...)
 eval_dataloader = torch.utils.data.DataLoader(...)
 state = init_fit_state(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, max_epochs=4)
 fit(state, fit_unit)
