Fit
=======================
Here we lay out the steps needed to configure and run your fit loop. The fit loop interleaves training with evaluation,
to give you a closer look into how your model is performing.

TrainUnit and EvalUnit
~~~~~~~~~~~~~~~~~~~~~~~

In TorchTNT the Unit interface allows you to customize your fit loop when run by :py:func:`~torchtnt.runner.fit`.
For fit, you must create a class which subclasses :class:`~torchtnt.runner.unit.TrainUnit` and :class:`~torchtnt.runner.unit.EvalUnit`.
You must implement the ``train_step`` and ``eval_step`` methods on your class, and then you can optionally implement any of the hooks which allow you to control the behavior of the loop at different points.
Below is a simple example of a user's fit Unit that implements a basic ``train_step``, ``eval_step``, and the ``on_train_epoch_end`` hook.


.. code-block:: python

 from torchtnt.runner.unit import TrainUnit, EvalUnit

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

To run your fit loop, call the fit loop entry point: :py:func:`~torchtnt.runner.fit`.

The ``fit`` entry point takes as arguments one TrainUnit/EvalUnit, one iterable containing your training data and one iterable containing your eval
data (can be *any* iterable, including PyTorch DataLoader, numpy, etc.), an optional list of callbacks(described below), and several optional parameters to control
run duration of the loop.

Below is an example of calling the :py:func:`~torchtnt.runner.fit` entry point with the TrainUnit/EvalUnit created above.

.. code-block:: python

 from torchtnt.runner.fit import fit

 fit_unit = MyFitUnit(module=..., optimizer=..., lr_scheduler=...)
 train_dataloader = torch.utils.data.DataLoader(...)
 eval_dataloader = torch.utils.data.DataLoader(...)
 fit(fit_unit, train_dataloader, eval_dataloader, max_epochs=4)
