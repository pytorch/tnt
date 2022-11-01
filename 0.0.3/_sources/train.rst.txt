Train
=======================
Here we lay out the steps needed to configure and run your training loop.

TrainUnit
~~~~~~~~~~~~~

In TorchTNT, :class:`~torchtnt.runner.unit.TrainUnit` is the interface that allows you to customize your training loop when run by :py:func:`~torchtnt.runner.train`.
To use, you must create a class which subclasses :class:`~torchtnt.runner.unit.TrainUnit`.
You must implement the ``train_step`` method on your class, and then you can optionally implement any of the hooks which allow you to control the behavior of the loop at different points.
Below is a simple example of a user's subclass of :class:`~torchtnt.runner.unit.TrainUnit` that implements a basic ``train_step``, and the ``on_train_epoch_end`` hook.


.. code-block:: python

 from torchtnt.runner import TrainUnit

 class MyTrainUnit(TrainUnit[Batch]):
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

     def on_train_epoch_end(self, state: State) -> None:
        # step the learning rate scheduler
        self.lr_scheduler.step()

 train_unit = MyTrainUnit(module=..., optimizer=..., lr_scheduler=...)

Train Entry Point
~~~~~~~~~~~~~~~~~~~~

To run your training loop, call the training loop entry point: :py:func:`~torchtnt.runner.train`.

The :py:func:`~torchtnt.runner.train` entry point takes a :class:`~torchtnt.runner.TrainUnit` object, a :class:`~torchtnt.runner.State` object, and an optional list of callbacks.

The :class:`~torchtnt.runner.State` object can be initialized with :func:`~torchtnt.runner.init_train_state`, which takes in a dataloader (can be *any* iterable, including PyTorch DataLoader, numpy, etc.) and some parameters to control the run duration of the loop.

Below is an example of calling the :py:func:`~torchtnt.runner.train` entry point with ``MyTrainUnit`` created above.

.. code-block:: python

 from torchtnt.runner import init_train_state, train

 train_unit = MyTrainUnit(module=..., optimizer=..., lr_scheduler=...)
 dataloader = torch.utils.data.DataLoader(...)
 state = init_train_state(dataloader=dataloader, max_epochs=4)
 train(state, train_unit)
