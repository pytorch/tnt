Unit
=========

The Unit concept represents the primary place to organize your model code in TorchTNT. TorchTNT offers three different types of Unit classes for training, evaluation, and prediction. These interfaces are mutually exclusive and can be combined as needed, e.g. in the case of fitting (interleaving training and evaluation).

TrainUnit
~~~~~~~~~~~~~~~~~
.. autoclass:: torchtnt.framework.unit.TrainUnit
   :members:
   :undoc-members:

EvalUnit
~~~~~~~~~~~~~~~~~
.. autoclass:: torchtnt.framework.unit.EvalUnit
   :members:
   :undoc-members:

PredictUnit
~~~~~~~~~~~~~~~~~
.. autoclass:: torchtnt.framework.unit.PredictUnit
   :members:
   :undoc-members:

Combining Multiple Units
~~~~~~~~~~~~~~~~~~~~~~~~~~
In some cases, it is convenient to implement multiple Unit interfaces under the same class, e.g. if you plan to use your class to run several different phases;
for example, running training and then prediction, or running training and evaluation interleaved (referred to as fitting).
Here is an example of a unit which extends TrainUnit, EvalUnit, and PredictUnit.

.. code-block:: python

 from torchtnt.framework.unit import TrainUnit, EvalUnit, PredictUnit

 Batch = Tuple[torch.tensor, torch.tensor]

 class MyUnit(TrainUnit[Batch], EvalUnit[Batch], PredictUnit[Batch]):
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

    def predict_step(self, state: State, data: Batch) -> torch.tensor:
         inputs, targets = data
         outputs = self.module(inputs)
         return outputs

     def on_train_epoch_end(self, state: State) -> None:
        # step the learning rate scheduler
        self.lr_scheduler.step()

 my_unit = MyUnit(module=..., optimizer=..., lr_scheduler=...)
