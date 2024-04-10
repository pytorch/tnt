Overview
================================

Welcome! TNT is a lightweight library for PyTorch training tools and utilities. It has two main components, which are the top-level modules of the repo:

1. **torchtnt.framework**: contains a lightweight training framework to simplify maintaining training, evaluation, and prediction loops.
2. :doc:`torchtnt.utils </utils/utils>`: contains a grab-bag of various independent, training-related utilities, including data related abstractions and wrappers around different publishers to simplify logging metrics.

.. figure:: assets/TNTDiagram.png
    :align: center


Training Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Core Functional APIs
--------------------------------------

These are the core apis used in TorchTNT to train & evaluate models with:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - API Call
     - Description
   * - :py:func:`~torchtnt.framework.train.train`
     - The train entry point is intended to train models
   * - :py:func:`~torchtnt.framework.evaluate.evaluate`
     - The evaluate entry point is intended for use immediately after training the models
   * - :py:func:`~torchtnt.framework.predict.predict`
     - The predict entry point is intended to do model inference
   * - :py:func:`~torchtnt.framework.fit.fit`
     - The fit entry point is intended to interleave training and evaluation of models at specified intervals


Organizing your model code
--------------------------------------
The Unit concept represents the primary place to organize your model code in TorchTNT. There are three types of units: :class:`~torchtnt.framework.unit.TrainUnit`, :class:`~torchtnt.framework.unit.EvalUnit`, and :class:`~torchtnt.framework.unit.PredictUnit`. These interfaces are mutually exclusive and can be combined as needed, e.g. in the case of fitting (interleaving training and evaluation).

.. code-block:: python

    class MyExampleUnit(TrainUnit, PredictUnit):
        """
        Basic implemention of a unit, subclassing the train and predict interface

        Args:
            module: nn.Module to train
            device: device to move the module and data to
            optimizer: optimizer to use on the module
            log_every_n_steps: frequency to log stats
        """

        def __init__(
            self,
            module: torch.nn.Module,
            device: torch.device,
            optimizer: Optional[torch.optim.Optimizer] = None,
            log_every_n_steps: Optional[int] = None,
        ) -> None:
            super().__init__()
            self._module = module.to(device)
            self._device = device
            self._optimizer = optimizer
            self._log_every_n_steps = log_every_n_steps

            self._accuracy = BinaryAccuracy() # use any metrics library here
            self._tb_logger = TensorBoardLogger() # use preferred logger here

        # train_step is a method which is invoked by TorchTNT trainer
        # here we implement the training part of our task
        def train_step(self, state: State, data: Batch) -> None:
            data = copy_data_to_device(data, device=self.device)
            inputs, targets = data

            outputs = self.module(inputs)
            outputs = torch.squeeze(outputs)

            # update metrics
            self.accuracy.update(outputs, targets)

            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
            loss.backward()

            # update optimizer
            self.optimizer.step()
            self.optimizer.zero_grad()

            # log to tensorboard in the specified interval
            step_count = self.train_progress.num_steps_completed
            if step_count % self.log_every_n_steps == 0:
                acc = self.accuracy.compute()
                self._tb_logger.log_dict(
                    {"train_loss": loss, "train_accuracy": acc}, step_count
                )

        def on_train_epoch_end(self, state: State) -> None:
            # compute and log the metric at the end of the epoch
            step_count = self.train_progress.num_steps_completed
            acc = self.accuracy.compute()
            self._tb_logger.log("train_accuracy_epoch", acc, step_count)

            # reset the metric at the end of every epoch
            self.accuracy.reset()

        def predict_step(
            self, state: State, data: PredictBatch
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            inputs = copy_data_to_device(data, device=self.device)
            outputs = self.module(inputs)
            outputs = torch.squeeze(outputs)
            return (data, outputs)

    my_unit = MyExampleUnit(
        module=torch.nn.Linear(256, 10),
        device=torch.device("cuda"),
        log_every_n_steps=1000
        ...
    )

    # instantiate train dataloader
    train_dataloader = ...

    # use train api to train the model
    train(my_unit, train_dataloader, max_epochs=5)

Here, the loss computation, backwards pass, etc must all be invoked manually. However, for users who want automatic optimization to be handled for them, and who don’t necessarily need to have control over their backward pass/optimizer step themselves, we offer an extension called the :class:`~torchtnt.framework.auto_unit.AutoUnit`.

The :class:`~torchtnt.framework.auto_unit.AutoUnit` implements the TrainUnit, EvalUnit, and PredictUnit interfaces. The user must define their ``compute_loss`` function and ``configure_optimizers_and_lr_schedulers``. The AutoUnit handles

- moving models and data to device appropriately
- applying distributed training (DDP, FSDP)
- mixed precision
- gradient accumulation
- anomaly detection
- gradient clipping
- torch.compile
- and more!

.. code-block:: python

    class MyUnit(AutoUnit):
        def __init__(
            self,
            module: torch.nn.Module,
            device: torch.device,
            strategy: Optional[str],
            precision: Optional[str],
            gradient_accumulation_steps: int,
            *,
            tb_logger: TensorBoardLogger,
            train_accuracy: MulticlassAccuracy,
            log_every_n_steps: int,
            lr: float,
            gamma: float,
        ) -> None:
            super().__init__(
                module=module,
                device=device,
                strategy=strategy,
                precision=precision,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            self.tb_logger = tb_logger
            self.lr = lr
            self.gamma = gamma

            # create an accuracy metric to compute the accuracy of training
            self.train_accuracy = train_accuracy
            self.log_every_n_steps = log_every_n_steps

        def configure_optimizers_and_lr_scheduler(
            self, module: torch.nn.Module
        ) -> Tuple[torch.optim.Optimizer, Optional[TLRScheduler]]:
            optimizer = Adadelta(module.parameters(), lr=self.lr)
            lr_scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
            return optimizer, lr_scheduler

        def compute_loss(
            self, state: State, data: Batch
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            inputs, targets = data
            outputs = self.module(inputs)
            outputs = torch.squeeze(outputs)
            loss = torch.nn.functional.nll_loss(outputs, targets)

            return loss, outputs

        def on_train_step_end(
            self,
            state: State,
            data: Batch,
            step: int,
            results: TrainStepResults,
        ) -> None:
            loss, outputs = results.loss, results.outputs
            _, targets = data
            self.train_accuracy.update(outputs, targets)
            if step % self.log_every_n_steps == 0:
                accuracy = self.train_accuracy.compute()
                self.tb_logger.log("accuracy", accuracy, step)
                self.tb_logger.log("loss", loss, step)

        def on_train_epoch_end(self, state: State) -> None:
            super().on_train_epoch_end(state)
            # reset the metric every epoch
            self.train_accuracy.reset()




Callbacks
--------------------------------------
Callbacks are the mechanism to inject additional functionality within the train/eval/predict loops at specified hooks. Callbacks are the recommended way to checkpoint, do progress monitoring with, etc. TorchTNT has several built-in callbacks. See the :class:`~torchtnt.framework.callback.Callback` page for more details.

.. code-block:: python

    from torchtnt.framework.callback import Callback
    from torchtnt.framework.state import State
    from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit

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

Utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TorchTNT also offers a suite of training related utilities, ranging from distributed to debugging tools. These are framework independent and can be used as needed.
