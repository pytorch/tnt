Callbacks
=======================

In TorchTNT a :class:`~torchtnt.runner.callback.Callback` can be used to supplement your loop with additional functionality. A Callback is simply
a class with various hooks that are called during loop execution. To benefit from a Callback, you must extend the Callback interface and implement the necessary hooks
to add your additional functionality. For a list of hooks which can be implemented see the :class:`~torchtnt.runner.callback.Callback` API reference.

Below is an example of a ``PyTorchProfiler`` callback which can be included by the user to enable profiling automatically.

.. code-block:: python

 from torchtnt.runner.callback import Callback

 class PyTorchProfiler(Callback):
    """
    A callback which profiles user code using PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html
    """

    def __init__(
        self,
        profiler: torch.profiler.profile,
    ) -> None:
        self.profiler: torch.profiler.profile = profiler

    def on_train_start(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        self.profiler.start()

    def on_train_step_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        self.profiler.step()

    def on_train_end(self, state: State, unit: TrainUnit[TTrainData]) -> None:
        self.profiler.stop()

    def on_eval_start(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        # if in fit do nothing since the profiler was already started in on_train_start
        if state.entry_point == EntryPoint.EVALUATE:
            self.profiler.start()

    def on_eval_step_end(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        self.profiler.step()

    def on_eval_end(self, state: State, unit: EvalUnit[TEvalData]) -> None:
        # if in fit do nothing since the profiler will be stopped in on_train_end
        if state.entry_point == EntryPoint.EVALUATE:
            self.profiler.stop()

    def on_predict_start(self, state: State, unit: PredictUnit[TPredictData]) -> None:
        self.profiler.start()

    def on_predict_step_end(
        self, state: State, unit: PredictUnit[TPredictData]
    ) -> None:
        self.profiler.step()

    def on_predict_end(self, state: State, unit: PredictUnit[TPredictData]) -> None:
        self.profiler.stop()

To include this in your loop, instantiate the class and pass it in the `callbacks` parameter to the :py:func:`~torchtnt.runner.train`, :py:func:`~torchtnt.runner.evaluate`,
:py:func:`~torchtnt.runner.predict`, or :py:func:`~torchtnt.runner.fit` entry point.

.. code-block:: python

 profiler = PyTorchProfiler(
        profiler=torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=path),
            with_stack=True,
        )
    )
 train(train_unit, train_dataloader, callbacks=[profiler])
