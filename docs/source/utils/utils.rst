Utils
=============

Training related utilities. These are independent of the framework and can be used as needed.



Data Utils
~~~~~~~~~~~~~~~~~~~~~


.. currentmodule:: torchtnt.utils.data
.. autosummary::
   :toctree: generated
   :nosignatures:

   AbstractRandomDataset
   AllDatasetBatchesIterator
   CudaDataPrefetcher
   InOrderIterator
   MultiDataLoader
   MultiIterator
   RandomizedBatchSamplerIterator
   RoundRobinIterator



Device Utils
~~~~~~~~~~~~~~~~~~~~~


.. currentmodule:: torchtnt.utils.device
.. autosummary::
   :toctree: generated
   :nosignatures:

   get_device_from_env
   copy_data_to_device
   record_data_in_stream
   get_nvidia_smi_gpu_stats
   get_psutil_cpu_stats
   set_float32_precision


Distributed Utils
~~~~~~~~~~~~~~~~~~~~~


.. currentmodule:: torchtnt.utils.distributed
.. autosummary::
   :toctree: generated
   :nosignatures:

   PGWrapper
   get_global_rank
   get_local_rank
   get_world_size
   barrier
   destroy_process_group
   get_process_group_backend_from_device
   get_file_init_method
   get_tcp_init_method
   all_gather_tensors
   rank_zero_fn
   revert_sync_batchnorm
   spawn_multi_process
   sync_bool


Early Stop Checker
~~~~~~~~~~~~~~~~~~~~~


.. currentmodule:: torchtnt.utils.early_stop_checker
.. autosummary::
   :toctree: generated
   :nosignatures:

   EarlyStopChecker


Environment Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.env
.. autosummary::
   :toctree: generated
   :nosignatures:

   init_from_env
   seed


Flops Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.flops
.. autosummary::
   :toctree: generated
   :nosignatures:

   FlopTensorDispatchMode


Filesystem Spec Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.fsspec
.. autosummary::
   :toctree: generated
   :nosignatures:

   get_filesystem


Logger Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.loggers
.. autosummary::
   :toctree: generated
   :nosignatures:

   FileLogger
   MetricLogger
   CSVLogger
   InMemoryLogger
   StdoutLogger
   JSONLogger
   TensorBoardLogger


Memory Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.memory
.. autosummary::
   :toctree: generated
   :nosignatures:

   RSSProfiler
   measure_rss_deltas


MemorySnapshotProfiler Utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.memory_snapshot_profiler
.. autosummary::
   :toctree: generated
   :nosignatures:

   MemorySnapshotProfiler


Module Summary Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.module_summary
.. autosummary::
   :toctree: generated
   :nosignatures:

   ModuleSummary
   get_module_summary
   get_summary_table
   prune_module_summary


OOM Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.oom
.. autosummary::
   :toctree: generated
   :nosignatures:

   is_out_of_cpu_memory
   is_out_of_cuda_memory
   is_out_of_memory_error
   log_memory_snapshot
   attach_oom_observer


Optimizer Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.optimizer
.. autosummary::
   :toctree: generated
   :nosignatures:

   init_optim_state


Precision Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.precision
.. autosummary::
   :toctree: generated
   :nosignatures:

   convert_precision_str_to_dtype


Prepare Module Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.prepare_module
.. autosummary::
   :toctree: generated
   :nosignatures:

   prepare_module
   prepare_ddp
   prepare_fsdp
   convert_str_to_strategy
   DDPStrategy
   FSDPStrategy


Progress Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.progress
.. autosummary::
   :toctree: generated
   :nosignatures:

   Progress
   estimated_steps_in_epoch
   estimated_steps_in_loop
   estimated_steps_in_fit


Rank Zero Log Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.rank_zero_log
.. autosummary::
   :toctree: generated
   :nosignatures:

   rank_zero_print
   rank_zero_debug
   rank_zero_info
   rank_zero_warn
   rank_zero_error
   rank_zero_critical


Stateful
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.stateful
.. autosummary::
   :toctree: generated
   :nosignatures:

   Stateful
   MultiStateful


SWA
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.swa
.. autosummary::
   :toctree: generated
   :nosignatures:

   AveragedModel


Test Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.test_utils
.. autosummary::
   :toctree: generated
   :nosignatures:

   get_pet_launch_config
   is_asan
   is_tsan
   skip_if_asan


Timer Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.timer
.. autosummary::
   :toctree: generated
   :nosignatures:

   log_elapsed_time
   TimerProtocol
   Timer
   FullSyncPeriodicTimer
   BoundedTimer
   get_timer_summary
   get_durations_histogram
   get_synced_durations_histogram
   get_synced_timer_histogram
   get_recorded_durations_table


TQDM Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.tqdm
.. autosummary::
   :toctree: generated
   :nosignatures:

   create_progress_bar
   update_progress_bar
   close_progress_bar


Version Utils
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchtnt.utils.version
.. autosummary::
   :toctree: generated
   :nosignatures:

   is_windows
   get_python_version
   get_torch_version


Misc Utils
~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: torchtnt.utils.misc
.. autosummary::
   :toctree: generated
   :nosignatures:

   days_to_secs
   transfer_batch_norm_stats
