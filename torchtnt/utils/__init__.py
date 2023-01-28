# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .device import (
    copy_data_to_device,
    CPUStats,
    get_device_from_env,
    get_nvidia_smi_gpu_stats,
    get_psutil_cpu_stats,
    GPUStats,
    maybe_enable_tf32,
)
from .distributed import (
    all_gather_tensors,
    get_global_rank,
    get_process_group_backend_from_device,
    get_world_size,
    PGWrapper,
    sync_bool,
)
from .early_stop_checker import EarlyStopChecker
from .env import init_from_env
from .fsspec import get_filesystem
from .lr_scheduler import TLRScheduler
from .memory import get_tensor_size_bytes_map, measure_rss_deltas, RSSProfiler
from .misc import days_to_secs, transfer_batch_norm_stats, transfer_weights
from .oom import is_out_of_cpu_memory, is_out_of_cuda_memory, is_out_of_memory_error
from .rank_zero_log import (
    rank_zero_critical,
    rank_zero_debug,
    rank_zero_error,
    rank_zero_info,
    rank_zero_print,
    rank_zero_warn,
)
from .seed import seed
from .timer import FullSyncPeriodicTimer, get_timer_summary, Timer
from .version import (
    get_python_version,
    get_torch_version,
    is_torch_version_ge_1_13_1,
    is_torch_version_geq_1_10,
    is_torch_version_geq_1_11,
    is_torch_version_geq_1_12,
    is_torch_version_geq_1_13,
    is_torch_version_geq_1_14,
    is_torch_version_geq_1_8,
    is_torch_version_geq_1_9,
    is_torch_version_geq_2_0,
    is_windows,
)

__all__ = [
    "copy_data_to_device",
    "CPUStats",
    "get_device_from_env",
    "get_nvidia_smi_gpu_stats",
    "get_psutil_cpu_stats",
    "GPUStats",
    "maybe_enable_tf32",
    "all_gather_tensors",
    "get_global_rank",
    "get_process_group_backend_from_device",
    "get_world_size",
    "PGWrapper",
    "sync_bool",
    "EarlyStopChecker",
    "init_from_env",
    "get_filesystem",
    "get_tensor_size_bytes_map",
    "measure_rss_deltas",
    "RSSProfiler",
    "days_to_secs",
    "is_out_of_cpu_memory",
    "is_out_of_cuda_memory",
    "is_out_of_memory_error",
    "rank_zero_critical",
    "rank_zero_debug",
    "rank_zero_error",
    "rank_zero_info",
    "rank_zero_print",
    "rank_zero_warn",
    "seed",
    "FullSyncPeriodicTimer",
    "get_timer_summary",
    "transfer_batch_norm_stats",
    "transfer_weights",
    "Timer",
    "TLRScheduler",
    "get_python_version",
    "get_torch_version",
    "is_torch_version_ge_1_13_1",
    "is_torch_version_geq_1_10",
    "is_torch_version_geq_1_11",
    "is_torch_version_geq_1_12",
    "is_torch_version_geq_1_13",
    "is_torch_version_geq_1_14",
    "is_torch_version_geq_1_8",
    "is_torch_version_geq_1_9",
    "is_torch_version_geq_2_0",
    "is_windows",
]
