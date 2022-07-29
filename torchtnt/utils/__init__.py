# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from .data_prefetcher import CudaDataPrefetcher
from .device import (
    copy_data_to_device,
    CPUStats,
    get_device_from_env,
    get_nvidia_smi_gpu_stats,
    get_psutil_cpu_stats,
    GPUStats,
)
from .version import (
    get_python_version,
    get_torch_version,
    is_torch_version_geq_1_10,
    is_torch_version_geq_1_11,
    is_torch_version_geq_1_12,
    is_torch_version_geq_1_8,
    is_torch_version_geq_1_9,
    is_windows,
)


__all__ = [
    "CudaDataPrefetcher",
    "copy_data_to_device",
    "CPUStats",
    "get_device_from_env",
    "get_nvidia_smi_gpu_stats",
    "get_psutil_cpu_stats",
    "GPUStats",
    "get_python_version",
    "get_torch_version",
    "is_torch_version_geq_1_10",
    "is_torch_version_geq_1_11",
    "is_torch_version_geq_1_12",
    "is_torch_version_geq_1_8",
    "is_torch_version_geq_1_9",
    "is_windows",
]
