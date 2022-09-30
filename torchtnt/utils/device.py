#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import fields, is_dataclass
from typing import Any, Mapping, TypeVar

import torch
from torchtnt.utils.version import is_torch_version_geq_1_12
from typing_extensions import Protocol, runtime_checkable, TypedDict


def get_device_from_env() -> torch.device:
    """Function that gets the torch.device based on the current environment.

    This currently supports only CPU and GPU devices. If CUDA is available, this function also sets the CUDA device.

    Within a distributed context, this function relies on the ``LOCAL_RANK`` environment variable
    to be made available by the program launcher for setting the appropriate device index.

    Raises:
        RuntimeError
            If ``LOCAL_RANK`` is outside the range of available GPU devices.
    """
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                "The local rank is larger than the number of available GPUs."
            )
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif (
        is_torch_version_geq_1_12()
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


T = TypeVar("T")
TSelf = TypeVar("TSelf")


@runtime_checkable
class _CopyableData(Protocol):
    def to(self: TSelf, device: torch.device, *args: Any, **kwargs: Any) -> TSelf:
        """Copy data to the specified device"""
        ...


def _is_named_tuple(x: T) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields")


def copy_data_to_device(data: T, device: torch.device, *args: Any, **kwargs: Any) -> T:
    """Function that recursively copies data to a torch.device.

    Args:
        data: The data to copy to device
        device: The device to which the data should be copied
        args: positional arguments that will be passed to the `to` call
        kwargs: keyword arguments that will be passed to the `to` call

    Returns:
        The data on the correct device
    """

    # Redundant isinstance(data, tuple) check is required here to make pyre happy
    if _is_named_tuple(data) and isinstance(data, tuple):
        return type(data)(
            **copy_data_to_device(data._asdict(), device, *args, **kwargs)
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(copy_data_to_device(e, device, *args, **kwargs) for e in data)
    elif isinstance(data, defaultdict):
        return type(data)(
            data.default_factory,
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            },
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            }
        )
    elif is_dataclass(data) and not isinstance(data, type):
        new_data_class = type(data)(
            **{
                field.name: copy_data_to_device(
                    getattr(data, field.name), device, *args, **kwargs
                )
                for field in fields(data)
                if field.init
            }
        )
        for field in fields(data):
            if not field.init:
                setattr(
                    new_data_class,
                    field.name,
                    copy_data_to_device(
                        getattr(data, field.name), device, *args, **kwargs
                    ),
                )
        return new_data_class
    elif isinstance(data, _CopyableData):
        return data.to(device, *args, **kwargs)
    return data


class GPUStats(TypedDict):
    utilization_gpu_percent: float
    utilization_memory_percent: float
    fan_speed_percent: float
    memory_used_mb: int
    memory_free_mb: int
    temperature_gpu_celsius: float
    temperature_memory_celsius: float


def get_nvidia_smi_gpu_stats(device: torch.device) -> GPUStats:  # pragma: no-cover
    """Get GPU stats from nvidia smi.

    Args:
         device: A GPU torch.device to get stats from.

    Returns:
        dict (str, float): a dict that maps gpu stats to their values.

        Keys:
            - 'utilization_gpu_percent'
            - 'utilization_memory_percent'
            - 'fan_speed_percent'
            - 'memory_used_mb'
            - 'memory_free_mb'
            - 'temperature_gpu_celsius'
            - 'temperature_memory_celsius'

    Raises:
        FileNotFoundError:
            If nvidia-smi command is not found.
    """
    # Check for nvidia-smi
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        raise FileNotFoundError("nvidia-smi: command not found.")

    # Prepare keys
    gpu_stat_keys = [
        "utilization_gpu_percent",
        "utilization_memory_percent",
        "fan_speed_percent",
        "memory_used_mb",
        "memory_free_mb",
        "temperature_gpu_celsius",
        "temperature_memory_celsius",
    ]

    # Format as "utilization.gpu,utilization.memory,fan.speed,etc"
    smi_query = ",".join([".".join(key.split("_")[:-1]) for key in gpu_stat_keys])

    gpu_id = torch._utils._get_device_index(device)

    # Get values from nvidia-smi
    result = subprocess.run(
        [
            nvidia_smi_path,
            f"--query-gpu={smi_query}",
            "--format=csv,nounits,noheader",
            f"--id={gpu_id}",
        ],
        encoding="utf-8",
        capture_output=True,
        check=True,
    )

    # Format output
    output = result.stdout.strip()
    stats = []
    for value in output.split(", "):
        try:
            float_val = float(value)
        except ValueError:
            float_val = 0.0
        stats.append(float_val)

    # Add units to keys and populate values
    # This is not a dict comprehension to prevent pyre warnings.
    gpu_stats: GPUStats = {
        "utilization_gpu_percent": stats[0],
        "utilization_memory_percent": stats[1],
        "fan_speed_percent": stats[2],
        "memory_used_mb": stats[3],
        "memory_free_mb": stats[4],
        "temperature_gpu_celsius": stats[5],
        "temperature_memory_celsius": stats[6],
    }
    return gpu_stats


class CPUStats(TypedDict):
    cpu_vm_percent: float
    cpu_percent: float
    cpu_swap_percent: float


def get_psutil_cpu_stats() -> CPUStats:
    """Get CPU process stats using psutil.

    Returns:
        Dict[str, float]: a dict that maps cpu stats to their values.

        Keys:

            - 'cpu_vm_percent'
            - 'cpu_percent'
            - 'cpu_swap_percent'
    """
    try:
        import psutil
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "`get_cpu_process_metrics` requires `psutil` to be installed."
            " Install it by running `pip install -U psutil`."
        )

    stats: CPUStats = {
        "cpu_vm_percent": psutil.virtual_memory().percent,
        "cpu_percent": psutil.cpu_percent(),
        "cpu_swap_percent": psutil.swap_memory().percent,
    }
    return stats
