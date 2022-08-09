#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import typing as T
from datetime import timedelta

import torch
from torch.distributed.constants import default_pg_timeout
from torchtnt.utils import get_device_from_env, get_process_group_backend_from_device

_log: logging.Logger = logging.getLogger(__name__)


def _check_dist_env() -> bool:
    """
    Check if all environment variables required to initialize torch.distributed are set
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    """
    env_required = (
        os.environ.get("MASTER_PORT"),
        os.environ.get("MASTER_ADDR"),
        os.environ.get("WORLD_SIZE"),
        os.environ.get("RANK"),
    )
    return all(env is not None for env in env_required)


def init_from_env(
    device_type: T.Optional[str] = None,
    pg_backend: T.Optional[str] = None,
    pg_timeout: timedelta = default_pg_timeout,
) -> torch.device:
    """Utility function that initializes the device and process group, if applicable.

    The global process group is initialized only if

        - torch distributed is not already initialized
        - the program has been launched on multiple processes

    This is intended as a convenience to include at the beginning of scripts that follow
    a SPMD-style execution model.


    Args:
        device_type (str, optional): Device type to initialize. If None, device will be initialized
                                  based on environment
        pg_backend (str, optional): The process group backend to use. If None, it will use the
                                    defaut process group backend from the device
        pg_timeout (timedelta, optional): Timeout for operations executed against the process
                                          group. Default value equals 30 minutes
    """
    device = torch.device("cpu") if device_type == "cpu" else get_device_from_env()

    if device_type is not None and device.type != device_type:
        raise RuntimeError(
            f"Device type is specified to {device_type} but got {device.type} from env"
        )

    if _check_dist_env():
        if torch.distributed.is_initialized():
            _log.warning(
                "torch.distributed is already initialized. Skipping initializing the process group."
            )
            return device
        pg_backend = (
            pg_backend
            if pg_backend is not None
            else get_process_group_backend_from_device(device)
        )
        torch.distributed.init_process_group(backend=pg_backend, timeout=pg_timeout)
    return device
