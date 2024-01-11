#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import os

import sys
import unittest
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
from io import StringIO
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    TextIO,
    Tuple,
    TypeVar,
)

import torch

import torch.distributed.launcher as pet
from pyre_extensions import ParameterSpecification
from torch import distributed as dist, multiprocessing

from torch.distributed.elastic.utils.distributed import get_free_port
from torchtnt.utils.distributed import destroy_process_group


TParams = ParameterSpecification("TParams")
TReturn = TypeVar("TReturn")


@dataclass
class ProcessGroupSetupParams:
    backend: str
    port: str
    world_size: int


def get_pet_launch_config(nproc: int) -> pet.LaunchConfig:
    """
    Initialize pet.LaunchConfig for single-node, multi-rank functions.

    Args:
        nproc: The number of processes to launch.

    Returns:
        An instance of pet.LaunchConfig for single-node, multi-rank functions.

    Example:
        >>> from torch.distributed import launcher
        >>> launch_config = get_pet_launch_config(nproc=8)
        >>> launcher.elastic_launch(config=launch_config, entrypoint=train)()
    """
    return pet.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=nproc,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        monitor_interval=1,
    )


def is_asan() -> bool:
    """Determines if the Python interpreter is running with ASAN"""
    return hasattr(ctypes.CDLL(""), "__asan_init")


def is_tsan() -> bool:
    """Determines if the Python interpreter is running with TSAN"""
    return hasattr(ctypes.CDLL(""), "__tsan_init")


def is_asan_or_tsan() -> bool:
    return is_asan() or is_tsan()


def skip_if_asan(
    func: Callable[TParams, TReturn]
) -> Callable[TParams, Optional[TReturn]]:
    """Skip test run if we are in ASAN mode."""

    @wraps(func)
    def wrapper(*args: TParams.args, **kwargs: TParams.kwargs) -> Optional[TReturn]:
        if is_asan_or_tsan():
            print("Skipping test run since we are in ASAN mode.")
            return
        return func(*args, **kwargs)

    return wrapper


def spawn_multi_process(
    world_size: int,
    backend: str,
    test_method: Callable[TParams, TReturn],
    *test_method_args: Any,
    **test_method_kwargs: Any,
) -> List[TReturn]:
    """
    Spawn single node, multi-rank function.
    Uses localhost and free port to communicate.

    Args:
        world_size: number of processes
        backend: backend to use. for example, "nccl", "gloo", etc
        test_method: callable to spawn. first 3 arguments are rank, world_size and mp output dict
        test_method_args: args for the test method
        test_method_kwargs: kwargs for the test method

    Returns:
        A list, l, where l[i] is the return value of test_method on rank i
    """
    manager = multiprocessing.Manager()
    mp_output_dict = manager.dict()

    port = str(get_free_port())
    torch.multiprocessing.spawn(
        # torch.multiprocessing.spawn sends rank as the first param
        # https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn
        _init_pg_and_rank_and_launch_test,
        args=(
            ProcessGroupSetupParams(backend=backend, port=port, world_size=world_size),
            mp_output_dict,
            test_method,
            test_method_args,
            test_method_kwargs,
        ),
        nprocs=world_size,
    )

    output_list = []
    for i in range(world_size):
        output_list.append(mp_output_dict[i])

    return output_list


def _init_pg_and_rank_and_launch_test(
    rank: int,
    pg_setup_params: ProcessGroupSetupParams,
    mp_output_dict: Dict[int, object],
    test_method: Callable[TParams, TReturn],
    args: List[object],
    kwargs: Dict[str, object],
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = pg_setup_params.port
    os.environ["WORLD_SIZE"] = str(pg_setup_params.world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        rank=rank,
        world_size=pg_setup_params.world_size,
        backend=pg_setup_params.backend,
        timeout=timedelta(seconds=10),  # setting up timeout for distributed collectives
    )
    try:
        mp_output_dict[rank] = test_method(*args, **kwargs)  # pyre-fixme

    finally:
        destroy_process_group()


@contextmanager
def captured_output() -> Generator[Tuple[TextIO, TextIO], None, None]:
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


"""Decorator for tests to ensure running on a GPU."""
skip_if_not_gpu: Callable[..., Callable[..., object]] = unittest.skipUnless(
    torch.cuda.is_available(), "Skipping test since GPU is not available"
)

"""Decorator for tests to ensure running when distributed is available."""
skip_if_not_distributed: Callable[..., Callable[..., object]] = unittest.skipUnless(
    torch.distributed.is_available(), "Skipping test since distributed is not available"
)
