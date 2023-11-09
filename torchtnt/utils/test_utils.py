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
from functools import wraps
from io import StringIO
from typing import Any, Callable, Dict, Generator, Optional, TextIO, Tuple, TypeVar

import torch.distributed.launcher as pet
from pyre_extensions import ParameterSpecification
from torch import distributed as dist, multiprocessing
from torch.distributed.elastic.utils.distributed import get_free_port


TParams = ParameterSpecification("TParams")
TReturn = TypeVar("TReturn")


def get_pet_launch_config(nproc: int) -> pet.LaunchConfig:
    """
    Initialize pet.LaunchConfig for single-node, multi-rank functions.

    Args:
        nproc: The number of processes to launch.

    Returns:
        An instance of pet.LaunchConfig for single-node, multi-rank functions.
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
    *args: Any,
) -> Dict[int, TReturn]:
    """
    Spawn single node, multi-rank function.
    Uses localhost and free port to communicate.

    Args:
        world_size: number of processes
        backend: backend to use. for example, "nccl", "gloo", etc
        test_method: callable to spawn. first 3 arguments are rank, world_size and mp output dict
        args: additional args for func

    Returns:
        A dictionary of rank -> func return value
    """
    os.environ["MASTER_PORT"] = str(get_free_port())
    os.environ["MASTER_ADDR"] = "127.0.0.1"

    processes = []
    manager = multiprocessing.Manager()
    mp_output_dict = manager.dict()
    tc = unittest.TestCase()
    ctx = multiprocessing.get_context("spawn")
    for rank in range(world_size):
        p = ctx.Process(
            target=init_pg_and_rank_and_launch_test,
            args=(
                test_method,
                rank,
                world_size,
                backend,
                mp_output_dict,
                *args,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        tc.assertEqual(p.exitcode, 0)

    return mp_output_dict


def init_pg_and_rank_and_launch_test(
    test_method: Callable[TParams, TReturn],
    rank: int,
    world_size: int,
    backend: str,
    # pyre-fixme[2]
    mp_output_dict: Dict[int, Any],
    *args: Any,
) -> None:
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
    os.environ["LOCAL_RANK"] = str(rank)
    mp_output_dict[rank] = test_method(*args)  # pyre-fixme[29]


@contextmanager
def captured_output() -> Generator[Tuple[TextIO, TextIO], None, None]:
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err
