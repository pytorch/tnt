#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def is_out_of_cpu_memory(exception: BaseException) -> bool:
    """Returns True if the exception is related to CPU OOM"""
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


def is_out_of_cuda_memory(exception: BaseException) -> bool:
    """Returns True if the exception is related to CUDA OOM"""
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and (
            "RuntimeError: cuda runtime error (2) : out of memory" in exception.args[0]
            or "CUDA out of memory." in exception.args[0]
        )
    )


def is_out_of_memory_error(exception: BaseException) -> bool:
    """Returns Ture if an exception is due to an OOM based on error message"""
    return is_out_of_cpu_memory(exception) or is_out_of_cuda_memory(exception)
