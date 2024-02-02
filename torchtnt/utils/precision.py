#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Mapping, Optional

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torchtnt.utils.prepare_module import _is_fsdp_module

_DTYPE_STRING_TO_DTYPE_MAPPING: Mapping[str, Optional[torch.dtype]] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": None,
}


def convert_precision_str_to_dtype(precision: str) -> Optional[torch.dtype]:
    """
    Converts precision as a string to a torch.dtype

    Args:
        precision: string containing the precision

    Raises:
        ValueError if an invalid precision string is passed.

    """
    if precision not in _DTYPE_STRING_TO_DTYPE_MAPPING:
        raise ValueError(
            f"Precision {precision} not supported. Please use one of {list(_DTYPE_STRING_TO_DTYPE_MAPPING.keys())}"
        )
    return _DTYPE_STRING_TO_DTYPE_MAPPING[precision]


def get_grad_scaler_from_precision(
    precision: torch.dtype, module: torch.nn.Module
) -> Optional[torch.amp.GradScaler]:
    """
    Returns the correct grad scaler to use based on the precision and whether
    or not the model is FSDP.

    Args:
        precision: the precision being used
        module: the module being trained

    Returns:
        The appropriate grad scaler to use, ``None`` if no grad scaler should be used.
    """

    if precision == torch.float16:
        if _is_fsdp_module(module):
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            return ShardedGradScaler()
        else:
            return GradScaler()
    return None
