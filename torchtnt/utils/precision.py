#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Mapping, Optional

import torch

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
