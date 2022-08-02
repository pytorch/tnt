#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from enum import Enum
from typing import Any, Dict, Mapping, Sequence

import torch


def _is_named_tuple(
    # pyre-ignore: Missing parameter annotation [2]: Parameter `x` must have a type other than `Any`.
    x: Any,
) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields")


def get_tensor_size_bytes_map(
    # pyre-ignore: Missing parameter annotation [2]: Parameter `obj` must have a type other than `Any`.
    obj: Any,
) -> Dict[torch.Tensor, int]:
    tensor_map = {}
    attributes_q = deque()
    attributes_q.append(obj)
    while attributes_q:
        attribute = attributes_q.popleft()
        if isinstance(attribute, torch.Tensor):
            tensor_map[attribute] = attribute.size().numel() * attribute.element_size()
        elif _is_named_tuple(attribute):
            attributes_q.extend(attribute._asdict().values())
        elif isinstance(attribute, Mapping):
            attributes_q.extend(attribute.values())
        elif isinstance(attribute, Sequence) and not isinstance(attribute, str):
            attributes_q.extend(attribute)
        elif hasattr(attribute, "__dict__") and not isinstance(attribute, Enum):
            attributes_q.extend(attribute.__dict__.values())
    return tensor_map
