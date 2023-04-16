# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields, is_dataclass
from typing import Mapping, TypeVar

import torch
from typing_extensions import Protocol, runtime_checkable

TData = TypeVar("TData")


def _is_named_tuple(x: TData) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields")


def record_data_in_stream(data: TData, stream: torch.cuda.streams.Stream) -> None:
    """
    As mentioned in
    https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html, PyTorch
    uses the "caching allocator" for memory allocation for tensors. When a tensor is
    freed, its memory is likely to be reused by newly constructed tensors. By default,
    this allocator traces whether a tensor is still in use by only the CUDA stream where
    it was created. When a tensor is used by additional CUDA streams, we need to call
    `record_stream` to tell the allocator about these streams. Otherwise, the allocator
    might free the underlying memory of the tensor once it is no longer used by the
    creator stream. This is a notable programming trick when we write programs using
    multiple CUDA streams.

    Args:
        data: The data on which to call record_stream
        stream: The CUDA stream with which to call record_stream
    """

    # Redundant isinstance(data, tuple) check is required here to make pyre happy
    if _is_named_tuple(data) and isinstance(data, tuple):
        record_data_in_stream(data._asdict(), stream)
    elif isinstance(data, (list, tuple)):
        for e in data:
            record_data_in_stream(e, stream)
    elif isinstance(data, Mapping):
        for _, v in data.items():
            record_data_in_stream(v, stream)
    elif is_dataclass(data) and not isinstance(data, type):
        for field in fields(data):
            record_data_in_stream(getattr(data, field.name), stream)
    elif isinstance(data, _MultistreamableData):
        data.record_stream(stream)


@runtime_checkable
class _MultistreamableData(Protocol):
    """
    Objects implementing this interface are allowed to be transferred
    from one CUDA stream to another.
    torch.Tensor implements this interface.
    """

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        """
        See https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html
        """
        ...
