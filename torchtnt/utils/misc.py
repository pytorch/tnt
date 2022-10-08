#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

_SEC_IN_DAY: int = 60 * 60 * 24


def days_to_secs(days: Optional[int]) -> Optional[int]:
    """Convert time from days to seconds"""
    if days is None:
        return None
    if days < 0:
        raise ValueError(f"days must be non-negative, but was given {days}")
    return days * _SEC_IN_DAY
