#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Optional

from torchtnt.utils.distributed import get_global_rank

_LOGGER: logging.Logger = logging.getLogger(__name__)


def rank_zero_print(*args: Any, **kwargs: Any) -> None:
    if get_global_rank() != 0:
        return
    print(*args, **kwargs)


def rank_zero_debug(
    *args: Any, logger: Optional[logging.Logger] = None, **kwargs: Any
) -> None:
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.debug(*args, stacklevel=2, **kwargs)


def rank_zero_info(
    *args: Any, logger: Optional[logging.Logger] = None, **kwargs: Any
) -> None:
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.info(*args, stacklevel=2, **kwargs)


def rank_zero_warn(
    *args: Any, logger: Optional[logging.Logger] = None, **kwargs: Any
) -> None:
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.warn(*args, stacklevel=2, **kwargs)


def rank_zero_error(
    *args: Any, logger: Optional[logging.Logger] = None, **kwargs: Any
) -> None:
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.error(*args, stacklevel=2, **kwargs)


def rank_zero_critical(
    *args: Any, logger: Optional[logging.Logger] = None, **kwargs: Any
) -> None:
    if get_global_rank() != 0:
        return
    logger = logger or _LOGGER
    logger.critical(*args, stacklevel=2, **kwargs)
