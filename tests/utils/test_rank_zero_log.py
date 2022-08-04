#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

from torchtnt.utils.rank_zero_log import (
    rank_zero_critical,
    rank_zero_debug,
    rank_zero_error,
    rank_zero_info,
    rank_zero_warn,
)


class RankZeroLogTest(unittest.TestCase):
    @patch.dict("os.environ", {"RANK": "0"}, clear=True)
    def test_rank_zero_fn_rank_zero(self):

        logger = MagicMock()

        rank_zero_debug("foo", logger=logger)
        logger.debug.assert_called_once_with("foo", stacklevel=2)

        rank_zero_info("foo", logger=logger)
        logger.info.assert_called_once_with("foo", stacklevel=2)

        rank_zero_warn("foo", logger=logger)
        logger.warn.assert_called_once_with("foo", stacklevel=2)

        rank_zero_error("foo", logger=logger)
        logger.error.assert_called_once_with("foo", stacklevel=2)

        rank_zero_critical("foo", logger=logger)
        logger.critical.assert_called_once_with("foo", stacklevel=2)

    @patch.dict("os.environ", {"RANK": "1"}, clear=True)
    def test_rank_zero_fn_rank_non_zero(self) -> None:

        logger = MagicMock()

        rank_zero_debug("foo", logger=logger)
        logger.debug.assert_not_called()

        rank_zero_info("foo", logger=logger)
        logger.info.assert_not_called()

        rank_zero_warn("foo", logger=logger)
        logger.warn.assert_not_called()

        rank_zero_error("foo", logger=logger)
        logger.error.assert_not_called()

        rank_zero_critical("foo", logger=logger)
        logger.critical.assert_not_called()
