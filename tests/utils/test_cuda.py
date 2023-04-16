#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
from torchtnt.utils.cuda import record_data_in_stream


class CudaTest(unittest.TestCase):

    cuda_available = torch.cuda.is_available()

    @unittest.skipUnless(
        condition=(cuda_available), reason="This test must run on a GPU host."
    )
    def test_record_data_in_stream_dict(self) -> None:
        curr_stream = torch.cuda.current_stream()
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        data = {"a": a, "b": b}

        with patch.object(a, "record_stream") as mock_record_stream_a, patch.object(
            b, "record_stream"
        ) as mock_record_stream_b:
            record_data_in_stream(data, curr_stream)
            mock_record_stream_a.assert_called_once()
            mock_record_stream_b.assert_called_once()

    @unittest.skipUnless(
        condition=(cuda_available), reason="This test must run on a GPU host."
    )
    def test_record_data_in_stream_tuple(self) -> None:
        curr_stream = torch.cuda.current_stream()
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        data = (a, b)

        with patch.object(a, "record_stream") as mock_record_stream_a, patch.object(
            b, "record_stream"
        ) as mock_record_stream_b:
            record_data_in_stream(data, curr_stream)
            mock_record_stream_a.assert_called_once()
            mock_record_stream_b.assert_called_once()

    @unittest.skipUnless(
        condition=(cuda_available), reason="This test must run on a GPU host."
    )
    def test_record_data_in_stream_list(self) -> None:
        curr_stream = torch.cuda.current_stream()
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        data = [a, b]

        with patch.object(a, "record_stream") as mock_record_stream_a, patch.object(
            b, "record_stream"
        ) as mock_record_stream_b:
            record_data_in_stream(data, curr_stream)
            mock_record_stream_a.assert_called_once()
            mock_record_stream_b.assert_called_once()
