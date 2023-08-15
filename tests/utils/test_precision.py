#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtnt.utils.precision import convert_precision_str_to_dtype


class PrecisionTest(unittest.TestCase):
    def test_convert_precision_str_to_dtype_success(self) -> None:
        for (precision_str, expected_dtype) in [
            ("fp16", torch.float16),
            ("bf16", torch.bfloat16),
            ("fp32", None),
        ]:
            with self.subTest(
                precision_str=precision_str, expected_dtype=expected_dtype
            ):
                self.assertEqual(
                    convert_precision_str_to_dtype(precision_str), expected_dtype
                )

    def test_convert_precision_str_to_dtype_throws(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Precision foo not supported. Please use one of .*",
        ):
            convert_precision_str_to_dtype("foo")
