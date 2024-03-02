#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import patch

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from torchtnt.utils.precision import (
    convert_precision_str_to_dtype,
    get_grad_scaler_from_precision,
)


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

    def test_get_grad_scaler_from_precision(self) -> None:
        grad_scaler = get_grad_scaler_from_precision(
            torch.float32, torch.nn.Linear(2, 2)
        )
        self.assertIsNone(grad_scaler)

        grad_scaler = get_grad_scaler_from_precision(
            torch.float16, torch.nn.Linear(2, 2)
        )
        self.assertTrue(isinstance(grad_scaler, GradScaler))

        with patch("torchtnt.utils.precision._is_fsdp_module", return_value=True):
            grad_scaler = get_grad_scaler_from_precision(
                torch.float16, torch.nn.Linear(2, 2)
            )
            self.assertTrue(isinstance(grad_scaler, ShardedGradScaler))
