#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import os
import unittest
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Any, Dict
from unittest import mock
from unittest.mock import patch

import torch
from torchtnt.utils.device import (
    copy_data_to_device,
    get_device_from_env,
    get_nvidia_smi_gpu_stats,
    get_psutil_cpu_stats,
    maybe_enable_tf32,
    record_data_in_stream,
)


class DeviceTest(unittest.TestCase):

    cuda_available: bool = torch.cuda.is_available()

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_cpu_device(self, _) -> None:
        device = get_device_from_env()
        self.assertEqual(device.type, "cpu")
        self.assertEqual(device.index, None)

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_get_gpu_device(self) -> None:
        device_idx = torch.cuda.device_count() - 1
        self.assertGreaterEqual(device_idx, 0)
        with mock.patch.dict(os.environ, {"LOCAL_RANK": str(device_idx)}, clear=True):
            device = get_device_from_env()
            self.assertEqual(device.type, "cuda")
            self.assertEqual(device.index, device_idx)
            self.assertEqual(device.index, torch.cuda.current_device())

        invalid_device_idx = device_idx + 10
        with mock.patch.dict(os.environ, {"LOCAL_RANK": str(invalid_device_idx)}):
            with self.assertRaises(
                RuntimeError,
                msg="The local rank is larger than the number of available GPUs",
            ):
                device = get_device_from_env()

        # Test that we fall back to 0 if LOCAL_RANK is not specified
        device = get_device_from_env()
        self.assertEqual(device.type, "cuda")
        self.assertEqual(device.index, 0)
        self.assertEqual(device.index, torch.cuda.current_device())

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_copy_data_to_device_tensor(self) -> None:
        cuda_0 = torch.device("cuda:0")
        a = torch.tensor([1, 2, 3])
        self.assertEqual(a.device.type, "cpu")
        a = copy_data_to_device(a, cuda_0)
        self.assertEqual(a.device.type, "cuda")

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_copy_data_to_device_module(self) -> None:
        cuda_0 = torch.device("cuda:0")
        model = torch.nn.Linear(1, 1)
        for param in model.parameters():
            self.assertEqual(param.device.type, "cpu")
        model = copy_data_to_device(model, cuda_0)
        for param in model.parameters():
            self.assertEqual(param.device.type, "cuda")

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_copy_data_to_device_list(self) -> None:
        cuda_0 = torch.device("cuda:0")
        b = torch.tensor([1, 2, 3])
        c = torch.tensor([4, 5, 6])
        original_list = [b, c]
        self.assertEqual(b.device.type, "cpu")
        self.assertEqual(c.device.type, "cpu")
        new_list = copy_data_to_device(original_list, cuda_0)
        for elem in new_list:
            self.assertEqual(elem.device.type, "cuda")

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_copy_data_to_device_tuple(self) -> None:
        cuda_0 = torch.device("cuda:0")
        d = torch.tensor([1, 2, 3])
        e = torch.tensor([4, 5, 6])
        original_tuple = (d, e)
        self.assertEqual(d.device.type, "cpu")
        self.assertEqual(e.device.type, "cpu")
        new_tuple = copy_data_to_device(original_tuple, cuda_0)
        for elem in new_tuple:
            self.assertEqual(elem.device.type, "cuda")

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_copy_data_to_device_dict(self) -> None:
        cuda_0 = torch.device("cuda:0")
        f = torch.tensor([1, 2, 3])
        g = torch.tensor([4, 5, 6])
        original_dict = {"f": f, "g": g}
        self.assertEqual(f.device.type, "cpu")
        self.assertEqual(g.device.type, "cpu")
        new_dict = copy_data_to_device(original_dict, cuda_0)
        for key in new_dict.keys():
            self.assertEqual(new_dict[key].device.type, "cuda")

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_copy_data_to_device_named_tuple(self) -> None:
        cuda_0 = torch.device("cuda:0")

        # named tuple of tensors
        h = torch.tensor([1, 2, 3])
        i = torch.tensor([4, 5, 6])
        tensor_tuple = namedtuple("tensor_tuple", ["tensor_a", "tensor_b"])
        original_named_tuple = tensor_tuple(h, i)
        self.assertEqual(h.device.type, "cpu")
        self.assertEqual(i.device.type, "cpu")
        new_named_tuple = copy_data_to_device(original_named_tuple, cuda_0)
        for elem in new_named_tuple:
            self.assertEqual(elem.device.type, "cuda")

        self.assertIsNotNone(new_named_tuple.tensor_a)
        self.assertIsNotNone(new_named_tuple.tensor_b)
        self.assertEqual(type(original_named_tuple), type(new_named_tuple))

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_copy_data_to_device_dataclass(self) -> None:
        cuda_0 = torch.device("cuda:0")

        # dataclass of tensors
        @dataclass
        class TestTensorDataClass:
            val: torch.Tensor

        original_data_class = TestTensorDataClass(
            val=torch.tensor([1, 2, 3]),
        )
        self.assertEqual(original_data_class.val.device.type, "cpu")
        new_data_class = copy_data_to_device(original_data_class, cuda_0)
        self.assertEqual(new_data_class.val.device.type, "cuda")

        # frozen dataclass
        @dataclass(frozen=True)
        class FrozenDataClass:
            val: torch.Tensor

        original_data_class = FrozenDataClass(
            val=torch.tensor([1, 2, 3]),
        )
        self.assertEqual(original_data_class.val.device.type, "cpu")
        new_data_class = copy_data_to_device(original_data_class, cuda_0)
        self.assertEqual(new_data_class.val.device.type, "cuda")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            # pyre-fixme[41]: Cannot reassign final attribute `val`.
            new_data_class.val = torch.tensor([1, 2, 3], device=cuda_0)

        # no-init field
        @dataclass
        class NoInitDataClass:
            val: torch.Tensor = dataclasses.field(init=False)

            def __post_init__(self):
                self.val = torch.tensor([0, 1])

        original_data_class = NoInitDataClass()
        original_data_class.val = torch.tensor([1, 2])
        self.assertEqual(original_data_class.val.device.type, "cpu")
        new_data_class = copy_data_to_device(original_data_class, cuda_0)
        self.assertEqual(new_data_class.val.device.type, "cuda")
        self.assertTrue(
            torch.equal(new_data_class.val, torch.tensor([1, 2], device=cuda_0))
        )

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_copy_data_to_device_defaultdict(self) -> None:
        cuda_0 = torch.device("cuda:0")

        dd = defaultdict(torch.Tensor)
        dd[1] = torch.tensor([1, 2, 3])
        # dd[2] takes the default value, an empty tensor
        _ = dd[2]

        self.assertEqual(dd[1].device.type, "cpu")
        self.assertEqual(dd[2].device.type, "cpu")

        new_dd = copy_data_to_device(dd, cuda_0)

        self.assertEqual(new_dd[1].device, cuda_0)
        self.assertEqual(new_dd[2].device, cuda_0)

        # make sure the type of new keys is the same
        self.assertEqual(type(dd[3]), type(new_dd[3]))

    @unittest.skipUnless(
        condition=cuda_available, reason="This test needs a GPU host to run."
    )
    def test_copy_data_to_device_nested(self) -> None:
        h = torch.tensor([1, 2, 3])
        i = torch.tensor([4, 5, 6])
        j = torch.tensor([7, 8, 9])
        k = torch.tensor([10, 11])
        m = torch.tensor([12, 13])
        n = torch.tensor([14, 15])
        self.assertEqual(h.device.type, "cpu")
        self.assertEqual(i.device.type, "cpu")
        self.assertEqual(j.device.type, "cpu")
        self.assertEqual(k.device.type, "cpu")
        self.assertEqual(m.device.type, "cpu")
        self.assertEqual(n.device.type, "cpu")

        nested_list = [(h, i), (j, k)]
        nested_dict = {"1": nested_list, "2": [m], "3": n, "4": 2.0, "5": "string"}

        @dataclass
        class NestedDataClass:
            dict_container: Dict[str, Any]

        nested_data_class = NestedDataClass(dict_container=nested_dict)

        cuda_0 = torch.device("cuda:0")
        new_data_class = copy_data_to_device(nested_data_class, cuda_0)
        for val in new_data_class.dict_container.values():
            if isinstance(val, list):
                for list_item in val:
                    if isinstance(list_item, torch.Tensor):
                        self.assertEqual(list_item.device.type, "cuda")
                    if isinstance(list_item, tuple):
                        for tuple_item in list_item:
                            print(tuple_item)
                            self.assertEqual(tuple_item.device.type, "cuda")
            elif isinstance(val, torch.Tensor):
                self.assertEqual(val.device.type, "cuda")
            # check that float is unchanged
            elif isinstance(val, float):
                self.assertEqual(val, 2.0)
            # check that string is unchanged
            elif isinstance(val, str):
                self.assertEqual(val, "string")

    def test_get_cpu_stats(self) -> None:
        """Get CPU stats, check that values are populated."""
        cpu_stats = get_psutil_cpu_stats()
        # Check that percentages are between 0 and 100
        self.assertGreaterEqual(cpu_stats["cpu_vm_percent"], 0)
        self.assertLessEqual(cpu_stats["cpu_vm_percent"], 100)
        self.assertGreaterEqual(cpu_stats["cpu_percent"], 0)
        self.assertLessEqual(cpu_stats["cpu_percent"], 100)
        self.assertGreaterEqual(cpu_stats["cpu_swap_percent"], 0)
        self.assertLessEqual(cpu_stats["cpu_swap_percent"], 100)

    def test_get_gpu_stats(self) -> None:
        """Get Nvidia GPU stats, check that values are populated."""
        device = torch.device("cuda:0")

        with mock.patch("shutil.which"), mock.patch(
            "torchtnt.utils.device.subprocess.run"
        ) as subprocess_run_mock:
            subprocess_run_mock.return_value.stdout = "0, 0, 0, 2, 16273, 38, 15"
            gpu_stats = get_nvidia_smi_gpu_stats(device)

        # Check that percentages are between 0 and 100
        self.assertGreaterEqual(gpu_stats["utilization_gpu_percent"], 0)
        self.assertLessEqual(gpu_stats["utilization_gpu_percent"], 100)
        self.assertGreaterEqual(gpu_stats["utilization_memory_percent"], 0)
        self.assertLessEqual(gpu_stats["utilization_memory_percent"], 100)
        self.assertGreaterEqual(gpu_stats["fan_speed_percent"], 0)
        self.assertLessEqual(gpu_stats["fan_speed_percent"], 100)

        # Check that values are greater than zero
        self.assertGreaterEqual(gpu_stats["memory_used_mb"], 0)
        self.assertGreaterEqual(gpu_stats["memory_free_mb"], 0)
        self.assertGreaterEqual(gpu_stats["temperature_gpu_celsius"], 0)
        self.assertGreaterEqual(gpu_stats["temperature_memory_celsius"], 0)

    @unittest.skipUnless(
        condition=(cuda_available), reason="This test must run on a GPU host."
    )
    def test_record_data_in_stream_dict(self) -> None:
        curr_stream = torch.cuda.current_stream()
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        data = {"a": a, "b": b}

        with mock.patch.object(
            a, "record_stream"
        ) as mock_record_stream_a, mock.patch.object(
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

        with mock.patch.object(
            a, "record_stream"
        ) as mock_record_stream_a, mock.patch.object(
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

        with mock.patch.object(
            a, "record_stream"
        ) as mock_record_stream_a, mock.patch.object(
            b, "record_stream"
        ) as mock_record_stream_b:
            record_data_in_stream(data, curr_stream)
            mock_record_stream_a.assert_called_once()
            mock_record_stream_b.assert_called_once()

    @unittest.skipUnless(
        condition=(cuda_available), reason="This test must run on a GPU host."
    )
    def test_maybe_enable_tf32(self) -> None:
        maybe_enable_tf32("highest")
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

        maybe_enable_tf32("high")
        self.assertEqual(torch.get_float32_matmul_precision(), "high")
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
