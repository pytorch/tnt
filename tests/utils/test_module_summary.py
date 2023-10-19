#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Dict, Optional, Tuple

import torch
from torchtnt.utils.module_summary import (
    _get_human_readable_count,
    get_module_summary,
    ModuleSummary,
    prune_module_summary,
)
from torchvision import models


def get_summary_and_prune(
    model: torch.nn.Module,
    *,
    max_depth: int,
    # pyre-ignore
    module_args: Optional[Tuple[Any, ...]] = None,
    module_kwargs: Optional[Dict[str, Any]] = None,
) -> ModuleSummary:
    """Utility function to get module summary and then prune it"""
    module_summary = get_module_summary(
        model, module_args=module_args, module_kwargs=module_kwargs
    )
    prune_module_summary(module_summary, max_depth=max_depth)
    return module_summary


class ModuleSummaryTest(unittest.TestCase):

    maxDiff = 10000

    def _test_module_summary_text(self, ms1: str, ms2: str) -> None:
        # utility method to make testing summary method text more robust to terminal differences
        for l1, l2 in zip(ms1.strip().split("\n"), ms2.strip().split("\n")):
            self.assertEqual(l1.strip(), l2.strip())

    def test_module_summary_layer(self) -> None:
        """Make sure ModuleSummary works for a single layer."""
        model = torch.nn.Conv2d(3, 8, 3)
        ms1 = get_module_summary(model)
        ms2 = get_module_summary(model, module_args=(torch.randn(1, 3, 8, 8),))

        self.assertEqual(ms1.module_name, "")
        self.assertEqual(ms1.module_type, "Conv2d")
        self.assertEqual(ms1.num_parameters, 224)
        self.assertEqual(ms1.num_trainable_parameters, 224)
        self.assertEqual(ms1.size_bytes, 224 * 4)
        self.assertEqual(ms1.submodule_summaries, {})
        self.assertFalse(ms1.has_uninitialized_param)

        self.assertEqual(ms1.module_name, ms2.module_name)
        self.assertEqual(ms1.module_type, ms2.module_type)
        self.assertEqual(ms1.num_parameters, ms2.num_parameters)
        self.assertEqual(ms1.num_trainable_parameters, ms2.num_trainable_parameters)
        self.assertEqual(ms1.size_bytes, ms2.size_bytes)
        self.assertEqual(ms1.submodule_summaries, ms2.submodule_summaries)

        self.assertEqual(ms2.flops_forward, 7776)
        self.assertEqual(ms2.flops_backward, 7776)

        self.assertEqual(ms1.in_size, "?")
        self.assertEqual(ms1.out_size, "?")
        self.assertEqual(ms2.in_size, [1, 3, 8, 8])
        self.assertEqual(ms2.out_size, [1, 8, 6, 6])

    def test_activation_size(self) -> None:
        """Make sure activation size is correct for more complex module"""

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = torch.nn.Linear(10, 5)
                self.relu = torch.nn.ReLU()
                self.out = torch.nn.Linear(5, 3)
                self.softmax = torch.nn.Softmax(dim=3)

            def forward(self, x):
                x = self.hidden(x)
                x = self.relu(x)
                x = self.out(x)
                x = self.softmax(x)
                return x

        model = TestModule()
        ms = get_module_summary(model, (torch.randn(1, 3, 10, 10),))
        self.assertEqual(ms.module_name, "")
        self.assertEqual(ms.in_size, [1, 3, 10, 10])
        self.assertEqual(ms.out_size, [1, 3, 10, 3])
        self.assertEqual(ms.module_type, "TestModule")

        ms_hidden = ms.submodule_summaries["hidden"]
        self.assertEqual(ms_hidden.module_name, "hidden")
        self.assertEqual(ms_hidden.in_size, [1, 3, 10, 10])
        self.assertEqual(ms_hidden.out_size, [1, 3, 10, 5])
        self.assertEqual(ms_hidden.module_type, "Linear")

        ms_relu = ms.submodule_summaries["relu"]
        self.assertEqual(ms_relu.module_name, "relu")
        self.assertEqual(ms_relu.in_size, [1, 3, 10, 5])
        self.assertEqual(ms_relu.out_size, [1, 3, 10, 5])
        self.assertEqual(ms_relu.module_type, "ReLU")

        ms_out = ms.submodule_summaries["out"]
        self.assertEqual(ms_out.module_name, "out")
        self.assertEqual(ms_out.in_size, [1, 3, 10, 5])
        self.assertEqual(ms_out.out_size, [1, 3, 10, 3])
        self.assertEqual(ms_out.module_type, "Linear")

        ms_softmax = ms.submodule_summaries["softmax"]
        self.assertEqual(ms_softmax.module_name, "softmax")
        self.assertEqual(ms_softmax.in_size, [1, 3, 10, 3])
        self.assertEqual(ms_softmax.out_size, [1, 3, 10, 3])
        self.assertEqual(ms_softmax.module_type, "Softmax")

    def test_invalid_max_depth(self) -> None:
        """Test for ValueError when providing bad max_depth"""
        model = torch.nn.Conv2d(3, 8, 3)
        summary = get_module_summary(model)
        with self.assertRaisesRegex(ValueError, "Got -2."):
            prune_module_summary(summary, max_depth=-2)
        with self.assertRaisesRegex(ValueError, "Got 0."):
            prune_module_summary(summary, max_depth=0)

    def test_lazy_tensor(self) -> None:
        """Check for warning when passing in a lazy weight Tensor"""
        model = torch.nn.LazyLinear(10)
        ms = get_module_summary(model)
        with self.assertWarns(Warning):
            ms.num_parameters
        with self.assertWarns(Warning):
            ms.num_trainable_parameters
        self.assertTrue(ms.has_uninitialized_param)

    def test_lazy_tensor_flops(self) -> None:
        """Check for warnings when passing in a lazy weight Tensor
        Even when asking for flops calculation."""
        model = torch.nn.LazyLinear(10)
        ms = get_module_summary(model, module_args=(torch.randn(1, 10),))
        with self.assertWarns(Warning):
            ms.num_parameters
        with self.assertWarns(Warning):
            ms.num_trainable_parameters
        self.assertTrue(ms.has_uninitialized_param)
        self.assertEqual(ms.flops_backward, "?")
        self.assertEqual(ms.flops_forward, "?")

    def test_resnet_max_depth(self) -> None:
        """Test the behavior of max_depth on a layered model like ResNet"""
        pretrained_model = models.resnet.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # max_depth = None
        ms1 = get_module_summary(pretrained_model)

        self.assertEqual(len(ms1.submodule_summaries), 10)
        self.assertEqual(len(ms1.submodule_summaries["layer2"].submodule_summaries), 2)
        self.assertEqual(
            len(
                ms1.submodule_summaries["layer2"]
                .submodule_summaries["layer2.0"]
                .submodule_summaries
            ),
            6,
        )

        ms2 = get_summary_and_prune(pretrained_model, max_depth=1)
        self.assertEqual(len(ms2.submodule_summaries), 0)
        self.assertNotIn("layer2", ms2.submodule_summaries)

        ms3 = get_summary_and_prune(pretrained_model, max_depth=2)
        self.assertEqual(len(ms3.submodule_summaries), 10)
        self.assertEqual(len(ms1.submodule_summaries["layer2"].submodule_summaries), 2)
        self.assertNotIn(
            "layer2.0", ms3.submodule_summaries["layer2"].submodule_summaries
        )
        inp = torch.randn(1, 3, 224, 224)
        ms4 = get_summary_and_prune(pretrained_model, max_depth=2, module_args=(inp,))

        self.assertEqual(len(ms4.submodule_summaries), 10)
        self.assertEqual(ms4.flops_forward, 1814073344)
        self.assertEqual(ms4.flops_backward, 3510132736)
        self.assertEqual(ms4.submodule_summaries["layer2"].flops_forward, 411041792)
        self.assertEqual(ms4.submodule_summaries["layer2"].flops_backward, 822083584)

        # These should stay constant for all max_depth values
        ms_list = [ms1, ms2, ms3, ms4]
        for ms in ms_list:
            self.assertEqual(ms.module_name, "")
            self.assertEqual(ms.module_type, "ResNet")
            self.assertEqual(ms.num_parameters, 11689512)
            self.assertEqual(ms.num_trainable_parameters, 11689512)
            self.assertFalse(ms.has_uninitialized_param)

    def test_module_summary_layer_print(self) -> None:
        model = torch.nn.Conv2d(3, 8, 3)
        ms1 = get_module_summary(model)

        summary_table = """
Name | Type   | # Parameters | # Trainable Parameters | Size (bytes) | Contains Uninitialized Parameters?
---------------------------------------------------------------------------------------------------------
     | Conv2d | 224          | 224                    | 896          | No
"""
        self._test_module_summary_text(summary_table, str(ms1))

    def test_alexnet_print(self) -> None:
        pretrained_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        ms1 = get_summary_and_prune(pretrained_model, max_depth=1)
        ms2 = get_summary_and_prune(pretrained_model, max_depth=2)
        ms3 = get_summary_and_prune(pretrained_model, max_depth=3)
        ms4 = get_module_summary(pretrained_model)

        summary_table1 = """
Name | Type    | # Parameters | # Trainable Parameters | Size (bytes) | Contains Uninitialized Parameters?
----------------------------------------------------------------------------------------------------------
     | AlexNet | 61.1 M       | 61.1 M                 | 244 M        | No
"""
        summary_table2 = """
Name       | Type              | # Parameters | # Trainable Parameters | Size (bytes) | Contains Uninitialized Parameters?
--------------------------------------------------------------------------------------------------------------------------
           | AlexNet           | 61.1 M       | 61.1 M                 | 244 M        | No
features   | Sequential        | 2.5 M        | 2.5 M                  | 9.9 M        | No
avgpool    | AdaptiveAvgPool2d | 0            | 0                      | 0            | No
classifier | Sequential        | 58.6 M       | 58.6 M                 | 234 M        | No
"""

        self._test_module_summary_text(summary_table1, str(ms1))
        self._test_module_summary_text(summary_table2, str(ms2))
        self.assertEqual(str(ms3), str(ms4))

    def test_alexnet_with_input_tensor(self) -> None:
        pretrained_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        inp = torch.randn(1, 3, 224, 224)
        ms1 = get_summary_and_prune(pretrained_model, max_depth=1, module_args=(inp,))
        ms2 = get_summary_and_prune(pretrained_model, max_depth=2, module_args=(inp,))

        self.assertEqual(ms1.module_type, "AlexNet")
        self.assertEqual(ms1.num_parameters, 61100840)
        self.assertFalse(ms1.has_uninitialized_param)
        self.assertEqual(ms1.flops_forward, 714188480)
        self.assertEqual(ms1.flops_backward, 1358100160)
        self.assertEqual(ms1.in_size, [1, 3, 224, 224])
        self.assertEqual(ms1.out_size, [1, 1000])

        ms_features = ms2.submodule_summaries["features"]
        self.assertEqual(ms_features.module_type, "Sequential")
        self.assertFalse(ms_features.has_uninitialized_param)
        self.assertEqual(ms_features.flops_forward, 655566528)
        self.assertEqual(ms_features.flops_backward, 1240856256)
        self.assertEqual(ms_features.in_size, [1, 3, 224, 224])
        self.assertEqual(ms_features.out_size, [1, 256, 6, 6])

        ms_avgpool = ms2.submodule_summaries["avgpool"]
        self.assertEqual(ms_avgpool.module_type, "AdaptiveAvgPool2d")
        self.assertFalse(ms_avgpool.has_uninitialized_param)
        self.assertEqual(ms_avgpool.flops_forward, 0)
        self.assertEqual(ms_avgpool.flops_backward, 0)
        self.assertEqual(ms_avgpool.in_size, [1, 256, 6, 6])
        self.assertEqual(ms_avgpool.out_size, [1, 256, 6, 6])

        ms_classifier = ms2.submodule_summaries["classifier"]
        self.assertEqual(ms_classifier.module_type, "Sequential")
        self.assertFalse(ms_classifier.has_uninitialized_param)
        self.assertEqual(ms_classifier.flops_forward, 58621952)
        self.assertEqual(ms_classifier.flops_backward, 117243904)
        self.assertEqual(ms_classifier.in_size, [1, 9216])
        self.assertEqual(ms_classifier.out_size, [1, 1000])

    def test_get_human_readable_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "received -1"):
            _get_human_readable_count(-1)
        with self.assertRaisesRegex(TypeError, "received <class 'float'>"):
            # pyre-fixme[6]: For 1st param expected `int` but got `float`.
            _get_human_readable_count(0.1)
        self.assertEqual(_get_human_readable_count(1), "1  ")
        self.assertEqual(_get_human_readable_count(123), "123  ")
        self.assertEqual(_get_human_readable_count(1234), "1.2 K")
        self.assertEqual(_get_human_readable_count(1254), "1.3 K")
        self.assertEqual(_get_human_readable_count(1960), "2.0 K")
        self.assertEqual(_get_human_readable_count(int(1e4)), "10.0 K")
        self.assertEqual(_get_human_readable_count(int(1e6)), "1.0 M")
        self.assertEqual(_get_human_readable_count(int(1e9)), "1.0 B")
        self.assertEqual(_get_human_readable_count(int(1e12)), "1.0 T")
        self.assertEqual(_get_human_readable_count(int(1e15)), "1,000 T")

    def test_module_summary_multiple_inputs(self) -> None:
        class SimpleConv(torch.nn.Module):
            def __init__(self):
                super(SimpleConv, self).__init__()
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=True),
                )
                self.offset = 1

            def forward(self, x, y, offset=1):
                self.offset = offset
                x1 = self.features(x)
                x2 = self.features(y)
                x = torch.cat((x1, x2), 1)
                return x

        model = SimpleConv()
        x = torch.randn(1, 1, 224, 224)
        y = torch.randn(1, 1, 224, 224)

        ms1 = get_summary_and_prune(model, max_depth=3, module_args=(x, y))

        self.assertEqual(ms1.module_type, "SimpleConv")
        self.assertEqual(ms1.num_parameters, 10)
        self.assertFalse(ms1.has_uninitialized_param)
        self.assertEqual(ms1.in_size, [[1, 1, 224, 224], [1, 1, 224, 224]])
        self.assertEqual(ms1.out_size, [1, 2, 224, 224])

        ms_features = ms1.submodule_summaries["features"]
        self.assertEqual(ms_features.module_type, "Sequential")
        self.assertFalse(ms_features.has_uninitialized_param)
        self.assertEqual(ms_features.in_size, [1, 1, 224, 224])
        self.assertEqual(ms_features.out_size, [1, 1, 224, 224])

        ms_avgpool = ms1.submodule_summaries["features"].submodule_summaries[
            "features.0"
        ]
        self.assertEqual(ms_avgpool.module_type, "Conv2d")
        self.assertFalse(ms_avgpool.has_uninitialized_param)
        self.assertEqual(ms_avgpool.in_size, [1, 1, 224, 224])
        self.assertEqual(ms_avgpool.out_size, [1, 1, 224, 224])

        ms_classifier = ms1.submodule_summaries["features"].submodule_summaries[
            "features.1"
        ]
        self.assertEqual(ms_classifier.module_type, "ReLU")
        self.assertFalse(ms_classifier.has_uninitialized_param)
        self.assertEqual(ms_classifier.flops_forward, 0)
        self.assertEqual(ms_classifier.flops_backward, 0)
        self.assertEqual(ms_classifier.in_size, [1, 1, 224, 224])
        self.assertEqual(ms_classifier.out_size, [1, 1, 224, 224])

    def test_forward_elapsed_time(self) -> None:
        pretrained_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        inp = torch.randn(1, 3, 224, 224)
        ms1 = get_summary_and_prune(pretrained_model, module_args=(inp,), max_depth=4)
        stack = [ms1] + [
            ms1.submodule_summaries[key] for key in ms1.submodule_summaries
        ]
        # check all submodule summaries have been timed
        for ms in stack:
            self.assertNotEqual(ms.forward_elapsed_time_ms, "?")
            self.assertGreater(float(ms.forward_elapsed_time_ms), 0)
            stack.extend(
                [ms.submodule_summaries[key] for key in ms.submodule_summaries]
            )
