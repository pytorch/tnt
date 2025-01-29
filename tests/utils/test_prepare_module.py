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
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtnt.utils.distributed import spawn_multi_process
from torchtnt.utils.env import init_from_env
from torchtnt.utils.prepare_module import (
    DDPStrategy,
    FSDPStrategy,
    materialize_meta_params,
    NOOPStrategy,
    on_meta_device,
    prepare_module,
    TorchCompileParams,
)
from torchtnt.utils.test_utils import skip_if_not_distributed
from torchtnt.utils.version import is_torch_version_geq


class PrepareModelTest(unittest.TestCase):
    torch_version_geq_2_1_0: bool = is_torch_version_geq("2.1.0")

    def test_invalid_fsdp_strategy_str_values(self) -> None:
        from torchtnt.utils.prepare_module import MixedPrecision as _MixedPrecision

        with self.assertRaisesRegex(ValueError, "Invalid BackwardPrefetch 'foo'"):
            FSDPStrategy(backward_prefetch="foo")

        with self.assertRaisesRegex(ValueError, "Invalid ShardingStrategy 'FOO'"):
            FSDPStrategy(sharding_strategy="FOO")

        with self.assertRaisesRegex(ValueError, "Invalid StateDictType 'FOO'"):
            FSDPStrategy(state_dict_type="FOO")

        with self.assertRaisesRegex(
            ValueError,
            "Invalid module class 'torch.nn.modules._BatchNorm': module 'torch.nn.modules' has no attribute '_BatchNorm'",
        ):
            FSDPStrategy(
                mixed_precision=_MixedPrecision(
                    _module_classes_to_ignore=[
                        # correct type is torch.nn.modules.batchnorm._BatchNorm
                        "torch.nn.modules._BatchNorm"
                    ]
                )
            )
        with self.assertRaisesRegex(
            ValueError,
            "Invalid module class 'foo.bar.Baz': No module named 'foo'",
        ):
            FSDPStrategy(
                mixed_precision=_MixedPrecision(
                    _module_classes_to_ignore=["foo.bar.Baz"]
                )
            )

    # # test strategy options
    def test_prepare_module_strategy_invalid_str(self) -> None:
        """
        Test that an exception is raised with an invalid strategy string
        """

        with self.assertRaisesRegex(ValueError, "Strategy foo not supported"):
            prepare_module(
                module=torch.nn.Linear(2, 2),
                device=init_from_env(),
                strategy="foo",
            )

    def test_prepare_module_invalid_strategy(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown strategy received"):
            prepare_module(
                module=torch.nn.Linear(2, 2),
                device=init_from_env(),
                # pyre-ignore: Incompatible parameter type [6] (intentional to test error raised)
                strategy={"_strategy_": "DDPStrategy"},
            )

    def test_prepare_noop(self) -> None:
        device = torch.device("cuda")  # Suppose init_from_env returns cuda

        module = torch.nn.Linear(2, 2)  # initialize on cpu
        module = prepare_module(module, device, strategy=NOOPStrategy())
        self.assertNotEqual(next(module.parameters()).device, device)

        module2 = torch.nn.Linear(2, 2)  # initialize on cpu
        module2 = prepare_module(module2, device, strategy="noop")
        self.assertNotEqual(next(module2.parameters()).device, device)

    @skip_if_not_distributed
    def test_prepare_module_with_ddp(self) -> None:
        """
        Launch tests of DDP strategy
        """

        spawn_multi_process(
            2,
            "gloo",
            self._test_prepare_module_ddp_strategy_wrapped_in_ddp,
        )
        spawn_multi_process(
            2,
            "gloo",
            self._test_prepare_module_ddp_string_wrapped_in_ddp,
        )
        spawn_multi_process(
            2,
            "gloo",
            self._test_prepare_module_ddp_throws_with_compile_params_and_static_graph,
        )

    @staticmethod
    def _test_prepare_module_ddp_strategy_wrapped_in_ddp() -> None:
        """
        Test that the module is correctly wrapped in DDP
        """

        ddp_module = prepare_module(
            module=torch.nn.Linear(2, 2),
            device=init_from_env(),
            strategy=DDPStrategy(),
        )
        tc = unittest.TestCase()

        tc.assertTrue(isinstance(ddp_module, DDP))

    @staticmethod
    def _test_prepare_module_ddp_string_wrapped_in_ddp() -> None:
        """
        Test that the module is correctly wrapped in DDP when passing "ddp" as a string
        """

        ddp_module = prepare_module(
            module=torch.nn.Linear(2, 2),
            device=init_from_env(),
            strategy="ddp",
        )
        tc = unittest.TestCase()

        tc.assertTrue(isinstance(ddp_module, DDP))

    @staticmethod
    def _test_prepare_module_ddp_throws_with_compile_params_and_static_graph() -> None:
        """
        Test that we throw an exception when we are using DDP static graph with compile params
        """

        tc = unittest.TestCase()
        with patch(
            "torchtnt.utils.prepare_module.is_torch_version_geq", return_value=False
        ):
            with tc.assertRaisesRegex(
                RuntimeError,
                "Torch version >= 2.1.0 required",
            ):
                prepare_module(
                    module=torch.nn.Linear(2, 2),
                    device=init_from_env(),
                    strategy=DDPStrategy(static_graph=True),
                    torch_compile_params=TorchCompileParams(backend="inductor"),
                )

    def test_prepare_module_compile_invalid_backend(self) -> None:
        """
        verify error is thrown on invalid backend
        """

        with self.assertRaises(Exception):
            prepare_module(
                module=torch.nn.Linear(2, 2),
                device=init_from_env(),
                torch_compile_params=TorchCompileParams(backend="foo"),
            )

    def test_prepare_module_incompatible_FSDP_torchcompile_params(self) -> None:
        """
        verify error is thrown when FSDP's use_orig_params and torch compile is enabled
        """

        with self.assertRaises(RuntimeError):
            prepare_module(
                module=torch.nn.Linear(2, 2),
                device=init_from_env(),
                strategy=FSDPStrategy(use_orig_params=False),
                torch_compile_params=TorchCompileParams(),
            )

    @unittest.skipUnless(
        torch_version_geq_2_1_0,
        reason="Must be on torch 2.1.0+ to run test",
    )
    def test_prepare_module_compile_module_state_dict(self) -> None:
        device = init_from_env()
        my_module = torch.nn.Linear(2, 2, device=device)
        my_module_state_dict = my_module.state_dict()
        self.assertIsNone(my_module._compiled_call_impl)
        compiled_module = prepare_module(
            module=my_module,
            device=device,
            torch_compile_params=TorchCompileParams(backend="inductor"),
        )
        compiled_state_dict = compiled_module.state_dict()
        self.assertCountEqual(compiled_state_dict.keys(), my_module_state_dict.keys())
        for k in compiled_state_dict.keys():
            self.assertTrue(
                torch.allclose(my_module_state_dict[k], compiled_state_dict[k])
            )
        self.assertIsNotNone(compiled_module._compiled_call_impl)

    @unittest.skipUnless(
        torch_version_geq_2_1_0,
        reason="Must be on torch 2.1.0+ to run test",
    )
    def test_materialize_meta_params(self) -> None:
        # Create a simple module with parameters on the meta device
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super(SimpleModule, self).__init__()
                self.linear1 = torch.nn.Linear(10, 10, device="meta")
                self.linear2 = torch.nn.Linear(10, 10, device="cpu")

        module = SimpleModule()
        device = torch.device("cpu")

        self.assertFalse(on_meta_device(module))  # top level module has no params
        self.assertTrue(on_meta_device(module.linear1))
        self.assertFalse(on_meta_device(module.linear2))

        # Call the function to test
        materialize_meta_params(module, device)

        # Check if the parameters are moved to the specified device
        for param in module.parameters():
            self.assertEqual(param.device, device)
