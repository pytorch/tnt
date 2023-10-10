# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest
from copy import deepcopy
from typing import List, Tuple

import torch

# TODO: torch/optim/swa_utils.pyi needs to be updated
# pyre-ignore: Undefined import [21]
from torch.optim.swa_utils import get_ema_multi_avg_fn, get_swa_multi_avg_fn

from torchtnt.utils.swa import AveragedModel


class TestSWA(unittest.TestCase):
    def _test_averaged_model(
        self, net_device: torch.device, swa_device: torch.device, ema: bool
    ) -> None:
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(5, momentum=0.3),
            torch.nn.Conv2d(5, 2, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 10),
        ).to(net_device)

        averaged_params, averaged_dnn = self._run_averaged_steps(dnn, swa_device, ema)

        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            torch.testing.assert_close(p_avg, p_swa, check_device=False)
            # Check that AveragedModel is on the correct device
            self.assertTrue(p_swa.device == swa_device)
            self.assertTrue(p_avg.device == net_device)
        self.assertTrue(averaged_dnn.n_averaged.device == swa_device)

    def _run_averaged_steps(
        self, dnn: torch.nn.Module, swa_device: torch.device, ema: bool
    ) -> Tuple[List[torch.Tensor], torch.nn.Module]:
        ema_decay = 0.999
        multi_avg_fn = (
            # pyre-ignore: Undefined attribute [16]
            get_ema_multi_avg_fn(ema_decay)
            if ema
            # pyre-ignore: Undefined attribute [16]
            else get_swa_multi_avg_fn()
        )
        averaged_dnn = AveragedModel(
            dnn,
            device=swa_device,
            multi_avg_fn=multi_avg_fn,
        )

        averaged_params = [torch.zeros_like(param) for param in dnn.parameters()]

        n_updates = 10
        for i in range(n_updates):
            for p, p_avg in zip(dnn.parameters(), averaged_params):
                p.detach().add_(torch.randn_like(p))
                if ema:
                    p_avg += (
                        p.detach()
                        * ema_decay ** (n_updates - i - 1)
                        * ((1 - ema_decay) if i > 0 else 1.0)
                    )
                else:
                    p_avg += p.detach() / n_updates
            averaged_dnn.update_parameters(dnn)

        return averaged_params, averaged_dnn

    def test_averaged_model_all_devices(self) -> None:
        cpu = torch.device("cpu")
        self._test_averaged_model(cpu, cpu, ema=True)
        self._test_averaged_model(cpu, cpu, ema=False)
        if torch.cuda.is_available():
            cuda = torch.device(0)
            combos = itertools.product([cuda, cpu], [cuda, cpu], [True, False])
            for device1, device2, ema in combos:
                self._test_averaged_model(device1, device2, ema=ema)

    def test_averaged_model_state_dict(self) -> None:
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10)
        )
        averaged_dnn = AveragedModel(dnn)
        averaged_dnn2 = AveragedModel(dnn)
        n_updates = 10
        for _ in range(n_updates):
            for p in dnn.parameters():
                p.detach().add_(torch.randn_like(p))
            averaged_dnn.update_parameters(dnn)
        averaged_dnn2.load_state_dict(averaged_dnn.state_dict())
        for p_swa, p_swa2 in zip(averaged_dnn.parameters(), averaged_dnn2.parameters()):
            torch.testing.assert_close(p_swa, p_swa2, check_device=False)
        self.assertTrue(averaged_dnn.n_averaged == averaged_dnn2.n_averaged)

    def test_averaged_model_exponential(self) -> None:
        combos = itertools.product([True, False], [True, False], [True, False])
        for use_multi_avg_fn, use_buffers, skip_deepcopy in combos:
            self._test_averaged_model_exponential(
                use_multi_avg_fn, use_buffers, skip_deepcopy
            )

    def _test_averaged_model_exponential(
        self, use_multi_avg_fn: bool, use_buffers: bool, skip_deepcopy: bool
    ) -> None:
        # Test AveragedModel with EMA as avg_fn and use_buffers as True.
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.BatchNorm2d(5, momentum=0.3),
            torch.nn.Linear(5, 10),
        )
        decay: float = 0.9

        if use_multi_avg_fn:
            averaged_dnn = AveragedModel(
                deepcopy(dnn) if skip_deepcopy else dnn,
                # pyre-ignore Undefined attribute [16]
                multi_avg_fn=get_ema_multi_avg_fn(decay),
                use_buffers=use_buffers,
                skip_deepcopy=skip_deepcopy,
            )
        else:

            def avg_fn(
                p_avg: torch.Tensor, p: torch.Tensor, n_avg: float
            ) -> torch.Tensor:
                return decay * p_avg + (1 - decay) * p

            averaged_dnn = AveragedModel(dnn, avg_fn=avg_fn, use_buffers=use_buffers)

        if use_buffers:
            dnn_params = list(itertools.chain(dnn.parameters(), dnn.buffers()))
        else:
            dnn_params = list(dnn.parameters())

        averaged_params = [
            torch.zeros_like(param)
            for param in dnn_params
            if param.size() != torch.Size([])
        ]

        n_updates = 10
        for i in range(n_updates):
            updated_averaged_params = []
            for p, p_avg in zip(dnn_params, averaged_params):
                if p.size() == torch.Size([]):
                    continue
                p.detach().add_(torch.randn_like(p))
                if i == 0:
                    updated_averaged_params.append(p.clone())
                else:
                    updated_averaged_params.append(
                        (p_avg * decay + p * (1 - decay)).clone()
                    )
            averaged_dnn.update_parameters(dnn)
            averaged_params = updated_averaged_params

        if use_buffers:
            for p_avg, p_swa in zip(
                averaged_params,
                itertools.chain(
                    averaged_dnn.module.parameters(), averaged_dnn.module.buffers()
                ),
            ):
                torch.testing.assert_close(p_avg, p_swa, check_device=False)
        else:
            for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
                torch.testing.assert_close(p_avg, p_swa, check_device=False)
            for b_avg, b_swa in zip(dnn.buffers(), averaged_dnn.module.buffers()):
                torch.testing.assert_close(b_avg, b_swa, check_device=False)

    def test_averaged_model_skip_deepcopy(self) -> None:
        device = torch.device("cpu")
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(5, momentum=0.3),
            torch.nn.Conv2d(5, 2, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 10),
        ).to(device)
        averaged_dnn = AveragedModel(dnn, device, skip_deepcopy=True)
        # check both modules are pointing to same reference (since no deepcopy)
        self.assertEqual(id(dnn), id(averaged_dnn.module))

        averaged_dnn2 = AveragedModel(dnn, device)
        self.assertNotEqual(id(dnn), id(averaged_dnn2.module))
