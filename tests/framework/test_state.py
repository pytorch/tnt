#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchtnt.framework.state import _check_loop_condition, PhaseState


class StateTest(unittest.TestCase):
    def test_check_loop_condition(self) -> None:
        var = "foo"
        _check_loop_condition(var, None)
        _check_loop_condition(var, 100)
        with self.assertRaisesRegex(ValueError, f"Invalid value provided for {var}"):
            _check_loop_condition(var, -1)

    def test_phase_state_validation(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Invalid value provided for max_epochs"
        ):
            PhaseState(dataloader=[], max_epochs=-2)
        with self.assertRaisesRegex(ValueError, "Invalid value provided for max_steps"):
            PhaseState(dataloader=[], max_steps=-2)
        with self.assertRaisesRegex(
            ValueError, "Invalid value provided for max_steps_per_epoch"
        ):
            PhaseState(dataloader=[], max_steps_per_epoch=-2)
        with self.assertRaisesRegex(
            ValueError, "Invalid value provided for evaluate_every_n_steps"
        ):
            PhaseState(dataloader=[], evaluate_every_n_steps=-2)
        with self.assertRaisesRegex(
            ValueError, "Invalid value provided for evaluate_every_n_epochs"
        ):
            PhaseState(dataloader=[], evaluate_every_n_epochs=-2)
