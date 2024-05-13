# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torch import nn

from torchtnt.framework._test_utils import (
    DummyAutoUnit,
    DummyTrainUnit,
    get_dummy_eval_state,
    get_dummy_fit_state,
    get_dummy_train_state,
)

from torchtnt.framework.callbacks._checkpoint_utils import (
    _get_step_phase_mapping,
    _prepare_app_state_for_checkpoint,
)
from torchtnt.utils.checkpoint import Phase


class CheckpointUtilsTest(unittest.TestCase):

    def test_get_app_state(self) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        state = get_dummy_train_state()

        app_state = _prepare_app_state_for_checkpoint(state, my_unit, intra_epoch=False)
        self.assertCountEqual(
            app_state.keys(),
            ["module", "optimizer", "loss_fn", "train_progress"],
        )

    def test_get_step_phase_mapping(self) -> None:
        unit = DummyAutoUnit(module=nn.Linear(2, 2))
        unit.train_progress._num_steps_completed = 5
        unit.eval_progress._num_steps_completed = 7

        fit_state = get_dummy_fit_state()
        self.assertEqual(
            {Phase.TRAIN: 5, Phase.EVALUATE: 7},
            _get_step_phase_mapping(fit_state, unit),
        )

        train_state = get_dummy_train_state()
        self.assertEqual({Phase.TRAIN: 5}, _get_step_phase_mapping(train_state, unit))

        eval_state = get_dummy_eval_state()
        self.assertEqual({Phase.EVALUATE: 7}, _get_step_phase_mapping(eval_state, unit))
