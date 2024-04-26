# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchtnt.framework._test_utils import DummyTrainUnit, get_dummy_train_state

from torchtnt.framework.callbacks._checkpoint_utils import (
    _prepare_app_state_for_checkpoint,
)


class CheckpointUtilsTest(unittest.TestCase):

    def test_get_app_state(self) -> None:
        my_unit = DummyTrainUnit(input_dim=2)
        state = get_dummy_train_state()

        app_state = _prepare_app_state_for_checkpoint(state, my_unit, intra_epoch=False)
        self.assertCountEqual(
            app_state.keys(),
            ["module", "optimizer", "loss_fn", "train_progress"],
        )
