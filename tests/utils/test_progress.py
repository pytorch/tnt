#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtnt.utils.progress import Progress


class ProgressTest(unittest.TestCase):
    def test_progress_state_dict(self) -> None:
        """
        Test the state_dict and load_state_dict methods of Progress
        """
        progress = Progress(
            num_epochs_completed=2,
            num_steps_completed=8,
            num_steps_completed_in_epoch=4,
        )

        state_dict = progress.state_dict()

        new_progress = Progress()

        self.assertEqual(new_progress.num_epochs_completed, 0)
        self.assertEqual(new_progress.num_steps_completed, 0)
        self.assertEqual(new_progress.num_steps_completed_in_epoch, 0)

        new_progress.load_state_dict(state_dict)

        self.assertEqual(new_progress.num_epochs_completed, 2)
        self.assertEqual(new_progress.num_steps_completed, 8)
        self.assertEqual(new_progress.num_steps_completed_in_epoch, 4)
