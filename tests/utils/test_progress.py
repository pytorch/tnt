#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtnt.framework._test_utils import generate_random_dataloader

from torchtnt.utils.progress import estimated_steps_in_epoch, Progress


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

    def test_estimated_steps_in_epoch(self) -> None:

        input_dim = 2
        dataset_len = 20
        batch_size = 2
        dataloader_size = dataset_len / batch_size

        dataloader = generate_random_dataloader(dataset_len, input_dim, batch_size)

        self.assertEqual(
            estimated_steps_in_epoch(
                dataloader, num_steps_completed=0, max_steps=5, max_steps_per_epoch=5
            ),
            5,
        )
        self.assertEqual(
            estimated_steps_in_epoch(
                dataloader, num_steps_completed=4, max_steps=5, max_steps_per_epoch=4
            ),
            1,
        )
        self.assertEqual(
            estimated_steps_in_epoch(
                dataloader, num_steps_completed=0, max_steps=4, max_steps_per_epoch=10
            ),
            4,
        )
        self.assertEqual(
            estimated_steps_in_epoch(
                dataloader,
                num_steps_completed=0,
                max_steps=None,
                max_steps_per_epoch=None,
            ),
            dataloader_size,
        )
        self.assertEqual(
            estimated_steps_in_epoch(
                dataloader,
                num_steps_completed=0,
                max_steps=None,
                max_steps_per_epoch=500,
            ),
            dataloader_size,
        )
