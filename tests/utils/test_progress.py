#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtnt.framework._test_utils import generate_random_dataloader

from torchtnt.utils.progress import (
    estimated_steps_in_epoch,
    estimated_steps_in_loop,
    Progress,
)


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

    def test_estimated_steps_in_loop(self) -> None:
        dataset_len = 10
        batch_size = 2
        dataloader = generate_random_dataloader(
            num_samples=dataset_len, input_dim=2, batch_size=batch_size
        )

        self.assertEqual(
            estimated_steps_in_loop(
                dataloader,
                max_steps=20,
                max_steps_per_epoch=6,
                epochs=3,
            ),
            15,  # 5 steps per epoch because the dataset would be exhausted after that
        )

        self.assertEqual(
            estimated_steps_in_loop(
                dataloader,
                max_steps=20,
                max_steps_per_epoch=4,
                epochs=3,
            ),
            12,  # 4 steps per epoch, not exhausting all samples
        )

        self.assertEqual(
            estimated_steps_in_loop(
                dataloader,
                max_steps=8,
                max_steps_per_epoch=6,
                epochs=3,
            ),
            8,  # we finish in the 'middle' of an epoch because of max_steps
        )

        self.assertEqual(
            estimated_steps_in_loop(
                dataloader,
                max_steps=None,
                max_steps_per_epoch=3,
                epochs=3,
            ),
            9,  # when max_steps is none, we use epochs
        )

        self.assertEqual(
            estimated_steps_in_loop(
                dataloader,
                max_steps=None,
                max_steps_per_epoch=None,
                epochs=3,
            ),
            15,  # when max_steps is none, we use epochs
        )

        self.assertEqual(
            estimated_steps_in_loop(
                dataloader,
                max_steps=7,
                max_steps_per_epoch=5,
                epochs=None,
            ),
            7,  # when epoch is none, we use max_steps
        )

        self.assertEqual(
            estimated_steps_in_loop(
                dataloader,
                max_steps=None,
                max_steps_per_epoch=4,
                epochs=None,
            ),
            None,
        )
