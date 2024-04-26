# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

from torchtnt.utils.checkpoint import CheckpointPath, MetricData


class CheckpointPathTest(unittest.TestCase):
    def test_from_str(self) -> None:
        # invalid paths
        malformed_paths = [
            "foo/step_20",
            "foo/epoch_50",
            "epoch_30",
            "foo/epoch_20_step",
            "foo/epoch_20_step_30_val_loss=1a",
            "foo/epoch_2_step_15_mean=hello",
            "foo/epoch_2.6_step_23",
        ]
        for path in malformed_paths:
            with self.assertRaisesRegex(
                ValueError, f"Attempted to parse malformed checkpoint path: {path}"
            ):
                CheckpointPath.from_str(path)

        # valid paths
        valid_paths = [
            ("foo/epoch_0_step_1", CheckpointPath("foo", epoch=0, step=1)),
            (
                "foo/epoch_14_step_3_mean=15.0",
                CheckpointPath(
                    "foo", epoch=14, step=3, metric_data=MetricData("mean", 15.0)
                ),
            ),
            (
                "foo/epoch_14_step_3_loss=-27.35",
                CheckpointPath(
                    "foo", epoch=14, step=3, metric_data=MetricData("loss", -27.35)
                ),
            ),
            (
                "/foo/epoch_14_step_3_loss=-27.35",
                CheckpointPath(
                    "/foo", epoch=14, step=3, metric_data=MetricData("loss", -27.35)
                ),
            ),
            (
                "foo/bar/epoch_23_step_31_mean_loss_squared=0.0",
                CheckpointPath(
                    "foo/bar/",
                    epoch=23,
                    step=31,
                    metric_data=MetricData("mean_loss_squared", 0.0),
                ),
            ),
            (
                "file://some/path/checkpoints/0b20e70f-9ad2-4904-b7d6-e8da48087d61/epoch_2_step_1_acc=0.98",
                CheckpointPath(
                    "file://some/path/checkpoints/0b20e70f-9ad2-4904-b7d6-e8da48087d61",
                    epoch=2,
                    step=1,
                    metric_data=MetricData("acc", 0.98),
                ),
            ),
        ]
        for path, expected_ckpt in valid_paths:
            parsed_ckpt = CheckpointPath.from_str(path)
            self.assertEqual(parsed_ckpt, expected_ckpt)
            self.assertEqual(parsed_ckpt.path, path)

        # with a trailing slash
        ckpt = CheckpointPath.from_str("foo/epoch_0_step_1/")
        self.assertEqual(ckpt, CheckpointPath("foo", epoch=0, step=1))
        self.assertEqual(ckpt.path, "foo/epoch_0_step_1")

    def test_compare_by_recency(self) -> None:
        old = CheckpointPath("foo", epoch=0, step=1)
        new = CheckpointPath("foo", epoch=1, step=1)
        self.assertTrue(new.newer_than(old))
        self.assertFalse(old.newer_than(new))
        self.assertFalse(new == old)

        old = CheckpointPath("foo", epoch=3, step=5)
        new = CheckpointPath("foo", epoch=3, step=9)
        self.assertTrue(new.newer_than(old))
        self.assertFalse(old.newer_than(new))
        self.assertFalse(new == old)

        twin1 = CheckpointPath(
            "foo", epoch=2, step=5, metric_data=MetricData("foo", 1.0)
        )
        almost_twin = CheckpointPath(
            "foo", epoch=2, step=5, metric_data=MetricData("bar", 2.0)
        )

        self.assertFalse(twin1.newer_than(almost_twin))
        self.assertFalse(almost_twin.newer_than(twin1))
        self.assertFalse(twin1 == almost_twin)

        twin2 = CheckpointPath(
            "foo", epoch=2, step=5, metric_data=MetricData("foo", 1.0)
        )
        self.assertTrue(twin1 == twin2)

    def test_compare_by_optimality(self) -> None:
        # not both metric aware
        ckpt1 = CheckpointPath("foo", epoch=0, step=1)
        ckpt2 = CheckpointPath("foo", epoch=1, step=1)
        ckpt3 = CheckpointPath(
            "foo", epoch=1, step=1, metric_data=MetricData("bar", 1.0)
        )
        for ckpt in (ckpt2, ckpt3):
            with self.assertRaisesRegex(
                AssertionError,
                "Attempted to compare optimality of non metric-aware checkpoints",
            ):
                ckpt1.more_optimal_than(ckpt, mode="min")

        # tracking different metrics
        ckpt4 = CheckpointPath(
            "foo", epoch=1, step=1, metric_data=MetricData("baz", 1.0)
        )
        with self.assertRaisesRegex(
            AssertionError,
            "Attempted to compare optimality of checkpoints tracking different metrics",
        ):
            ckpt3.more_optimal_than(ckpt4, mode="min")

        smaller = CheckpointPath(
            "foo", epoch=0, step=1, metric_data=MetricData("foo", 1.0)
        )
        larger = CheckpointPath(
            "foo", epoch=0, step=1, metric_data=MetricData("foo", 2.0)
        )
        self.assertTrue(larger.more_optimal_than(smaller, mode="max"))
        self.assertFalse(smaller.more_optimal_than(larger, mode="max"))
        self.assertTrue(smaller.more_optimal_than(larger, mode="min"))
        self.assertFalse(larger.more_optimal_than(smaller, mode="min"))
