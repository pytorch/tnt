#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from functools import partial
from inspect import getmembers, isfunction
from typing import Tuple

import torch
from torchtnt.runner._test_utils import (
    DummyEvalUnit,
    DummyPredictUnit,
    DummyTrainUnit,
    generate_random_dataloader,
)
from torchtnt.runner.callback import Callback
from torchtnt.runner.callbacks.lambda_callback import Lambda
from torchtnt.runner.evaluate import evaluate
from torchtnt.runner.predict import predict
from torchtnt.runner.state import State
from torchtnt.runner.train import train
from torchtnt.runner.unit import TrainUnit

Batch = Tuple[torch.Tensor, torch.Tensor]


class DummyTrainExceptUnit(TrainUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()

    def train_step(self, state: State, data: Batch) -> None:
        raise RuntimeError("testing")


def _get_members_in_different_name(cls: Callback, phase: str):
    # retrieve Callback in different phases, including: train, predict, fit, eval
    return {
        h
        for h, _ in getmembers(cls, predicate=isfunction)
        if phase in h and (not h.startswith("_"))
    }


class LambdaTest(unittest.TestCase):
    def test_lambda_callback_train(self) -> None:
        input_dim = 2
        train_dataset_len = 10
        batch_size = 2
        max_epochs = 4
        max_steps_per_epoch = 6
        train_dataloader = generate_random_dataloader(
            train_dataset_len, input_dim, batch_size
        )
        checker = set()

        def call(hook, *_, **__):
            checker.add(hook)

        hooks = _get_members_in_different_name(Callback, "train")
        hooks_args = {h: partial(call, h) for h in hooks}
        my_train_unit = DummyTrainUnit(input_dim=input_dim)
        _ = train(
            my_train_unit,
            train_dataloader,
            callbacks=[Lambda(**hooks_args)],
            max_epochs=max_epochs,
            max_steps_per_epoch=max_steps_per_epoch,
        )
        self.assertEqual(checker, hooks)

    def test_lambda_callback_eval(self) -> None:
        input_dim = 2
        eval_dataset_len = 6
        batch_size = 2
        max_steps_per_epoch = 6
        eval_dataloader = generate_random_dataloader(
            eval_dataset_len, input_dim, batch_size
        )
        checker = set()

        def call(hook, *_, **__):
            checker.add(hook)

        hooks = _get_members_in_different_name(Callback, "eval")
        hooks_args = {h: partial(call, h) for h in hooks}
        my_eval_unit = DummyEvalUnit(input_dim=input_dim)
        _ = evaluate(
            my_eval_unit,
            eval_dataloader,
            callbacks=[Lambda(**hooks_args)],
            max_steps_per_epoch=max_steps_per_epoch,
        )
        self.assertEqual(checker, hooks)

    def test_lambda_callback_predict(self) -> None:
        input_dim = 2
        predict_dataset_len = 6
        batch_size = 2
        max_steps_per_epoch = 6
        checker = set()

        def call(hook, *_, **__):
            checker.add(hook)

        hooks = _get_members_in_different_name(Callback, "predict")
        hooks_args = {h: partial(call, h) for h in hooks}
        predict_dataloader = generate_random_dataloader(
            predict_dataset_len, input_dim, batch_size
        )
        my_predict_unit = DummyPredictUnit(input_dim=input_dim)
        _ = predict(
            my_predict_unit,
            predict_dataloader,
            callbacks=[Lambda(**hooks_args)],
            max_steps_per_epoch=max_steps_per_epoch,
        )
        self.assertEqual(checker, hooks)

    def test_lambda_callback_train_with_except(self) -> None:
        input_dim = 2
        train_dataset_len = 10
        batch_size = 2
        max_epochs = 4
        max_steps_per_epoch = 6
        train_dataloader = generate_random_dataloader(
            train_dataset_len, input_dim, batch_size
        )
        checker = set()

        def call(hook, *_, **__):
            checker.add(hook)

        # with on_exception, training will not be ended
        hooks = _get_members_in_different_name(Callback, "train") - {
            "on_train_end",
            "on_train_epoch_end",
            "on_train_step_end",
        }
        hooks.add("on_exception")
        hooks_args = {h: partial(call, h) for h in hooks}
        my_train_unit = DummyTrainExceptUnit(input_dim=input_dim)
        try:
            _ = train(
                my_train_unit,
                train_dataloader,
                callbacks=[Lambda(**hooks_args)],
                max_epochs=max_epochs,
                max_steps_per_epoch=max_steps_per_epoch,
            )
        except Exception:
            self.assertRaisesRegex(RuntimeError, "testing")
        self.assertEqual(checker, hooks)
