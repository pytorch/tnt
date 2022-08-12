# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Any, Dict

import torch
import torch.nn as nn

from torchtnt.runner.state import State

"""
This file defines mixins and interfaces for users to customize hooks in training, evaluation, and prediction loops.
"""


def _remove_from_dicts(name_to_remove: str, *dicts: Dict[str, Any]) -> None:
    for d in dicts:
        if name_to_remove in d:
            del d[name_to_remove]


class _AppStateMixin:
    """
    A mixin to track modules, optimizers, and LR schedulers to simplify checkpointing object states.
    This can be easily extended to cover types that conform to the Stateful protocol.
    The logic here is adapted from torch.nn.Module's handling to register states for buffers, parameters, and modules.
    """

    def __init__(self) -> None:
        self._modules: "OrderedDict[str, nn.Module]" = OrderedDict()
        self._optimizers: "OrderedDict[str, torch.optim.Optimizer]" = OrderedDict()
        self._lr_schedulers: "OrderedDict[str, torch.optim.lr_scheduler._LRScheduler]" = (
            OrderedDict()
        )
        # TODO: include other known statefuls
        # TODO: include catch-all for miscellaneous statefuls

    def app_state(self) -> Dict[str, Any]:
        """Join together all of the tracked stateful entities to simplify registration of snapshottable states"""
        # TODO: refine with Stateful typehint
        # TODO: Should we split this into app_state_to_load and app_state_to_save
        # in order to let users customize the saving & loading paths independently?
        # or should we assume this is done outside of the loop framework entirely?
        app_state = {**self._modules, **self._optimizers, **self._lr_schedulers}
        return app_state

    def tracked_modules(self) -> "OrderedDict[str, nn.Module]":
        return self._modules

    def tracked_optimizers(self) -> "OrderedDict[str, torch.optim.Optimizer]":
        return self._optimizers

    def tracked_lr_schedulers(
        self,
    ) -> "OrderedDict[str, torch.optim.lr_scheduler._LRScheduler]":
        return self._lr_schedulers

    # pyre-ignore: Missing return annotation [3]
    def __getattr__(self, name: str) -> Any:
        if "_modules" in self.__dict__:
            _modules = self.__dict__["_modules"]
            if name in _modules:
                return _modules[name]
        if "_optimizers" in self.__dict__:
            _optimizers = self.__dict__["_optimizers"]
            if name in _optimizers:
                return _optimizers[name]
        if "_lr_schedulers" in self.__dict__:
            _lr_schedulers = self.__dict__["_lr_schedulers"]
            if name in _lr_schedulers:
                return _lr_schedulers[name]
        # pyre-ignore: Undefined attribute [16]
        return super().__getattr__(name)

    # pyre-ignore: Missing parameter annotation [2]
    def __setattr__(self, name: str, value: Any) -> None:
        _modules = self.__dict__.get("_modules")
        if isinstance(value, nn.Module):
            if _modules is None:
                raise AttributeError(
                    "cannot assign modules before _AppStateMixin.__init__() call"
                )
            _remove_from_dicts(
                name,
                self.__dict__,
                self._modules,
                self._optimizers,
                self._lr_schedulers,
            )
            self._modules[name] = value
        elif _modules is not None and name in _modules:
            if value is not None:
                raise TypeError(
                    "cannot assign '{}' as parameter '{}' "
                    "(torch.nn.Module or None expected)".format(
                        torch.typename(value), name
                    )
                )
            else:
                del self._modules[name]
                super().__setattr__(name, value)
        else:
            _optimizers = self.__dict__.get("_optimizers")
            if isinstance(value, torch.optim.Optimizer):
                if _optimizers is None:
                    raise AttributeError(
                        "cannot assign optimizer before _AppStateMixin.__init__() call"
                    )
                _remove_from_dicts(
                    name,
                    self.__dict__,
                    self._modules,
                    self._optimizers,
                    self._lr_schedulers,
                )
                _optimizers[name] = value
            elif _optimizers is not None and name in _optimizers:
                if value is not None:
                    raise TypeError(
                        "cannot assign '{}' as optimizer '{}' "
                        "(torch.optim.Optimizer or None expected)".format(
                            torch.typename(value), name
                        )
                    )
                else:
                    del self._optimizers[name]
                    super().__setattr__(name, value)
            else:
                _lr_schedulers = self.__dict__.get("_lr_schedulers")
                if isinstance(value, torch.optim.lr_scheduler._LRScheduler):
                    if _lr_schedulers is None:
                        raise AttributeError(
                            "cannot assign LR Scheduler before _AppStateMixin.__init__() call"
                        )
                    _remove_from_dicts(
                        name,
                        self.__dict__,
                        self._modules,
                        self._optimizers,
                        self._lr_schedulers,
                    )
                    _lr_schedulers[name] = value
                elif _lr_schedulers is not None and name in _lr_schedulers:
                    if value is not None:
                        raise TypeError(
                            "cannot assign '{}' as LR Scheduler '{}' "
                            "(torch.optim._LRScheduler or None expected)".format(
                                torch.typename(value), name
                            )
                        )
                    else:
                        del self._lr_schedulers[name]
                        super().__setattr__(name, value)
                else:
                    super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._modules:
            del self._modules[name]
        elif name in self._optimizers:
            del self._optimizers[name]
        elif name in self._lr_schedulers:
            del self._lr_schedulers[name]
        else:
            super().__delattr__(name)


class _OnExceptionMixin:
    def on_exception(self, state: State, exc: BaseException) -> None:
        pass
