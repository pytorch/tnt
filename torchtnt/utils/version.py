#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import platform

import pkg_resources
import torch
from packaging.version import Version


def is_windows() -> bool:
    """
    Is the current program running in the Windows operating system?
    """
    return platform.system() == "Windows"


def get_python_version() -> Version:
    """
    Get the current runtime Python version as a Version.

    Example::

        # if running in Python 3.8.0
        >>> get_python_version()
        '3.8.0'
    """
    return Version(platform.python_version())


def get_torch_version() -> Version:
    """
    Get the PyTorch version for the current runtime environment as a Version.

    Example::

        # if running PyTorch 1.12.0
        >>> get_torch_version()
        '1.12.0'
    """
    try:
        if hasattr(torch, "__version__"):
            pkg_version = Version(torch.__version__)
        else:
            # try pkg_resources to infer version
            pkg_version = Version(pkg_resources.get_distribution("torch").version)
    except TypeError as e:
        raise TypeError("PyTorch version could not be detected automatically.") from e

    return pkg_version


def is_torch_version_geq_1_8() -> bool:
    return get_torch_version() >= Version("1.8.0")


def is_torch_version_geq_1_9() -> bool:
    return get_torch_version() >= Version("1.9.0")


def is_torch_version_geq_1_10() -> bool:
    return get_torch_version() >= Version("1.10.0")


def is_torch_version_geq_1_11() -> bool:
    return get_torch_version() >= Version("1.11.0")


def is_torch_version_geq_1_12() -> bool:
    return get_torch_version() >= Version("1.12.0")


def is_torch_version_geq_1_13() -> bool:
    return get_torch_version() >= Version("1.13.0")


def is_torch_version_ge_1_13_1() -> bool:
    return get_torch_version() > Version("1.13.1")


def is_torch_version_geq_1_14() -> bool:
    return get_torch_version() >= Version("1.14.0")


def is_torch_version_geq_2_0() -> bool:
    return get_torch_version() >= Version("2.0.0")
