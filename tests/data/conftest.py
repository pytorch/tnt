# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# displays parameterized arguments in pytest output
def pytest_make_parametrize_id(config, val, argname):
    return f"{argname}={val}"
