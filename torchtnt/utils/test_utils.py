#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import uuid

import torch.distributed.launcher as pet


def get_pet_launch_config(nproc: int) -> pet.LaunchConfig:
    """
    Initialize pet.LaunchConfig for single-node, multi-rank functions.

    Args:
        nproc: The number of processes to launch.

    Returns:
        An instance of pet.LaunchConfig for single-node, multi-rank functions.
    """
    return pet.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=nproc,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        monitor_interval=1,
    )
