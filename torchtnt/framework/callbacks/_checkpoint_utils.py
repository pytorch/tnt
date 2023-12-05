# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
from typing import List, Optional, Pattern

from pyre_extensions import none_throws
from torch import distributed as dist
from torchtnt.utils.distributed import get_global_rank, PGWrapper

from torchtnt.utils.fsspec import get_filesystem

logger: logging.Logger = logging.getLogger(__name__)


def get_latest_checkpoint_path(
    dirpath: str,
    metadata_fname: Optional[str] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Optional[str]:
    """
    Given a parent directory where checkpoints are saved, return the latest checkpoint subdirectory.

    Args:
        dirpath: parent directory where checkpoints are saved.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
        process_group: the process group on which the ranks will communicate on. default: ``None`` (the entire world)

    Raises:
        AssertionError if the checkpoint subdirectories are not named in the format epoch_{epoch}_step_{step}.
    """

    ret = None
    rank = get_global_rank()
    # Do all filesystem reads from rank 0 only
    if rank == 0:
        ret = _latest_checkpoint_path(dirpath, metadata_fname)

    # If not running in a distributed setting, return as is
    if not (dist.is_available() and dist.is_initialized()):
        return ret

    # Otherwise, broadcast result from rank 0 to all ranks
    pg = PGWrapper(process_group)
    path_container = [ret] if rank == 0 else [None]
    pg.broadcast_object_list(path_container, 0)
    val = path_container[0]
    return val


def _latest_checkpoint_path(
    dirpath: str, metadata_fname: Optional[str]
) -> Optional[str]:
    if dirpath[-1] == "/":
        # removes trailing forward slash if present
        # required for regex search to work
        dirpath = dirpath[:-1]

    fs = get_filesystem(dirpath)

    if not fs.exists(dirpath):
        logger.warning(f"Input dirpath doesn't exist: {dirpath}")
        return None

    contents = fs.ls(dirpath, detail=True)
    contents = [item["name"] for item in contents if item["type"] == "directory"]
    if len(contents) == 0:
        logger.warning(f"Input dirpath doesn't contain any subdirectories: {dirpath}")
        return None

    # Define the regex pattern to match the directory names
    pattern = rf"^{dirpath}/epoch_\d+_step_\d+"
    snapshot_dirpath_pattern: Pattern[str] = re.compile(pattern)
    candidate_dirpaths = list(filter(snapshot_dirpath_pattern.match, contents))

    if len(candidate_dirpaths) == 0:
        logger.warning(
            f"No valid checkpoint directories were found in input dirpath: {dirpath}"
        )
        return None

    # Initialize variables to store the largest epoch and step numbers
    largest_subdirectory = None
    largest_epoch = -1
    largest_step = -1

    # Iterate through all files and directories in the specified directory
    for candidate in candidate_dirpaths:
        if metadata_fname:
            dir_contents = fs.ls(candidate, False)
            if not any(metadata_fname == os.path.basename(f) for f in dir_contents):
                logger.warning(
                    f"Snapshot metadata is missing from {candidate}! Skipping this path"
                )
                continue

        # Extract the epoch and step numbers from the directory name
        dirname = os.path.basename(candidate)

        # dirname will be of the format epoch_N_step_M
        # where N is the epoch number and M is the step number as integers
        split = dirname.split("_")
        if len(split) != 4:
            raise AssertionError(
                f"Expected exactly 4 elements for pattern of epoch_N_step_M, but received {split})"
            )

        epoch_num, step_num = int(split[1]), int(split[3])
        # Check if the current epoch and step numbers are larger than the largest ones found so far
        if epoch_num > largest_epoch:
            largest_epoch = epoch_num
            largest_step = step_num
            largest_subdirectory = dirname
        elif largest_epoch == epoch_num and step_num > largest_step:
            largest_step = step_num
            largest_subdirectory = dirname

    if largest_subdirectory is None:
        return None

    # Rejoin with the parent directory path and return the largest subdirectory
    return os.path.join(dirpath, none_throws(largest_subdirectory))


def _retrieve_checkpoint_dirpaths(dirpath: str) -> List[str]:
    """
    Given a parent directory where checkpoints are saved, return the sorted checkpoint subdirectories
    from oldest to newest.

    Args:
        dirpath: parent directory where checkpoints are saved.
    """
    fs = get_filesystem(dirpath)

    contents = fs.ls(dirpath, detail=True)
    contents = [item["name"] for item in contents if item["type"] == "directory"]
    ckpt_dirpaths = []
    for path in contents:
        match = re.search(r"epoch_(\d+)_step_(\d+)", path)
        if match:
            ckpt_dirpaths.append(path)

    # sorts by epoch, then step
    ckpt_dirpaths.sort(key=lambda x: (int(x.split("_")[1]), int(x.split("_")[3])))
    return ckpt_dirpaths


def _delete_checkpoint(dirpath: str, metadata_fname: Optional[str] = None) -> None:
    fs = get_filesystem(dirpath)
    if metadata_fname and not fs.exists(os.path.join(dirpath, metadata_fname)):
        raise RuntimeError(f"{dirpath} does not contain {metadata_fname}")
    fs.rm(dirpath, recursive=True)
