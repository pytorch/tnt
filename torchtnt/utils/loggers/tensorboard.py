#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import atexit
import logging
from typing import Any, Dict, List, Mapping, Optional, Union

import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torchtnt.utils.distributed import get_global_rank, PGWrapper
from torchtnt.utils.loggers.logger import MetricLogger, Scalar

logger: logging.Logger = logging.getLogger(__name__)


class TensorBoardLogger(MetricLogger):
    """
    Simple logger for TensorBoard.

    On construction, the logger creates a new events file that logs
    will be written to.  If the environment variable `RANK` is defined,
    logger will only log if RANK = 0.

    Note:
        If using this logger with distributed training:

        - This logger should be constructed on all ranks
        - This logger can call collective operations
        - Logs will be written on rank 0 only
        - Logger must be constructed synchronously *after* initializing the distributed process group.

    Args:
        path (str): path to write logs to
        *args: Extra positional arguments to pass to SummaryWriter
        **kwargs: Extra keyword arguments to pass to SummaryWriter

    Examples::

        from torchtnt.utils.loggers import TensorBoardLogger
        logger = TensorBoardLogger(path="tmp/tb_logs")
        logger.log("accuracy", 23.56, 10)
        logger.close()
    """

    def __init__(self: TensorBoardLogger, path: str, *args: Any, **kwargs: Any) -> None:
        self._writer: Optional[SummaryWriter] = None

        self._rank: int = get_global_rank()
        self._sync_path_to_workers(path)

        if self._rank == 0:
            logger.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = SummaryWriter(log_dir=path, *args, **kwargs)
        else:
            logger.debug(
                f"Not logging metrics on this host because env RANK: {self._rank} != 0"
            )

        atexit.register(self.close)

    def _sync_path_to_workers(self: TensorBoardLogger, path: str) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            self._path: str = path
            return

        pg = PGWrapper(dist.group.WORLD)
        path_container: List[str] = [path] if self._rank == 0 else [""]
        pg.broadcast_object_list(path_container, 0)
        updated_path = path_container[0]
        if updated_path != path:
            # because the logger only logs on rank 0, if users pass in a different path
            # the logger will output the wrong `path` property, so we update it to match
            # the correct path.
            logger.info(f"Updating TensorBoard path to match rank 0: {updated_path}")
        self._path: str = updated_path

    @property
    def writer(self: TensorBoardLogger) -> Optional[SummaryWriter]:
        return self._writer

    @property
    def path(self: TensorBoardLogger) -> str:
        return self._path

    def log_dict(
        self: TensorBoardLogger, payload: Mapping[str, Scalar], step: int
    ) -> None:
        """Add multiple scalar values to TensorBoard.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int): step value to record
        """

        if self._writer:
            for k, v in payload.items():
                self.log(k, v, step)

    def log(self: TensorBoardLogger, name: str, data: Scalar, step: int) -> None:
        """Add scalar data to TensorBoard.

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int): step value to record
        """

        if self._writer:
            self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_text(self: TensorBoardLogger, name: str, data: str, step: int) -> None:
        """Add text data to summary.

        Args:
            name (string): tag name used to identify data
            data (string): string to save
            step (int): step value to record
        """

        if self._writer:
            self._writer.add_text(name, data, global_step=step)

    def log_hparams(
        self: TensorBoardLogger, hparams: Dict[str, Scalar], metrics: Dict[str, Scalar]
    ) -> None:
        """Add hyperparameter data to TensorBoard.

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            metrics (dict): dictionary of name of metric and corresponding values
        """

        if self._writer:
            self._writer.add_hparams(hparams, metrics)

    def log_image(self: TensorBoardLogger, *args: Any, **kwargs: Any) -> None:
        """Add image data to TensorBoard.


        Args:
            *args (Any): Positional arguments passed to SummaryWriter.add_image
            **kwargs(Any): Keyword arguments passed to SummaryWriter.add_image
        """
        writer = self._writer
        if writer:
            writer.add_image(*args, **kwargs)

    def log_images(self: TensorBoardLogger, *args: Any, **kwargs: Any) -> None:
        """Add batched image data to summary.

        Args:
            *args (Any): Positional arguments passed to SummaryWriter.add_images
            **kwargs(Any): Keyword arguments passed to SummaryWriter.add_images
        """
        writer = self._writer
        if writer:
            writer.add_images(*args, **kwargs)

    def log_audio(self: TensorBoardLogger, *args: Any, **kwargs: Any) -> None:
        """Add audio data to TensorBoard.

        Args:
            *args (Any): Positional arguments passed to SummaryWriter.add_audio
            **kwargs (Any): Keyword arguments passed to SummaryWriter.add_audio
        """
        writer = self._writer
        if writer:
            writer.add_audio(*args, **kwargs)

    def log_scalars(
        self: TensorBoardLogger,
        main_tag: str,
        tag_scalar_dict: Dict[str, Union[float, int]],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        """Log multiple values to TensorBoard.
        Args:
            main_tag (string): Parent name for the tags
            tag_scalar_dict (dict): dictionary of tag name and scalar value
            global_step (int): global step value to record
            walltime (float): Optional override default walltime (time.time())
        Returns:
            None
        """
        if self._writer:
            self._writer.add_scalars(
                main_tag=main_tag,
                tag_scalar_dict=tag_scalar_dict,
                global_step=global_step,
                walltime=walltime,
            )

    def flush(self: TensorBoardLogger) -> None:
        """Writes pending logs to disk."""

        if self._writer:
            self._writer.flush()

    def close(self: TensorBoardLogger) -> None:
        """Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        """

        if self._writer:
            self._writer.close()
            self._writer = None
