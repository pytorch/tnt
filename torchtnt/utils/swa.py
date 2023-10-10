# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional

import torch

from torch.optim.swa_utils import AveragedModel as PyTorchAveragedModel


TSWA_avg_fn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
TSWA_multi_avg_fn = Callable[[List[torch.Tensor], List[torch.Tensor], int], None]


class AveragedModel(PyTorchAveragedModel):
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        avg_fn: Optional[TSWA_avg_fn] = None,
        multi_avg_fn: Optional[TSWA_multi_avg_fn] = None,
        use_buffers: bool = False,
        skip_deepcopy: bool = False,
    ) -> None:
        """
        This class is a custom version of AveragedModel that allows us to skip the
        automatic deepcopy step. This gives flexibility to use modules that are not
        compatible with deepcopy, like FSDP wrapped modules. Check out
        https://github.com/pytorch/pytorch/blob/main/torch/optim/swa_utils.py#L66
        to see what the arguments entail.

        Args:
            skip_deepcopy: If True, will skip the deepcopy step. The user must ensure
                that the module passed in is already copied in someway
        """
        if skip_deepcopy:
            # calls parent init manually, but skips deepcopy step
            torch.nn.Module.__init__(self)  # inits grandparent class

            assert (
                avg_fn is None or multi_avg_fn is None
            ), "Only one of avg_fn and multi_avg_fn should be provided"
            self.module: torch.nn.Module = model
            self.register_buffer(
                "n_averaged", torch.tensor(0, dtype=torch.long, device=device)
            )
            self.avg_fn: Optional[TSWA_avg_fn] = avg_fn
            self.multi_avg_fn: Optional[TSWA_multi_avg_fn] = multi_avg_fn
            self.use_buffers: bool = use_buffers
        else:
            # use default init implementation

            # TODO: torch/optim/swa_utils.pyi needs to be updated
            # pyre-ignore Unexpected keyword [28]
            super().__init__(
                model,
                device=device,
                avg_fn=avg_fn,
                multi_avg_fn=multi_avg_fn,
                use_buffers=use_buffers,
            )
