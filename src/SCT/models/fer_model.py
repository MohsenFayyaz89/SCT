from typing import List

import torch.nn as nn
from torch import Tensor
from yacs.config import CfgNode

from .temporal_modules import WaveNetBlock


class Fer(nn.Module):
    """
    fe(X) & fr(Z)
    """
    def forward(self, x: Tensor) -> List[Tensor]:
        raise NotImplementedError


class WaveNet(Fer):
    def __init__(self, cfg: CfgNode, pooling_levels: List[int]):
        super().__init__()
        self.cfg = cfg.model.fer
        self.drop_on_frames = nn.Dropout(p=self.cfg.dropout_on_x)
        self.WaveNet = WaveNetBlock(
            in_channels=self.cfg.input_size, output_levels=self.cfg.output_levels,
            out_dims=self.cfg.hidden_size, pooling_levels=pooling_levels
        )
        self.group_norm = nn.GroupNorm(
            num_groups=self.cfg.gn_num_groups, num_channels=self.cfg.hidden_size
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        """

        :param x: 1 x D x T
        :return: 1 x D' x T'
        """
        # x = x.permute(0, 2, 1)  # [1 x T x D]
        x = self.drop_on_frames(x)
        # x = x.permute(0, 2, 1)  # [1 x D x T]
        outputs = self.WaveNet(x)
        outputs[-1] = self.group_norm(outputs[-1])
        return outputs
