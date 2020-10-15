import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List


class WaveNetLayer(nn.Module):
    def __init__(
        self, num_channels: int, kernel_size: int, dilation: int, drop: float = 0.25
    ):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dilated_conv = nn.Conv1d(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.dilation,
        )
        self.conv_1x1 = nn.Conv1d(
            in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=1
        )
        self.drop = nn.Dropout(drop)

    @staticmethod
    def apply_non_lin(y: Tensor) -> Tensor:
        return F.relu(y)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: [B x num_channels x T]
        :return: [B x num_channels x T]
        """
        y = self.dilated_conv.forward(x)
        y = self.apply_non_lin(y)  # non-linearity
        y = self.conv_1x1.forward(y)
        y = self.drop.forward(y)  # dropout
        y += x  # residual connection
        return y


class WaveNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stages: List[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
        pooling_levels: List[int] = [1, 2, 4, 8, 16],
        output_levels: List[int] = [],
        out_dims: int = 64,
        kernel_size: int = 3,
        pooling=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_stages = len(stages)
        self.stages = stages
        self.out_dims = out_dims
        self.kernel_size = kernel_size
        self.layers = []
        self.pooling = pooling
        self.pooling_levels = pooling_levels
        self.output_levels = output_levels

        self.first_conv = nn.Conv1d(
            in_channels=self.in_channels, out_channels=self.out_dims, kernel_size=1
        )
        self.last_conv = nn.Conv1d(
            in_channels=self.out_dims, out_channels=self.out_dims, kernel_size=1
        )

        for i in range(self.num_stages):
            stage = self.stages[i]
            layer = WaveNetLayer(
                self.out_dims, kernel_size=self.kernel_size, dilation=stage
            )
            self.layers.append(layer)
            self.add_module("l_{}".format(i), layer)

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        :param x: [B x in_channels x T]
        :return: [B x out_dims x T]
        """
        outputs = []
        x = F.relu(self.first_conv.forward(x))
        # fixme: clean the code
        pooling_levels = self.pooling_levels
        # if x.shape[2] > 10000:
        # pooling_levels = self.pooling_levels
        for i, l in enumerate(self.layers):
            x = l.forward(x)
            if i in pooling_levels and self.pooling:
                # fixme: clean the code
                # print("\n{}".format(x.shape))
                x = F.max_pool1d(x, kernel_size=2)
                # fixme: clean the code
                # print("  {}".format(x.shape))
            if i in self.output_levels:
                outputs.append(x)
        x = F.relu(x)
        x = self.last_conv.forward(x)
        outputs.append(x)

        return outputs
