import torch.nn as nn
from torch import Tensor
from torch.nn.functional import interpolate
from yacs.config import CfgNode

from SCT.datasets.general_dataset import BatchItem
from SCT.models.data_classes import ForwardOut


class GeneralFre(nn.Module):

    def forward(self, z: Tensor, t: int) -> Tensor:
        raise NotImplementedError

    def loss(self, batch: BatchItem, forward_out: ForwardOut) -> Tensor:
        raise NotImplementedError


class Conv(GeneralFre):
    def __init__(self, cfg: CfgNode, input_size):
        super().__init__()
        self.cfg = cfg
        self.relu = nn.ReLU()
        self.reconstructor0 = nn.Conv1d(input_size, input_size*2, 1)
        self.reconstructor1 = nn.Conv1d(input_size*2, cfg.model.ft.input_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, z: Tensor, t: int) -> Tensor:
        """

        :param z:
        :param t:
        :return:
        """
        output = self.reconstructor0(z)
        output = self.relu(output)
        output = self.reconstructor1(output)  # [1 x D x T']
        output = interpolate(input=output, size=t, mode='nearest')
        return output  # [1 x D x T]

    def loss(self, batch: BatchItem, forward_out: ForwardOut) -> Tensor:
        """

        :param batch:
        :param forward_out:
        :return:
        """
        x_prime = self.forward(forward_out.masks_classes, batch.feats.shape[2])
        x = batch.feats
        loss = self.criterion(x_prime, x)
        return loss




