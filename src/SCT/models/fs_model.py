import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from yacs.config import CfgNode


class Fs(nn.Module):
    """
    Fs(Z)
    """
    # noinspection PyPep8Naming
    def forward(self, Z: Tensor, T: int) -> Tensor:
        """
        Predicts the temporal action probabilities S using the intermediate representation Z
        :param Z: intermediate representation [1 x D' x T']
        :param T: input video X temporal length [1]
        :return: S [1 x C x T]
        """
        raise NotImplementedError


class Conv(Fs):
    def __init__(self, cfg: CfgNode, num_classes):
        super().__init__()
        self.cfg = cfg
        self.classifier = nn.Conv1d(self.cfg.model.fs.hidden_size, num_classes, 1)

    # noinspection PyPep8Naming
    def forward(self, Z: Tensor, T: int) -> Tensor:
        out = self.classifier(Z)  # [1 x C x T']
        out = F.interpolate(out, T)  # [1 x C x T]
        return out  # [1 x C x T]
