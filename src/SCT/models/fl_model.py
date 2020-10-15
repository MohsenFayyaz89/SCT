import torch.nn as nn
from torch import Tensor
from yacs.config import CfgNode


class Fl(nn.Module):
    """
    Abstract class for Fl(Z'): estimates the temporal length of each region r over Z
    """
    def forward(self, z_prime: Tensor) -> Tensor:
        raise NotImplementedError


class Conv(Fl):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg
        D_prime = self.cfg.model.fer.hidden_size
        self.conv1 = nn.Conv1d(D_prime, int(D_prime/2), 1)
        self.activation_1 = nn.ReLU()
        self.conv2 = nn.Conv1d(int(D_prime/2), 1, 1)
        self.activation_2 = nn.ReLU()

    def forward(self, z_prime: Tensor):
        out = self.conv1(z_prime)
        # out = self.activation_1(out)
        out = self.conv2(out)
        # out = self.activation_2(out)

        return out
