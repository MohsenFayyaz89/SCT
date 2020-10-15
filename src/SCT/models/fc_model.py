import torch.nn as nn
from torch import Tensor
from yacs.config import CfgNode


class Fc(nn.Module):
    """
    Abstract class for Fc(Z'): predicts the action probabilities for every region r over Z
    """
    def forward(self, z_prime: Tensor) -> Tensor:
        raise NotImplementedError


class Conv(Fc):
    def __init__(self, cfg: CfgNode, num_classes):
        super().__init__()
        self.cfg = cfg
        self.classifier = nn.Conv1d(self.cfg.model.fer.hidden_size, num_classes, 1)

    def forward(self, z_prime: Tensor):
        out = self.classifier(z_prime)
        return out
