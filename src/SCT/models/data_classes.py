from dataclasses import dataclass
from torch import Tensor


@dataclass
class ForwardOut:
    S: Tensor  # [1 x C x T ]
    Y: Tensor  # [1 x C x T]
    Z: Tensor  # [1 x D' x T']
    A: Tensor  # [1 x C x K]
    L: Tensor  # [1 x 1 x K]
    V: Tensor  # [1 x M x C]

@dataclass
class LossOut:
    total_loss: Tensor  # []  this is going to be used for backpropagation
    set_loss: Tensor  # []
    region_loss: Tensor  # []
    sct_loss: Tensor  # []
    temporal_consistency_loss: Tensor  # []
    length_loss: Tensor  # []
    inv_sparsity_loss: Tensor  # []
