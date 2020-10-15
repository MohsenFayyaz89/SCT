from typing import Set

import torch
from torch import Tensor

# noinspection PyPep8Naming
from torch.nn.functional import binary_cross_entropy, pad, one_hot, relu, cross_entropy
from torch.nn.functional import adaptive_max_pool1d


# noinspection PyPep8Naming
from yacs.config import CfgNode

from SCT.datasets.general_dataset import BatchItem
from SCT.models.data_classes import ForwardOut, LossOut


# OldName: OverlapLoss
# noinspection PyPep8Naming
def RegionLoss(A: Tensor, A_hat: Tensor) -> Tensor:
    """

    :param A: [1 x C x K]
    :param A_hat:
    :return:
    """
    indices = A_hat
    A = A.index_select(dim=1, index=indices)  # [1 x |A_hat| x K]
    A = A.permute([0, 2, 1])  # [1 x K x |A_hat|]
    max_classes = adaptive_max_pool1d(input=A, output_size=1).squeeze()  # [K]
    # max_classes = max_classes.view(-1, 1)  # [K x 1]
    target = torch.ones_like(max_classes)
    loss = binary_cross_entropy(max_classes, target)
    return loss


# noinspection PyPep8Naming
def SetLoss(A: Tensor, A_hat: Tensor, weight: Tensor) -> Tensor:
    """

    :param A:
    :param A_hat:
    :param weight:
    :return:
    """
    indices = A_hat
    A = A.index_select(dim=1, index=indices)  # [1 x |A_hat| x K]
    classes = adaptive_max_pool1d(input=A, output_size=1)  # [1 x |A_hat| x 1]
    classes = classes.view(-1, 1)  # [|A_hat| x 1]
    target = classes.new_ones(classes.shape[0])
    weight = weight.index_select(dim=0, index=indices)
    loss = binary_cross_entropy(classes.view(-1), target, weight=weight)
    return loss


# noinspection PyPep8Naming
def TemporalConsistencyLoss(A: Tensor, A_hat: Tensor) -> Tensor:
    """

    :param A:
    :param A_hat:
    :return:
    """
    A = A.index_select(dim=1, index=A_hat)  # [1 x |S| x T']
    shifted_right = pad(A, (1, 0), mode="replicate")
    shifted_left = pad(A, (0, 1), mode="replicate")
    loss = torch.abs(shifted_right - shifted_left).sum()
    return loss


def kl_div(p: Tensor, q: Tensor) -> Tensor:
    """
    args:
    :param p: Tensor same size as q
    :param q: Tensor same size as p
    :returns: kl divergence between the `p` and `q`
    """

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


# noinspection PyPep8Naming
def InverseSparsityLoss(
    A: Tensor, A_hat: Tensor, weight: Tensor, loss_type: str, activation: float = 1.0
) -> Tensor:
    """

    :param A: [1 x C x K]
    :param A_hat:
    :param weight:
    :param loss_type: kl_div or L1
    :param activation:
    :return:
    """
    target_indices = A_hat
    target_A = A.index_select(dim=1, index=target_indices)  # [1 x |A_hat| x K]
    w = weight.index_select(dim=0, index=target_indices)

    if loss_type == "kl_div":
        a = (
            torch.ones(
                size=[1, A_hat.shape[0]],
                dtype=target_A.dtype,
                device=target_A.device,
            )
            * activation
        )
        loss = (kl_div(target_A.mean(dim=2), a) * w).mean()
    elif loss_type == "L1":
        loss = (torch.abs(activation - target_A.mean(dim=2)) * w).mean()

    return loss


# noinspection PyPep8Naming
def un_normalized_length_regularizer(length_loss_width: float, L: Tensor) -> Tensor:
    """
    relu(s - w) + relu(- w - s)
    """
    y_right = relu(L - length_loss_width)
    y_left = relu(-length_loss_width - L)

    return (y_right + y_left).sum()


# noinspection PyPep8Naming
def loss_func(
    A: Tensor,
    L: Tensor,
    V: Tensor,
    A_hat: Tensor,
    cfg: CfgNode,
    weight: Tensor,
) -> LossOut:
    """

    :param A:
    :param L:
    :param V:
    :param A_hat:
    :param cfg:
    :param weight:
    :return:
    """

    # Regularizer over L --------
    length_loss = (
        un_normalized_length_regularizer(length_loss_width=cfg.length_loss_width, L=L)
        * cfg.length_loss_mul
    )

    # Losses over A --------
    set_loss = SetLoss(A=A, A_hat=A_hat, weight=weight) * cfg.set_loss_mul
    temporal_consistency_loss = (
        TemporalConsistencyLoss(A=A, A_hat=A_hat) * cfg.temporal_consistency_loss_mul
    )
    region_loss = RegionLoss(A=A, A_hat=A_hat) * cfg.region_loss_mul
    inv_sparsity_loss = (
        InverseSparsityLoss(
            A=A,
            A_hat=A_hat,
            weight=weight,
            loss_type=cfg.inv_sparsity_loss_type,
            activation=cfg.inv_sparsity_loss_activation,
        )
        * cfg.inv_sparsity_loss_mul
    )

    sct_loss = cross_entropy(V.squeeze(dim=0), A_hat, weight=weight) * cfg.sct_loss_mul

    total_loss = (
        length_loss
        + set_loss
        + temporal_consistency_loss
        + region_loss
        + inv_sparsity_loss
        + sct_loss
    )
    return LossOut(
        total_loss=total_loss,
        length_loss=length_loss,
        set_loss=set_loss,
        temporal_consistency_loss=temporal_consistency_loss,
        region_loss=region_loss,
        inv_sparsity_loss=inv_sparsity_loss,
        sct_loss=sct_loss,
    )
