from torch import Tensor


# noinspection PyPep8Naming
def get_masks(Y: Tensor, A_hat: Tensor) -> Tensor:
    """

    :param Y: upsampled predicted probabilities [1 x C x T]
    :param A_hat: A^ set of actions in the video [M] each item between 0 and C-1
    :return: W set of temporal masks [1 x M x T]
    """
    # indices = torch.Tensor(list(A_hat)).to(device=Y.device).long()
    W = Y.index_select(dim=1, index=A_hat)  # [1 x M x T]
    return W


# noinspection PyPep8Naming
def SCT(W: Tensor, S: Tensor, T: int) -> Tensor:
    """
    Set Constrained Temporal Transformation
    :param W: masks [1 x M x T]
    :param S: S intermediate temporal representation over input video X [1 x C x T]
    :param T: T input video temporal length [1]
    :return: V set of predicted actions [1 x M x C]
    """
    S = S.permute(0, 2, 1)  # [1 x T x C]
    V = (W @ S) / T
    return V
