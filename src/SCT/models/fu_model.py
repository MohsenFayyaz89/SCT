from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# noinspection PyPep8Naming
def project_lengths_softmax(T: int, L: Tensor) -> Tensor:
    """

    :param T: 1:int
    :param L: [1 x T']:float
    :return: [1 x T']:float
    """
    return T * F.softmax(L, dim=0)


class Fu(nn.Module):
    """
    Abstract class for the fu(A,L): upsamples the action probabilities A w.r.t.
    estimated actions' temporal lengths L
    """

    # noinspection PyPep8Naming
    def forward(self, A: Tensor, L: Tensor, T: int) -> Tensor:
        raise NotImplementedError


class TemporalSamplingUpSampler(Fu):
    def __init__(self):
        super().__init__()
        self.temp_width = 100

    # noinspection PyPep8Naming
    @staticmethod
    def _normalize_location(T: int, pis: Tensor, sis: Tensor) -> Tensor:
        """
        Normalizes the absolute value of z_where to the range that is appropriate for the network.
        :param T:
        :param pis:
        :param sis: unnormalized z_size
        :return:
        """
        x = pis.clone()
        x += sis / 2
        x -= T / 2
        x /= -(sis / 2)

        return x

    @staticmethod
    def _create_params_matrix(sis: Tensor, pis: Tensor) -> Tensor:
        n = sis.size(0)
        theta = sis.new_zeros(torch.Size([n, 3]))

        s = sis.clone()
        x = pis.clone()
        # y = 0

        theta[:, 0] = s.view(-1)
        theta[:, 1] = x.view(-1)
        theta[:, 2] = 0
        return theta.float()

    @staticmethod
    def _create_theta(params: Tensor) -> Tensor:
        # Takes 3-dimensional vectors, and massages them into 2x3 matrices with elements like so:
        # [s,x,y] -> [[s,0,x],
        #             [0,s,y]]
        n = params.size(0)
        expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3]).to(params.device)
        out = torch.cat((params.new_zeros([1, 1]).expand(n, 1), params), 1)
        return torch.index_select(out, 1, expansion_indices).view(n, 2, 3)

    # noinspection PyPep8Naming
    @staticmethod
    def _normalize_scale(T: int, sis: Tensor) -> Tensor:
        return T / sis

    # noinspection PyPep8Naming
    def forward(self, A: Tensor, L: Tensor, T: int) -> Tensor:
        """
        Given a set of predicted actions probabilities A_{i}s, upsamples them w.r.t. the given projected L_{i}s.
        :param L: [K] The projected lengths.
        :param A: [K x C] The predicted actions' probabilities.
        :return: [1 x C x ~T] Upsampled A_{k}s.
        """

        A = A.squeeze().permute(1, 0)  # [K x C]
        L = L.squeeze()  # [K]
        L_prime = project_lengths_softmax(T=T, L=L)
        K = A.shape[0]
        C = A.shape[1]
        l_max = int(L_prime.max() + 0.5)  # round to the nearest int
        pis = torch.zeros_like(L_prime)  # [K]

        normalized_l = self._normalize_scale(l_max, L_prime)
        normalized_p = self._normalize_location(l_max, pis, L_prime)

        params_mat = self._create_params_matrix(normalized_l, normalized_p)  # [K x 3]
        theta = self._create_theta(params_mat)  # [K x 2 x 3]

        grid = F.affine_grid(theta, torch.Size((K, C, 1, l_max)))

        temp_A = A.view(K, C, 1, 1).expand(-1, -1, -1, self.temp_width)
        upsampled_probs = F.grid_sample(temp_A, grid, mode="bilinear")
        upsampled_probs = upsampled_probs.view(K, C, l_max)  # [K x C x l_max]
        upsampled_cropped = []
        for i, prob in enumerate(upsampled_probs):
            prob_cropped = prob[:, 0 : round(L_prime[i].item())]
            upsampled_cropped.append(prob_cropped)

        out = torch.cat(upsampled_cropped, dim=1).unsqueeze(dim=0)  # [1 x C x ~T]
        out = F.interpolate(input=out, size=T)  # [1 x C x T]
        return out  # [1 x C x T]
