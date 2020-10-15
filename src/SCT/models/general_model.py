import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, argmax
from yacs.config import CfgNode

from SCT.models import Fs
from SCT.models.data_classes import ForwardOut
from SCT.models.fc_model import Fc
from SCT.models.fl_model import Fl
from .fu_model import Fu
from .fer_model import Fer
from torch.nn.functional import softmax
from ..datasets.general_dataset import BatchItem


class GeneralModel(nn.Module):
    def __init__(
        self,
        cfg: CfgNode,
        fer: Fer,
        fs: Fs,
        fc: Fc,
        fl: Fl,
        fu: Fu,
        sct,
        get_masks,
    ):
        super().__init__()
        self.cfg = cfg
        self.fer = fer
        self.fs = fs
        self.fc = fc
        self.fl = fl
        self.fu = fu
        self.sct = sct
        self.get_masks = get_masks

    # noinspection PyPep8Naming
    def forward(self, batch: BatchItem) -> ForwardOut:
        T = batch.T
        X = batch.feats.squeeze(dim=1)
        fer_outputs = self.fer.forward(X)  # [[1 x D' x T'], [1 x D', K]]
        Z = fer_outputs[0]  # [1 x D' x T']
        Z_prime = fer_outputs[-1]  # [1 x D' x K]
        S = self.fs(Z, T)  # [1 x C x T]
        A = self.fc(Z_prime)  # [1 x C x K]
        A = softmax(A / self.cfg.model.fc.softmax_temp, dim=1)
        L = self.fl(Z_prime)  # [1 x 1 x K]
        Y = self.fu(A=A, L=L, T=T)  # [1 x C x T]
        W = self.get_masks(Y=Y, A_hat=batch.A_hat)  # [1 x M x T]
        V = self.sct(W=W, S=S, T=T)  # [1 x M x C]
        # _res = F.interpolate(A, T)
        # Y = _res
        return ForwardOut(S=S, Y=Y, Z=Z, A=A, L=L, V=V)
