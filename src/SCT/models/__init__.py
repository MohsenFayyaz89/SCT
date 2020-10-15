import torch
from typing import Dict
from torch import Tensor, nn
from .fu_model import Fu
from yacs.config import CfgNode
from SCT.models.fc_model import Fc
from SCT.models.fl_model import Fl
from SCT.models.fs_model import Fs
from .fer_model import WaveNet, Fer
from .general_model import GeneralModel
from SCT.models.sct import SCT, get_masks


def _make_fer(cfg: CfgNode) -> Fer:
    if cfg.model.fer.name == "wavenet":
        if cfg.model.fer.wavenet_pooling_levels is []:
            pooling_levels = [1, 2, 4, 8, 16]
        else:
            pooling_levels = cfg.model.fer.wavenet_pooling_levels
        return WaveNet(cfg, pooling_levels=pooling_levels)
    else:
        raise Exception("Invalid fer name")


def _make_fs(cfg: CfgNode, num_classes: int) -> Fs:
    if cfg.model.fs.name == "conv":
        return fs_model.Conv(
            cfg,
            num_classes=num_classes,
        )
    else:
        raise Exception("Invalid Fs name")


def _make_fc(cfg: CfgNode, num_classes: int) -> Fc:
    if cfg.model.fc.name == "conv":
        return fc_model.Conv(cfg, num_classes=num_classes)
    else:
        raise Exception("Invalid fc name")


def _make_fl(cfg: CfgNode) -> Fl:
    if cfg.model.fl.name == "conv":
        return fl_model.Conv(cfg)
    else:
        raise Exception("Invalid fl name")


def _make_fu(cfg: CfgNode) -> Fu:
    if cfg.model.fu.name == "TemporalSampling":
        return fu_model.TemporalSamplingUpSampler()
    else:
        raise Exception("Invalid fu name")


def make_loss_weights(num_classes: int, weights: Dict) -> Tensor:
    loss_weights = torch.ones(num_classes)
    for (class_idx, weight) in weights:
        loss_weights[class_idx] = weight
    loss_weights = nn.Parameter(loss_weights, requires_grad=False)
    return loss_weights


def make_model(cfg: CfgNode, num_classes: int) -> GeneralModel:
    fer = _make_fer(cfg)
    fs = _make_fs(cfg, num_classes=num_classes)
    fc = _make_fc(cfg, num_classes=num_classes)
    fl = _make_fl(cfg)
    fu = _make_fu(cfg=cfg)
    sct_func = SCT
    get_masks_func = get_masks
    model = GeneralModel(
        cfg=cfg,
        fer=fer,
        fs=fs,
        fc=fc,
        fl=fl,
        fu=fu,
        sct=sct_func,
        get_masks=get_masks_func,
    )
    return model
