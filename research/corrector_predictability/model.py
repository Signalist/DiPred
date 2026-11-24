"""Capacity controls and target-masked residual direction objectives."""
import torch
from torch import nn

from research.corrector_consistency.model import (COS_FLOOR, MASK_THRESHOLD, PointCorrector,
                                                   neighbors)

FEATURE_GROUPS = ("XP-zero", "XPF", "XP-active-control")
DIRECTION_ARMS = ("base", "smooth", "match", "align")
PROXY_INDICES = tuple(i * 39 // 23 for i in range(24))


class FeatureCorrector(PointCorrector):
    def __init__(self):
        super().__init__()
        self.net[0] = nn.Conv2d(128, 16, 1)

    def forward(self, x, p, f):
        assert x.shape == p.shape and f.shape == (len(x), 48, *x.shape[-2:])
        fields = torch.cat((x.flatten(1, 2), p.flatten(1, 2), f), dim=1)
        return self.net(fields.contiguous(memory_format=torch.channels_last)).reshape_as(p)


def inputs(x, p, mean, units, mode, feature=None, stats=None):
    x, p = (x - mean) / units, (p - mean) / units
    if mode in ("Small", "Expanded", "XP"):
        return x, p
    if mode == "XP-zero":
        f = x.new_zeros((len(x), 48, *x.shape[-2:]))
    elif mode == "XP-active-control":
        f = torch.cat((x.flatten(1, 2)[:, PROXY_INDICES],
                       p.flatten(1, 2)[:, PROXY_INDICES]), dim=1)
    else:
        assert mode == "XPF" and feature is not None
        f = (feature - stats[0]) / stats[1]
    return x, p, f


def normalized_vectors(field):
    return field / torch.linalg.vector_norm(field, dim=1, keepdim=True).clamp_min(COS_FLOOR)


def masked_average(values, valid):
    return ((values * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1))


def direction_cases(pred, truth, arm):
    eligible = torch.linalg.vector_norm(truth, dim=1) > MASK_THRESHOLD
    pred_unit, true_unit = normalized_vectors(pred), normalized_vectors(truth)
    if arm == "align":
        return masked_average(1 - (pred_unit * true_unit).sum(1).clamp(-1, 1), eligible)
    totals = pred.new_zeros(len(pred))
    counts = torch.zeros(len(pred), dtype=torch.long, device=pred.device)
    for (a, b), (c, d), (v, w) in zip(neighbors(pred_unit), neighbors(true_unit), neighbors(eligible)):
        dot = (a*b).sum(1).clamp(-1, 1)
        measure = 1 - dot if arm == "smooth" else (dot - (c*d).sum(1).clamp(-1, 1)).square()
        mask = (v & w).flatten(1)
        totals = totals + (measure.flatten(1)*mask).sum(1)
        counts = counts + mask.sum(1)
    return totals / counts.clamp_min(1)


def direction(pred, truth, arm):
    return direction_cases(pred, truth, arm).mean()


def components(pred, truth, arm):
    mse = (pred-truth).square().mean()
    extra = mse.new_zeros(()) if arm == "base" else direction(pred, truth, arm)
    return mse, extra
