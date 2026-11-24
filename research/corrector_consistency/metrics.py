"""Window diagnostics and paired original-trajectory comparisons."""
from collections import defaultdict
from statistics import mean

import numpy as np
import torch
from torch.nn import functional as F

from .model import spatial, temporal


def local_gradient(field):
    dy = F.pad(field.diff(dim=3).square().sum(1), (0, 0, 0, 1))
    dx = F.pad(field.diff(dim=4).square().sum(1), (0, 1))
    return (dy + dx).sqrt()


def spatial_energy(field):
    return sum(field.diff(dim=d).square().flatten(1).mean(1) for d in (3, 4)) / 2


def region_masks(truth, thresholds):
    return (local_gradient(truth) >= thresholds["gradient_q90"],
            torch.linalg.vector_norm(truth, dim=1) >= thresholds["amplitude_q90"])


def case_metrics(p, y, correction, units, scale, thresholds):
    p, y, correction = p.double(), y.double(), correction.double()
    delta = y - p
    q = p + correction
    err = y - q
    dims = (1, 2, 3, 4)
    sq, mse = err.square(), err.square().mean(dims)
    base = delta.square().mean(dims)
    true_norm = base.sqrt()
    pred_norm = correction.square().mean(dims).sqrt()
    inner = (correction * delta).mean(dims)
    torch.testing.assert_close(mse - base, pred_norm.square() - 2 * inner, atol=1e-14, rtol=0)
    rr, rt = correction / scale, delta / scale
    measures = spatial(rr, rt)
    truth_measures = spatial(rt, rt)
    d1, d2 = temporal(rr)
    t1, t2 = temporal(rt)

    def energy(z):
        return z.square().mean(dims)

    def masked(mask):
        count = mask.sum((1, 2, 3))
        return (sq.mean(1) * mask).sum((1, 2, 3)) / count.clamp_min(1), count

    grad_mask, high_mask = region_masks(rt, thresholds)
    grad_mse, grad_count = masked(grad_mask)
    high_mse, high_count = masked(high_mask)
    fields = dict(mse=mse, mae=err.abs().mean(dims), standardized_mse=(err / units).square().mean(dims),
                  mse_u=sq[:, 0].mean((1, 2, 3)), mse_v=sq[:, 1].mean((1, 2, 3)),
                  correction_cosine=inner / (true_norm * pred_norm).clamp_min(1e-30),
                  correction_inner=inner, rms_ratio=pred_norm / true_norm.clamp_min(1e-30),
                  correction_energy=pred_norm.square(), before_mse=base, after_rms=mse.sqrt(),
                  identity_error=(mse - base - pred_norm.square() + 2 * inner).abs(),
                  d1_energy=energy(d1), d2_energy=energy(d2), d1_match=energy(d1 - t1),
                  d2_match=energy(d2 - t2), spatial_energy=spatial_energy(rr),
                  spatial_match=spatial_energy(rr - rt), true_spatial_energy=spatial_energy(rt),
                  true_d1_energy=energy(t1), true_d2_energy=energy(t2),
                  true_d1_match=energy(t1 - t1), true_d2_match=energy(t2 - t2),
                  true_spatial_match=spatial_energy(rt - rt),
                  high_gradient_mse=grad_mse, high_gradient_points=grad_count,
                  high_residual_mse=high_mse, high_residual_points=high_count)
    fields.update({**measures, **{f"true_{k}": v for k, v in truth_measures.items()}})
    leads = sq.mean((1, 3, 4))
    for start in range(0, 20, 5):
        fields[f"mse_{start+1}_{start+5}"] = leads[:, start:start+5].mean(1)
    fields.update({f"lead_{k+1:02d}": leads[:, k] for k in range(20)})
    rows = {k: v.cpu().tolist() for k, v in fields.items()}
    return [dict(zip(rows, vals)) for vals in zip(*rows.values())]


def aggregate(windows, phase, arm):
    groups = defaultdict(list)
    for row in windows:
        groups[row["trajectory"]].append(row)
    measures = [key for key in windows[0] if key not in ("trajectory", "re", "start", "phase", "arm")]
    rows = []
    for name, items in sorted(groups.items()):
        assert len(items) == 2 and {r["start"] for r in items} == {400, 420}
        rows.append(dict(phase=phase, arm=arm, trajectory=name, re=items[0]["re"],
                         **{m: mean(r[m] for r in items) for m in measures}))
    result = dict(phase=phase, arm=arm, trajectories=len(rows),
                  **{m: mean(r[m] for r in rows) for m in measures})
    return rows, result


def comparisons(trajectories):
    groups = defaultdict(dict)
    for row in trajectories:
        groups[row["phase"], row["arm"]][row["trajectory"]] = row
    metrics = ("mse", "mae", "standardized_mse", "mse_u", "mse_v", "mse_1_5",
               "mse_6_10", "mse_11_15", "mse_16_20", "high_gradient_mse",
               "high_residual_mse", "cosine", "magnitude", "d1_energy", "d2_energy",
               "d1_match", "d2_match", "spatial_match", "correction_cosine", "rms_ratio")
    out = []
    for (phase, arm), values in groups.items():
        if arm == "Native":
            continue
        controls = {
            "A": ("Native",),
            "E": ("Native", "A", "D"),
            "F": ("Native", "A", "D", "E"),
            "G": ("Native", "A", "B", "C", "D", "E", "F"),
        }.get(arm, ("Native", "A"))
        for base in controls:
            other = groups[phase, base]
            keys = sorted(values)
            assert set(keys) == set(other)
            draw = np.random.default_rng(42).integers(len(keys), size=(10000, len(keys)))
            for metric in metrics:
                x = np.array([values[k][metric] for k in keys])
                y = np.array([other[k][metric] for k in keys])
                diff = x - y
                ci = np.quantile(diff[draw].mean(1), (.025, .975))
                pct, lo, hi = "", "", ""
                if (y > 0).all():
                    pct = 100 * (1 - x.mean() / y.mean())
                    lo, hi = np.quantile(100 * (1 - x[draw].mean(1) / y[draw].mean(1)), (.025, .975))
                out.append(dict(phase=phase, arm=arm, baseline=base, metric=metric,
                                delta=diff.mean(), lower95=ci[0], upper95=ci[1], improvement_pct=pct,
                                improvement_lower95_pct=lo, improvement_upper95_pct=hi,
                                lower_trajectories=int((diff < 0).sum()), trajectories=len(keys)))
    return out
