"""Streaming full-field CDFs on one fixed grid in dataset physical units.

Quantile paths are monotonically rearranged per location. Between adjacent
quantiles the inverse CDF is linear; the two missing tails are endpoint atoms.
Ensemble CDFs use actual calibrated samples without a parametric approximation.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import torch

METHODS = ('xpf16_1', 'xpf16_02', 'noise8', 'noise32', 'noise256', 'noise2048')


def _add_at(acc, indices, values):
    acc.scatter_add_(0, indices.reshape(-1), values.reshape(-1).to(torch.float64))


def quantile_cdf_sum(q, probs, grid, chunk=8192):
    """Sum per-location CDFs, using a piecewise-linear quantile function.

    q has shape [levels, observations] and contains forecast minus truth.
    Probability mass outside the trained grid becomes endpoint point masses.
    The slope/intercept scan evaluates every grid point exactly, without
    materializing [observations, grid points].
    """
    if q.ndim != 2 or len(q) != len(probs):
        raise ValueError('Quantile and probability dimensions do not match.')
    n = len(grid)
    slope = torch.zeros(n + 1, dtype=torch.float64, device=q.device)
    intercept = torch.zeros_like(slope)
    probs = probs.to(device=q.device, dtype=torch.float64)
    if not torch.all(probs[1:] > probs[:-1]):
        raise ValueError('Probability levels must be strictly increasing.')
    for start in range(0, q.shape[1], chunk):
        values = q[:, start:start + chunk].to(torch.float64).sort(dim=0).values
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite XPF quantile.')
        left, right = values[:-1], values[1:]
        weights = (probs[1:] - probs[:-1])[:, None].expand_as(left)
        equal = right == left
        width = (right - left).clamp_min(1e-300)
        weight_slope = torch.where(equal, torch.zeros_like(width), weights / width)
        a = torch.bucketize(left.contiguous(), grid)
        b = torch.bucketize(right.contiguous(), grid)
        _add_at(slope, a, weight_slope)
        _add_at(slope, b, -weight_slope)
        _add_at(intercept, a, -weight_slope * left + weights * equal)
        _add_at(intercept, b, weight_slope * right)
        ones = torch.ones(values.shape[1], device=q.device, dtype=torch.float64)
        _add_at(intercept, torch.bucketize(values[0].contiguous(), grid), ones * probs[0])
        _add_at(intercept, torch.bucketize(values[-1].contiguous(), grid), ones * (1 - probs[-1]))
    return slope[:-1].cumsum(0) * grid + intercept[:-1].cumsum(0)


def ensemble_cdf_sum(samples, grid, chunk=16):
    """Sum empirical CDFs over locations, equally weighting actual paths."""
    counts = torch.zeros(len(grid) + 1, dtype=torch.float64, device=samples.device)
    for start in range(0, len(samples), chunk):
        block = samples[start:start + chunk]
        if not torch.isfinite(block).all():
            raise ValueError('Nonfinite calibrated ensemble sample.')
        indices = torch.bucketize(block.to(torch.float64).contiguous().flatten(), grid)
        counts += torch.bincount(indices, minlength=len(grid) + 1).to(torch.float64)
    return counts[:-1].cumsum(0) / len(samples)


class CDFAccumulator:
    def __init__(self, device, maximum=0.35, points=701):
        if points % 2 != 1:
            raise ValueError('A symmetric grid must contain zero.')
        self.grid = torch.linspace(-maximum, maximum, points, dtype=torch.float64, device=device)
        self.sums = torch.zeros(6, 2, points, dtype=torch.float64, device=device)
        self.count = 0
        self.starts = []

    def add(self, start, qx, paths, point, residual, scales, std):
        if start == 0:
            return  # first prediction is issued and revealed only to initialize memory
        if start in self.starts:
            raise ValueError('Duplicate forecast window.')
        # Drop the independently trained 50% head: the requested CDFs use 98/498 points.
        ids02 = [i for i in range(499) if i != 249]
        ids1 = [i for i in ids02 if (i + 1) % 5 == 0]
        probs02 = torch.tensor([(i + 1) / 500 for i in ids02], device=qx.device, dtype=torch.float64)
        probs1 = torch.tensor([(i + 1) / 500 for i in ids1], device=qx.device, dtype=torch.float64)
        assert len(ids02) == 498 and len(ids1) == 98
        for c in range(2):
            factor = std[0, c, 0, 0, 0].double()
            truth_r = residual[:, c].double()
            for method, ids, probs in ((0, ids1, probs1), (1, ids02, probs02)):
                q = ((qx[ids, :, c].double() - truth_r.unsqueeze(0)) * factor).flatten(1)
                self.sums[method, c] += quantile_cdf_sum(q, probs, self.grid)
                del q
            for method, count in enumerate((8, 32, 256, 2048), start=2):
                # The same causal 90%-proxy mechanism was fitted separately for each ensemble size.
                scale = scales[count][:, c].double()
                field = point[:, c].double()
                local = torch.zeros_like(self.grid)
                # Chunk over samples; output is divided by the full ensemble size.
                for j in range(0, count, 16):
                    sample = ((paths[j:min(j + 16, count), :, c].double() - field.unsqueeze(0)) * scale.unsqueeze(0)
                              - truth_r.unsqueeze(0)) * factor
                    local += ensemble_cdf_sum(sample, self.grid) * len(sample) / count
                self.sums[method, c] += local
        self.count += residual[:, 0].numel()
        self.starts.append(int(start))

    def save(self, directory, records, norm):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if not self.count or self.starts != list(range(20, 821, 20)):
            raise RuntimeError('Expected all 41 post-cold-start windows.')
        x = self.grid.cpu().numpy()
        values = (self.sums / self.count).cpu().numpy()
        if not np.isfinite(values).all() or (np.diff(values, axis=-1) < -1e-7).any():
            raise RuntimeError('Invalid averaged CDF.')
        config = dict(methods=METHODS, starts=self.starts, cold_start_excluded=0,
                      number_of_trajectories=len(records), trajectory_records=records,
                      observations_per_channel=self.count, physical_scale='dataset original scale after reversing training normalization; physical unit not specified',
                      horizontal_axis='forecast value z minus matching observed truth y',
                      fixed_grid=dict(minimum=-.35, maximum=.35, points=701, shared_across_methods_and_channels=True),
                      quantile_method='per-location monotone rearrangement; linear quantile interpolation; endpoint atoms outside trained probabilities',
                      ensemble_method='empirical CDF of actual paths after per-size causal online 90% proxy scaling',
                      calibration='independent per-trajectory states; updates only after complete 20-frame truth is revealed',
                      original_normalization_mean=norm['mean'].flatten().cpu().tolist(),
                      original_normalization_std=norm['std'].flatten().cpu().tolist(),
                      accumulator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      caveat='Previously used development and test trajectories; no formal coverage guarantee.')
        (directory / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
        for channel, index in (('u', 0), ('v', 1)):
            curves = {name: values[i, index] for i, name in enumerate(METHODS)}
            np.savez_compressed(directory / f'cdf_{channel}.npz', x=x, count=np.int64(self.count), **curves)
            with (directory / f'cdf_{channel}.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['x', *METHODS, 'observations'])
                for i, coordinate in enumerate(x):
                    writer.writerow([coordinate, *[curves[name][i] for name in METHODS], self.count])
