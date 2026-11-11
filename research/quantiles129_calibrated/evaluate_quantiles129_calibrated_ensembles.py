"""Replay 129 XPF quantiles and calibrated actual 8/32-path ensembles."""
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from conditional_online_risk import risk_features
from experiment_direct_error_uncertainty import metrics
from experiment_reliability_transport_e12 import geometry, history, issued, normalization, target
from export_aligned_test_distribution import STARTS, from_hdf
from memory_quantile_online_risk import MemoryQuantileCalibrator
from train_xpf_quantiles129 import OUT
from xpf_crps_metrics import ensemble_crps
from xpf_quantiles129_head import Quantiles129Head, quantile_levels

sys.path.insert(0, str(ROOT / 'github/DiPred'))
from reliability.quantiles19_predict import Quantiles19Predictor


OLD = ROOT / 'runs/xpf16_noise8_noise32_20261111'


def event(kind, split, start):
    with (OUT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event=kind, split=split, history_start=start, time=time.time())) + '\n')


def proxy_quantiles(samples):
    """Linearly interpolated empirical 5/95 quantiles, not Gaussian proxy."""
    sorted_samples = samples.sort(dim=0).values
    def at(level):
        index = level * (len(samples) - 1)
        lo = int(index)
        hi = min(lo + 1, len(samples) - 1)
        return sorted_samples[lo] + (index - lo) * (sorted_samples[hi] - sorted_samples[lo])
    return at(.05), at(.95)


def approximate_crps(pinball, observation, first_q, last_q, levels):
    """Trapezoidal CRPS for piecewise-linear quantiles and clamped endpoints.

    Clamp Q(tau) at Q(.0025)/Q(.9975) outside those levels; this declares
    0.25% point masses at both extremes. Interior pinball integration is
    numerical trapezoidal quadrature, not exact analytic integration.
    """
    central = torch.trapz(pinball, levels, dim=2)
    a = float(levels[0])
    e0 = observation - first_q
    e1 = observation - last_q
    left = (.5 * a * a * e0.clamp_min(0) + (.5 * a * a - a) * e0.clamp_max(0)).mean((2, 3, 4))
    right = ((a - .5 * a * a) * e1.clamp_min(0) - .5 * a * a * e1.clamp_max(0)).mean((2, 3, 4))
    return 2 * (central + left + right)


@torch.no_grad()
def evaluate():
    configure()
    assert json.loads((OUT / 'training_complete.json').read_text())['passed']
    norm = normalization()
    model = Quantiles129Head().cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(OUT / 'quantiles129.pt', map_location='cuda', weights_only=False)['model'])
    initial = torch.load(OLD / 'calibration.pt', map_location='cuda', weights_only=False)
    original = json.loads((OLD / 'per_case.json').read_text())
    original_lookup = {tuple(r[k] for k in ('split', 'file', 'start', 'channel', 'map')): r for r in original}
    oldq = json.loads((ROOT / 'runs/xpf16_quantiles19_20261111/per_case.json').read_text())
    q_lookup = {tuple(r[k] for k in ('split', 'file', 'start', 'channel', 'mode', 'percentile')): r for r in oldq}
    taus = torch.tensor(quantile_levels(), dtype=torch.float32, device='cuda')
    assert len(taus) == 129
    qrows, erows, snapshots = [], [], {}
    pixels = torch.linspace(0, 40959, 256, device='cuda').long()
    for split in ('development', 'test'):
        records = manifest()['files'][split]
        predictor = Quantiles19Predictor(geometry(records), 'cuda')
        controls = {name: MemoryQuantileCalibrator('memory_dynamic', initial[name], len(records))
                    for name in ('8-noise', '32-noise')}
        features = []
        hook = predictor.head.gaps.register_forward_pre_hook(lambda module, args: features.append(args[0]))
        for start in STARTS:
            observed = history(split, start) if start <= 760 else from_hdf(records, start)
            features.clear()
            result = predictor.predict(observed)
            assert len(features) == 1 and predictor.last_call_labels == [99, 66, 33, 0]
            point = predictor.pending
            raw19 = predictor.head.last_quantiles
            raw129 = model(features[0], raw19)
            assert torch.equal(raw129[:, :, 19:110:5], raw19)
            assert torch.all(raw129[:, :, 1:] >= raw129[:, :, :-1])
            xpf_lo, xpf_hi = predictor.calibrator.pending[2:4]
            raw_lo, raw_hi = raw19[:, :, 0], raw19[:, :, -1]
            midpoint = .5 * (raw_lo + raw_hi)
            xpf_scale = (xpf_hi - xpf_lo) / (raw_hi - raw_lo).clamp_min(1e-6)
            adjusted129 = midpoint.unsqueeze(2) + (raw129 - midpoint.unsqueeze(2)) * xpf_scale.unsqueeze(2)
            adjusted129[:, :, 19] = xpf_lo
            adjusted129[:, :, 109] = xpf_hi
            assert torch.all(adjusted129[:, :, 1:] >= adjusted129[:, :, :-1] - 1e-7)
            h = (observed - norm['mean']) / norm['std']
            physical = torch.stack([predictor.model.rollout(h, seed, sign=1) * norm['std'] + norm['mean']
                                    for seed in range(500000, 500032)])
            if start <= 760:
                ref = issued('diffusion', split, start)
                assert torch.equal(physical.mean(0), ref['mean']) and torch.equal(physical.std(0), ref['std32'])
            samples = (physical - norm['mean']) / norm['std']
            raw_intervals, adjusted_samples, bounds = {}, {}, {}
            for name, count in (('8-noise', 8), ('32-noise', 32)):
                radius = 1.6448536269514722 * (physical[:count].std(0) / norm['std']).clamp_min(1e-6)
                raw_intervals[name] = (-radius, radius)
                lo, hi = controls[name].issue(-radius, radius, risk_features(h, point, radius))
                bounds[name] = (lo, hi)
                scale = (hi - lo) / (2 * radius).clamp_min(1e-6)
                # Apply the same positive online scale around the point P to
                # the actual paths. Ensemble mean need not equal P.
                adjusted_samples[name] = point.unsqueeze(0) + (samples[:count] - point.unsqueeze(0)) * scale.unsqueeze(0)
            event('all_calibrated_forecasts_issued', split, start)
            truth = target(split, start) if start <= 760 else from_hdf(records, start + 20)
            truth_norm = (truth - norm['mean']) / norm['std']
            residual = truth_norm - point
            event('full20_target_revealed', split, start)
            if split == 'test' and start == 0:
                snapshots['pixels'] = pixels.cpu().numpy()
                snapshots['truth_normalized'] = truth_norm[[0, -1]].flatten(2)[:, :, pixels].cpu().numpy()
                snapshots['point_normalized'] = point[[0, -1]].flatten(2)[:, :, pixels].cpu().numpy()
                snapshots['quantiles129_online'] = adjusted129[[0, -1]].flatten(3)[:, :, :, pixels].cpu().numpy()
                snapshots['quantiles129_raw'] = raw129[[0, -1]].flatten(3)[:, :, :, pixels].cpu().numpy()
                snapshots['raw_samples32'] = samples[:, [0, -1]].flatten(3)[:, :, :, pixels].cpu().numpy()
                for name, values in adjusted_samples.items():
                    snapshots[name + '_calibrated_samples'] = values[:, [0, -1]].flatten(3)[:, :, :, pixels].cpu().numpy()
                    snapshots[name + '_raw_radius'] = raw_intervals[name][1][[0, -1]].flatten(2)[:, :, pixels].cpu().numpy()
                    snapshots[name + '_online_lo'] = bounds[name][0][[0, -1]].flatten(2)[:, :, pixels].cpu().numpy()
                    snapshots[name + '_online_hi'] = bounds[name][1][[0, -1]].flatten(2)[:, :, pixels].cpu().numpy()
            for mode, values in (('raw', raw129), ('online', adjusted129)):
                error = residual.unsqueeze(2) - values
                levels = taus.reshape(1, 1, 129, 1, 1, 1)
                point_pin = torch.maximum(levels * error, (levels - 1) * error)
                pinball = point_pin.mean((3, 4, 5))
                coverage = (error <= 0).float().mean((3, 4, 5))
                crps = approximate_crps(pinball, residual, values[:, :, 0], values[:, :, -1], taus)
                for i, rec in enumerate(records):
                    for c, channel in enumerate(('u', 'v')):
                        if start <= 760:
                            for k, percentile in enumerate(range(5, 100, 5)):
                                old = q_lookup[split, rec['file'], start, channel, mode, percentile]
                                pos = 19 + 5 * k
                                assert abs(float(pinball[i, c, pos]) - old['pinball']) < 2e-6
                                assert abs(float(coverage[i, c, pos]) - old['empirical_cdf']) <= 1 / 40960 + 1e-7
                        qrows.append(dict(split=split, start=start, file=rec['file'], group=rec['group'],
                            channel=channel, mode=mode, coverage=coverage[i, c].cpu().tolist(),
                            pinball=pinball[i, c].cpu().tolist(), approx_crps=float(crps[i, c])))
            for name, count in (('8-noise', 8), ('32-noise', 32)):
                lo, hi = bounds[name]
                interval_metrics = metrics(lo, hi, residual)
                for mode, values in (('raw', samples[:count]), ('online', adjusted_samples[name])):
                    score = ensemble_crps(values, truth_norm)
                    ens_lo, ens_hi = proxy_quantiles(values)
                    empirical_coverage = ((truth_norm >= ens_lo) & (truth_norm <= ens_hi)).float().mean((2, 3, 4))
                    a = {k: v.mean((2, 3, 4)).cpu() for k, v in score.items()}
                    for i, rec in enumerate(records):
                        for c, channel in enumerate(('u', 'v')):
                            if start <= 760 and mode == 'online':
                                old = original_lookup[split, rec['file'], start, channel, name]
                                for metric in ('coverage', 'standardized_width', 'standardized_interval_score'):
                                    assert abs(float(interval_metrics[metric][i, c]) - old[metric]) < 2e-6
                            erows.append(dict(split=split, start=start, file=rec['file'], group=rec['group'],
                                channel=channel, method=name, mode=mode, empirical_crps=float(a['empirical'][i, c]),
                                fair_crps=float(a['fair'][i, c]), empirical_interval_coverage=float(empirical_coverage[i, c]),
                                proxy_interval_coverage=float(interval_metrics['coverage'][i, c]) if mode == 'online' else None,
                                proxy_interval_width=float(interval_metrics['standardized_width'][i, c]) if mode == 'online' else None,
                                proxy_interval_score=float(interval_metrics['standardized_interval_score'][i, c]) if mode == 'online' else None))
                controls[name].reveal(residual)
            predictor.reveal(truth)
            if split == 'test' and start == 0:
                np.savez_compressed(OUT / 'snapshot.npz', **snapshots)
            if start in (0, 180, 380, 580, 760, 820):
                print('EVALUATE129', split, start, flush=True)
        hook.remove()
    dump(OUT / 'quantile_per_case.json', qrows)
    dump(OUT / 'ensemble_per_case.json', erows)
    assert len(qrows) == 2 * 42 * 12 * 2 * 2
    assert len(erows) == 2 * 42 * 12 * 2 * 2 * 2
    dump(OUT / 'evaluation_complete.json', dict(passed=True, levels=quantile_levels(),
        all_forecasts_issued_before_truth=True, feedback_only_after_complete20=True,
        prior39_19_anchors_reproduced=True, prior39_noise_proxy_intervals_reproduced=True,
        prior39_sample32_moments_reproduced=True, windows_per_trajectory=42,
        calibrated_sample_definition='P + (sample-P) * (calibrated_radius / raw_gaussian_proxy_radius)',
        caveat='Marginal calibration and reused historical test. Calibrated sample empirical 90% coverage need not match Gaussian proxy coverage. Quantile CRPS has numerical interpolation and explicit clamped outer tail assumptions.'))
    print('EVALUATION129_COMPLETE', flush=True)


if __name__ == '__main__':
    evaluate()
