"""Fit ordered interior quantiles while freezing published XPF16 endpoints.

The two endpoint predictions and the existing causal online calibrator are
unchanged. The 17 interior levels are learned jointly from real residuals.
"""
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from torch import nn

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from conditional_online_risk import risk_features
from experiment_direct_error_uncertainty import cached
from experiment_extended_train_error_adapters import LATE_RECORDS
from experiment_fullblock_memory_risk import HEAD_SOURCE
from experiment_reliability_transport_e12 import FIT_RECORDS, geometry, history, normalization, target
from experiment_selected16_quantile import new_head
from experiment_temporal_error_uncertainty import pack
from memory_quantile_online_risk import MemoryQuantileCalibrator
from selected16_quantile_head import BASE_INDICES
from temporal_mean_memory_control import training_data


OUT = ROOT / 'runs/xpf16_quantiles19_20261111'
OLD = ROOT / 'runs/xpf16_noise8_noise32_20261111'
WEIGHTS = ROOT / 'runs/selected16_quantile_20261110/compact16.pt'
STEPS = 2000


def load(path):
    return torch.load(path, map_location='cuda', weights_only=False)


class OrderedInterior(nn.Module):
    """Learn 18 positive gaps spanning the frozen 5% and 95% quantiles."""
    def __init__(self):
        super().__init__()
        self.endpoint = new_head('compact16', None)
        self.endpoint.load_state_dict(load(WEIGHTS)['model'])
        self.endpoint.eval().requires_grad_(False)
        torch.manual_seed(42)
        self.gaps = nn.Sequential(
            nn.Conv3d(16, 24, 3, padding=1),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv3d(24, 24, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv3d(24, 36, 1),
        )
        # Equal spacing at initialization; subsequent data drive gap shapes.
        nn.init.zeros_(self.gaps[-1].weight)
        nn.init.zeros_(self.gaps[-1].bias)

    def forward(self, base, last):
        with torch.no_grad():
            lo, hi = self.endpoint.bounds(base, last, None, None, False)
            batch, _, _, height, width = base.shape
            latent = self.endpoint.feature_projection(last).reshape(batch, 8, 20, height, width)
            y = torch.linspace(-1, 1, height, device=base.device, dtype=base.dtype)
            x = torch.linspace(-1, 1, width, device=base.device, dtype=base.dtype)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack((xx, yy))[None, :, None].expand(batch, 2, 20, height, width)
            features = torch.cat((base[:, list(BASE_INDICES)], latent, coords), dim=1)
        logits = self.gaps(features).reshape(batch, 2, 18, 20, height, width)
        fractions = logits.softmax(dim=2).cumsum(dim=2)[:, :, :17]
        inner = lo.unsqueeze(2) + (hi - lo).unsqueeze(2) * fractions
        return torch.cat((lo.unsqueeze(2), inner, hi.unsqueeze(2)), dim=2)


def event(name, split, start):
    with (OUT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event=name, split=split, start=start, time=time.time())) + '\n')


def fit(norm):
    base, last, history_value, memory, residual = training_data(norm)
    del history_value, memory
    assert len(base) == 819
    head = OrderedInterior().cuda()
    optimizer = torch.optim.AdamW(head.gaps.parameters(), lr=3e-4, weight_decay=1e-5)
    sampler = torch.Generator(device='cuda').manual_seed(42)
    levels = torch.arange(2, 19, device='cuda').reshape(1, 1, 17, 1, 1, 1) / 20
    schedule = hashlib.sha256()
    log = []
    for step in range(1, STEPS + 1):
        ix = torch.randint(819, (4,), device='cuda', generator=sampler)
        available = torch.rand(4, device='cuda', generator=sampler) >= .2
        schedule.update(ix.cpu().numpy().tobytes())
        schedule.update(available.float().cpu().numpy().tobytes())
        optimizer.param_groups[0]['lr'] = 1e-6 + (3e-4 - 1e-6) * .5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
        optimizer.zero_grad(set_to_none=True)
        values = head(base[ix], last[ix])
        error = residual[ix].unsqueeze(2) - values[:, :, 1:-1]
        loss = torch.maximum(levels * error, (levels - 1) * error).mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(head.gaps.parameters(), 1.0))
        optimizer.step()
        if step == 1 or step % 250 == 0:
            log.append(dict(step=step, loss=float(loss), elapsed=time.time()))
            print('QUANTILE_TRAIN', log[-1], flush=True)
    expected = json.loads((ROOT / 'runs/selected16_quantile_20261110/heads_frozen.json').read_text())['same_sampler_as_full28']['compact16']
    assert schedule.hexdigest() == expected
    torch.save(dict(model={key: value.cpu() for key, value in head.gaps.state_dict().items()},
                    endpoint_sha256=digest(WEIGHTS), steps=STEPS, seed=42, pairs=819), OUT / 'interior_quantiles.pt')
    dump(OUT / 'training.json', log)
    dump(OUT / 'frozen.json', dict(endpoint_sha256=digest(WEIGHTS), training_pairs=819, steps=STEPS,
        minibatches_match_original=True, added_trainable_parameters=sum(p.numel() for p in head.gaps.parameters()),
        checkpoint_sha256=digest(OUT / 'interior_quantiles.pt')))
    return head.eval().requires_grad_(False)


@torch.no_grad()
def evaluate(head, norm):
    stats = load(HEAD_SOURCE / 'statistics.pt')['statistics']
    initial = load(OLD / 'calibration.pt')['XPF16']
    oldrows = json.loads((OLD / 'per_case.json').read_text())
    old = {tuple(r[k] for k in ('split', 'file', 'start', 'channel')): r for r in oldrows if r['map'] == 'XPF16'}
    rows = []
    for split in ('development', 'test'):
        records = manifest()['files'][split]
        geo = geometry(records)
        control = MemoryQuantileCalibrator('memory_dynamic', initial, len(records))
        for start in range(0, 761, 20):
            point, base, last = pack(split, start, norm, geo)
            base = ((base - stats['base_mean']) / stats['base_std']).clamp(-10, 10)
            last = ((last - stats['last_mean']) / stats['last_std']).clamp(-10, 10)
            prediction = head(base, last)
            assert torch.all(prediction[:, :, 1:] >= prediction[:, :, :-1])
            lo, hi = prediction[:, :, 0], prediction[:, :, -1]
            h = (history(split, start) - norm['mean']) / norm['std']
            calibrated_lo, calibrated_hi = control.issue(lo, hi, risk_features(h, point, .5 * (hi - lo)))
            midpoint = (lo + hi) * .5
            ratio = (calibrated_hi - calibrated_lo) / (hi - lo).clamp_min(1e-6)
            calibrated = midpoint.unsqueeze(2) + (prediction - midpoint.unsqueeze(2)) * ratio.unsqueeze(2)
            # Preserve issued endpoints exactly despite float32 roundoff.
            calibrated[:, :, 0] = calibrated_lo
            calibrated[:, :, -1] = calibrated_hi
            assert torch.all(calibrated[:, :, 1:] >= calibrated[:, :, :-1] - 1e-7)
            event('all19_complete_quantiles_issued', split, start)
            residual = (target(split, start) - norm['mean']) / norm['std'] - point
            event('complete20_target_revealed', split, start)
            for mode, q in (('raw', prediction), ('online', calibrated)):
                error = residual.unsqueeze(2) - q
                levels = torch.arange(1, 20, device=q.device).reshape(1, 1, 19, 1, 1, 1) / 20
                hit = (error <= 0).float().mean((3, 4, 5)).cpu()
                pinball = torch.maximum(levels * error, (levels - 1) * error).mean((3, 4, 5)).cpu()
                strict_lower = (residual < calibrated_lo).float().mean((2, 3, 4)).cpu()
                for i, record in enumerate(records):
                    for c, channel in enumerate(('u', 'v')):
                        previous = old[(split, record['file'], start, channel)]
                        if mode == 'online':
                            # The old interval reports P(Y < lower); the CDF
                            # reports P(Y <= lower), which differs at ties.
                            assert abs(float(strict_lower[i, c]) - previous['lower_miss']) < 2e-6
                            assert abs(float(hit[i, c, -1]) - (1 - previous['upper_miss'])) < 2e-6
                        for k in range(19):
                            rows.append(dict(split=split, start=start, file=record['file'], group=record['group'],
                                channel=channel, mode=mode, percentile=(k + 1) * 5,
                                empirical_cdf=float(hit[i, c, k]), pinball=float(pinball[i, c, k])))
            control.reveal(residual)
            if start in (0, 180, 380, 580, 760):
                print('QUANTILE_EVAL', split, start, flush=True)
    dump(OUT / 'per_case.json', rows)
    assert len(rows) == 2 * 2 * 39 * 12 * 2 * 19
    groups = defaultdict(list)
    for row in rows:
        if row['start'] >= 20:
            groups[(row['split'], row['group'], row['channel'], row['mode'], row['percentile'])].append(row)
    summary = []
    for split in ('development', 'test'):
        for mode in ('raw', 'online'):
            for percentile in range(5, 100, 5):
                cells = [v for k, v in groups.items() if k[0] == split and k[3:] == (mode, percentile)]
                assert len(cells) == 6
                coverage = mean(mean(x['empirical_cdf'] for x in cell) for cell in cells)
                score = mean(mean(x['pinball'] for x in cell) for cell in cells)
                summary.append(dict(split=split, mode=mode, percentile=percentile,
                    empirical_cdf=coverage, calibration_error_pp=100 * coverage - percentile, pinball=score))
    dump(OUT / 'summary.json', summary)
    with (OUT / 'quantile_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    dump(OUT / 'complete.json', dict(passed=True, rows=len(rows), monotonic_all_windows=True,
        old_online_endpoints_reproduced=True, full_window_causality=True,
        probabilities='5..95 by5;17 new interior levels',
        caveat='Reused development/test trajectories; no fresh independent confirmation.19 quantiles do not uniquely define tail CDF or joint distribution.'))


def main():
    configure()
    if '--evaluate-only' in sys.argv:
        assert (OUT / 'interior_quantiles.pt').exists()
        event_path = OUT / 'events.jsonl'
        if event_path.exists():
            event_path.replace(OUT / 'events_first_attempt.jsonl')
        head = OrderedInterior().cuda()
        head.gaps.load_state_dict(load(OUT / 'interior_quantiles.pt')['model'])
        head.eval().requires_grad_(False)
        evaluate(head, normalization())
        print('QUANTILES19_COMPLETE', flush=True)
        return
    OUT.mkdir(exist_ok=False)
    fit_ids = {r['file'] for r in FIT_RECORDS + LATE_RECORDS}
    held_ids = {r['file'] for split, records in manifest()['files'].items() if split != 'train' for r in records}
    assert not fit_ids & held_ids
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in (WEIGHTS, OLD / 'calibration.pt')}
    dump(OUT / 'protocol.json', dict(source_sha256=digest(Path(__file__)), frozen_artifacts=frozen,
        model='Frozen XPF16 05/95 bounds, plus separately trained16-input positive-gap CNN predicting17 ordered interior quantiles.',
        train='819 real frozen predictor residual windows, training identities only; 2000 updates seed42, AdamW3e-4 cosine, batch4.',
        calibration='Unchanged XPF16 memory_dynamic controller, positive affine mapping applied to every quantile after full-window issue.',
        reporting='Empirical CDF Pr(Y<=Q_tau), standardized pinball, development and test, windows2..39, Re/channel equal weighting.',
        limitations='Prior data reused; empirical marginal statistics only; tails below05 and above95 unspecified.'))
    norm = normalization()
    head = fit(norm)
    evaluate(head, norm)
    assert all(digest(ROOT / path) == value for path, value in frozen.items())
    print('QUANTILES19_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
