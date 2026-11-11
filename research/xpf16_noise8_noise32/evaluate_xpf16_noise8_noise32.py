"""Exactly three uncertainty methods with one shared online calibration class.

XPF16 and32-noise already shared this policy; this study makes the equality
explicit, freshly calibrates all three, and adds actual8-path uncertainty.
"""

import json
import time

import numpy as np
import torch

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from experiment_xpf16_denoising_views import Provider as OriginalProvider, load
from experiment_reliability_transport_e12 import normalization, PREDICTOR, history, target, issued
from experiment_physical20_augmented_dynamics import new_model
from memory_quantile_online_risk import MemoryQuantileCalibrator
from conditional_online_risk import risk_features
from causal_shape_calibration import initial_state
from experiment_direct_error_uncertainty import metrics
from evaluate_reliability_transport_e12 import ranking
from xpf_crps_metrics import interval_completions


OUT = ROOT / 'runs/xpf16_noise8_noise32_20261111'
REFERENCE = ROOT / 'runs/requested16_28_noise32_20261110'
NAMES = ('XPF16', '8-noise', '32-noise')


class Provider:
    def __init__(self):
        self.norm = normalization()
        self.original = OriginalProvider(self.norm)
        self.original.heads = {'F0': self.original.heads['F0']}
        self.model = new_model('augmented_diffusion').eval().requires_grad_(False)
        self.model.load_state_dict(load(PREDICTOR / 'augmented_diffusion/model.pt')['model'])
        self.paths = None
        self.std8 = None

    @torch.no_grad()
    def bounds(self, split, start):
        point, raw = self.original.bounds(split, start)
        h = (history(split, start) - self.norm['mean']) / self.norm['std']
        assert torch.equal(point, self.model.rollout(h, 500000, sign=0))
        # Prefix of the existing32 seeds; never approximate8-path spread by
        # scaling the32-path std or sampling a fabricated Gaussian ensemble.
        self.paths = torch.stack([self.model.rollout(h, seed, sign=1) * self.norm['std'] + self.norm['mean']
                                  for seed in range(500000, 500008)])
        self.std8 = self.paths.std(dim=0)
        reference32 = issued('diffusion', split, start)['std32']
        radius8 = 1.6448536269514722 * (self.std8 / self.norm['std']).clamp_min(1e-6)
        radius32 = 1.6448536269514722 * (reference32 / self.norm['std']).clamp_min(1e-6)
        return h, point, {'XPF16': raw['F0'], '8-noise': (-radius8, radius8), '32-noise': (-radius32, radius32)}


def event(kind, split, start):
    with (OUT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event=kind, split=split, start=start, time=time.time())) + '\n')


@torch.no_grad()
def main():
    configure()
    OUT.mkdir(exist_ok=False)
    files = [PREDICTOR / 'augmented_diffusion/model.pt', ROOT / 'runs/selected16_quantile_20261110/compact16.pt',
             ROOT / 'memory_quantile_online_risk.py', ROOT / 'conditional_online_risk.py']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    dump(OUT / 'protocol.json', dict(source_sha256=digest(ROOT / 'evaluate_xpf16_noise8_noise32.py'),
        frozen_artifacts=hashes, methods=NAMES, four_call_center='Identical zero-noise P,20history->complete20future.',
        controller='Allthree instantiate the exact SAME MemoryQuantileCalibrator(memory_dynamic, own_initial, batch), samefeaturebuilder and update rules. No method-specific calibrator branches.',
        previous_protocol='XPF16 and32-noise in the prior matched table already shared this controller. No new claimed algorithmic fix; prior uncalibrated native ensemble CRPS was a separately labeled diagnostic.',
        raw_inputs='XPF16 learned05/95 residual bounds;8-noise and32-noise +/-1.64485*actual sample std. XPF16 may have learned residual midpointm. Base pointP is identical, interval centers can differ bym.',
        seeds8=list(range(500000, 500008)), seeds32=list(range(500000, 500032)),
        calibration='Fresh recomputation on same10independent IDs /9windows. Each method own shape/amplitude, beta=0 and bias=0. Reproduce frozen XPF16/32-noise initialstates exactly. No reuse of32 calibration for8.',
        time='Allthree complete20intervals issued before truth. Only complete-window reveal updates state; no neural training.',
        metrics='Gaussian/Laplace CRPS completions match eachissued5/95bounds; sameassumptions for allthree, notnative full distributions of quantilehead. Also report90%coverage/width/intervalscore/AURC and retainoriginal8metrics/96cells.',
        calls='Native deployment XPF16=4backbonecalls+head;8-noise=8*4+4commonP=36;32-noise=32*4+4commonP=132.32moments reused here; no speedup claimed from cache IO.',
        caveat='Reused historicaldevelopment/test, exploratory; no strict conditional or joint guarantee.'))
    provider = Provider()
    ratios = {name: [] for name in NAMES}
    for start in range(0, 161, 20):
        h, point, bounds = provider.bounds('calibration', start)
        event('all_raw_forecasts_issued', 'calibration', start)
        residual = (target('calibration', start) - provider.norm['mean']) / provider.norm['std'] - point
        event('full20_target_revealed', 'calibration', start)
        for name, (lo, hi) in bounds.items():
            ratio = (residual - .5 * (lo + hi)).abs() / (.5 * (hi - lo)).clamp_min(1e-6)
            ratios[name].append(ratio.cpu())
        print('CALIBRATION', start, flush=True)
    initial = {name: initial_state(torch.cat(values).cuda(), 'separable_integral') for name, values in ratios.items()}
    old = load(REFERENCE / 'calibration.pt')
    for name, key in (('XPF16', 'selected16'), ('32-noise', 'noise32')):
        assert all(torch.equal(value, old[key]['separable_integral'][k]) for k, value in initial[name].items())
    torch.save({name: {key: value.cpu() for key, value in state.items()} for name, state in initial.items()}, OUT / 'calibration.pt')
    rows, snapshot = [], {}
    pixels = torch.linspace(0, 40959, 256, device='cuda').long()
    for split in ('development', 'test'):
        records = manifest()['files'][split]
        # The constructor and all issue/reveal calls are identical for all
        # methods. Only their own raw bounds and initial fitted states differ.
        controls = {name: MemoryQuantileCalibrator('memory_dynamic', initial[name], len(records)) for name in NAMES}
        for start in range(0, 761, 20):
            h, point, bounds = provider.bounds(split, start)
            issued_bounds = {name: controls[name].issue(*bounds[name], risk_features(h, point, .5 * (bounds[name][1] - bounds[name][0])))
                             for name in NAMES}
            event('all3_complete_intervals_issued', split, start)
            truth = (target(split, start) - provider.norm['mean']) / provider.norm['std']
            residual = truth - point
            event('full20_target_revealed', split, start)
            if split == 'test' and start == 0:
                snapshot['pixels'] = pixels.cpu().numpy()
                snapshot['residual'] = residual[[0, -1]].flatten(2)[:, :, pixels].cpu().numpy()
                snapshot['samples8_physical'] = provider.paths[:, [0, -1]].flatten(3)[:, :, :, pixels].cpu().numpy()
                snapshot['std8_physical'] = provider.std8[[0, -1]].flatten(2)[:, :, pixels].cpu().numpy()
            mse = (residual * provider.norm['std']).square().mean((2, 3, 4))
            for name, (lo, hi) in issued_bounds.items():
                values = dict(**metrics(lo, hi, residual), **ranking(.5 * (hi - lo), residual.abs()))
                crps = interval_completions(lo, hi, residual)
                for family, score in crps.items():
                    values[family + '_crps'] = score.mean((2, 3, 4))
                    values['raw_' + family + '_crps'] = (score * provider.norm['std']).mean((2, 3, 4))
                for i, record in enumerate(records):
                    for c, channel in enumerate(('u', 'v')):
                        rows.append(dict(kind='diffusion', split=split, start=start,
                            period='confirmation' if start >= 580 else 'main', file=record['file'], group=record['group'],
                            channel=channel, map=name, variant='memory_dynamic', mse=float(mse[i, c]),
                            **{key: float(value[i, c]) for key, value in values.items()}))
                if split == 'test' and start == 0:
                    for label, value in (('lower', lo), ('upper', hi), *crps.items()):
                        snapshot[name + '_' + label] = value[[0, -1]].flatten(2)[:, :, pixels].cpu().numpy()
                controls[name].reveal(residual)
            if split == 'test' and start == 0:
                np.savez_compressed(OUT / 'audit_snapshot.npz', **snapshot)
            if start in (0, 180, 380, 580, 760):
                print('THREE_METHODS', split, start, flush=True)
    dump(OUT / 'per_case.json', rows)
    assert all(digest(ROOT / path) == value for path, value in hashes.items())
    dump(OUT / 'complete.json', dict(passed=True, records=len(rows), only_requested_three_methods=True,
        XPF16_noise32_initial_states_exact=True, same_calibrator_class_and_features=True,
        independent_noise8_calibration=True, no_neural_training=True, frozen_artifacts_unchanged=True))
    print('XPF16_NOISE8_NOISE32_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
