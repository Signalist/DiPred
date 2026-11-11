"""Evaluate actual 8/32-path marginal ensemble CRPS with reusable terms."""
import json
import time
from collections import defaultdict
from statistics import mean

import numpy as np
import torch

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from experiment_physical20_augmented_dynamics import new_model
from experiment_reliability_transport_e12 import PREDICTOR, history, issued, normalization, target
from xpf_crps_metrics import ensemble_crps


OUT = ROOT / 'runs/native_ensemble8_32_crps_20261111'


@torch.no_grad()
def main():
    configure()
    OUT.mkdir(exist_ok=False)
    norm = normalization()
    model_path = PREDICTOR / 'augmented_diffusion/model.pt'
    model = new_model('augmented_diffusion').eval().requires_grad_(False)
    model.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=False)['model'])
    old = json.loads((ROOT / 'runs/xpf16_crps_20261111/per_case_native.json').read_text())
    previous = {tuple(r[k] for k in ('split', 'file', 'start', 'channel', 'method')): r for r in old}
    dump(OUT / 'protocol.json', dict(backbone_sha256=digest(model_path),
        source_sha256=digest(ROOT / 'evaluate_native_ensemble8_32_crps.py'),
        seeds8=list(range(500000, 500008)), seeds32=list(range(500000, 500032)),
        target='complete next20 future frames; all sample paths generated before truth',
        scoring='Marginal empirical CRPS and fair estimator. Store mean absolute term A and half-pair term B; score=A-B. Separate original-unit and train-std-standardized terms.',
        caveats='Native ensembles are not online-corrected or recentered. Their means differ from the shared zero-noise point used by the matched interval table. Full path arrays are not distributed; a small auditable sample snapshot is retained. Reused historical test; exploratory.'))
    rows, snapshots = [], {}
    pixels = torch.linspace(0, 20 * 32 * 64 - 1, 256, device='cuda').long()
    for split in ('development', 'test'):
        records = manifest()['files'][split]
        for start in range(0, 761, 20):
            h = (history(split, start) - norm['mean']) / norm['std']
            physical = torch.stack([model.rollout(h, seed, sign=1) * norm['std'] + norm['mean']
                                    for seed in range(500000, 500032)])
            ref = issued('diffusion', split, start)
            assert torch.equal(physical.std(0), ref['std32']) and torch.equal(physical.mean(0), ref['mean'])
            with (OUT / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event='all32_paths_issued', split=split, start=start, time=time.time())) + '\n')
            truth_physical = target(split, start)
            with (OUT / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event='complete20_target_revealed', split=split, start=start, time=time.time())) + '\n')
            truth = (truth_physical - norm['mean']) / norm['std']
            paths = (physical - norm['mean']) / norm['std']
            if split == 'test' and start == 0:
                snapshots['samples32'] = paths[:, [0, -1]].flatten(3)[:, :, :, pixels].cpu().numpy()
                snapshots['truth'] = truth[[0, -1]].flatten(2)[:, :, pixels].cpu().numpy()
                snapshots['pixels'] = pixels.cpu().numpy()
            for count in (8, 32):
                samples = paths[:count]
                absolute = (samples - truth.unsqueeze(0)).abs().mean(0)
                scores = ensemble_crps(samples, truth)
                for estimator, score in scores.items():
                    pair = absolute - score
                    values = dict(crps=score, abs_term=absolute, half_pair_term=pair,
                                  raw_crps=score * norm['std'], raw_abs_term=absolute * norm['std'],
                                  raw_half_pair_term=pair * norm['std'])
                    reduced = {k: v.mean((2, 3, 4)).cpu() for k, v in values.items()}
                    for i, record in enumerate(records):
                        for c, channel in enumerate(('u', 'v')):
                            row = dict(split=split, start=start, file=record['file'], group=record['group'],
                                channel=channel, samples=count, estimator=estimator,
                                **{k: float(v[i, c]) for k, v in reduced.items()})
                            if count == 32:
                                old_row = previous[split, record['file'], start, channel, estimator]
                                assert row['crps'] == old_row['crps'] and row['raw_crps'] == old_row['raw_crps']
                            rows.append(row)
                    if split == 'test' and start == 0:
                        snapshots['score_{}_{}'.format(count, estimator)] = score[[0, -1]].flatten(2)[:, :, pixels].cpu().numpy()
            if start in (0, 180, 380, 580, 760):
                print('NATIVE_ENSEMBLE', split, start, flush=True)
    dump(OUT / 'per_case.json', rows)
    np.savez_compressed(OUT / 'snapshot.npz', **snapshots)
    buckets = defaultdict(list)
    for row in rows:
        if row['start'] >= 20:
            buckets[(row['split'], row['group'], row['channel'], row['samples'], row['estimator'])].append(row)
    summary = []
    for split in ('development', 'test'):
        for count in (8, 32):
            for estimator in ('empirical', 'fair'):
                cells = [v for k, v in buckets.items() if k[0] == split and k[3:] == (count, estimator)]
                assert len(cells) == 6
                metrics = ('crps', 'raw_crps', 'abs_term', 'half_pair_term', 'raw_abs_term', 'raw_half_pair_term')
                summary.append(dict(split=split, samples=count, estimator=estimator,
                    **{metric: mean(mean(r[metric] for r in cell) for cell in cells) for metric in metrics}))
    assert len(rows) == 7488
    dump(OUT / 'summary.json', summary)
    dump(OUT / 'complete.json', dict(passed=True, rows=len(rows), actual_paths=True,
        native32_reproduced_rows=3744, full32_cached_moments_exact=78, full_window_causal_pairs=78))
    print('NATIVE_ENSEMBLE8_32_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
