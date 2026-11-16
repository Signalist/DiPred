"""Aggregate actual replay and archived scores; never synthesize window values."""
import json
import sys
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from clean_reliability_common import manifest
from experiment_reliability_transport_e12 import normalization

RUN = ROOT / 'runs/legacy_xpf16_feature_ablation_20261116'
OUT = RUN / 'publication'
ARMS = ['X', 'P', 'F', 'XP', 'XF', 'PF', 'XPF']
COLORS = {'X':'#777777', 'P':'#9467bd', 'F':'#0072b2', 'XP':'#b59a00', 'XF':'#009e73', 'PF':'#cc79a7', 'XPF':'#d55e00'}
METRICS = ['crps', 'wis', 'crossing_before', 'crossing_any_before', 'within_pair_crossing', 'coverage',
           'width', 'interval_score', 'lower_miss', 'upper_miss']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = manifest()['files']['test']
    std = normalization()['std'].cpu().numpy().reshape(2)
    rows = []
    for arm in ARMS:
        directory = RUN / 'evaluation' / arm
        assert (directory / 'run.json').exists(), f'{arm} is incomplete'
        for start in range(20, 821, 20):
            with np.load(directory / f'{start:04d}.npz') as data:
                values = {f'{mode}_{metric}': data[f'{mode}_{metric}_lead'] for mode in ['raw', 'online'] for metric in METRICS}
                for mode in ['raw', 'online']:
                    for i, record in enumerate(records):
                        for c, channel in enumerate(['u', 'v']):
                            for lead in range(20):
                                row = dict(method=arm, mode=mode, start=start,
                                    window_index=start // 20 + 1, trajectory_id=record['file'],
                                    Re=record['re'], channel=channel, lead=lead + 1)
                                for metric in METRICS:
                                    row[metric] = float(values[f'{mode}_{metric}'][i, c, lead])
                                for metric in ['crps', 'wis', 'width', 'interval_score']:
                                    row[f'{metric}_physical'] = row[metric] * float(std[c])
                                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'per_frame.csv.gz', index=False, compression='gzip')
    keys = ['method', 'mode', 'start', 'window_index', 'trajectory_id', 'Re', 'channel']
    windows = frame.groupby(keys, as_index=False, sort=False).mean(numeric_only=True).drop(columns='lead')
    windows.to_csv(OUT / 'seven_features_per_window.csv', index=False)
    windows.groupby(['method', 'mode', 'Re', 'channel'], as_index=False).mean(numeric_only=True).to_csv(OUT / 'by_re_channel.csv', index=False)
    windows.groupby(['method', 'mode', 'trajectory_id', 'Re', 'channel'], as_index=False).mean(numeric_only=True).to_csv(OUT / 'by_trajectory_channel.csv', index=False)
    frame.groupby(['method', 'mode', 'channel', 'lead'], as_index=False).mean(numeric_only=True).to_csv(OUT / 'by_future_frame.csv', index=False)
    summary = windows.groupby(['method', 'mode'], as_index=False).mean(numeric_only=True)
    summary.to_csv(OUT / 'summary_trajectory_channel_equal.csv', index=False)
    re_equal = windows.groupby(['method', 'mode', 'Re', 'channel'], as_index=False).mean(numeric_only=True).groupby(['method', 'mode'], as_index=False).mean(numeric_only=True)
    re_equal.to_csv(OUT / 'summary_re_channel_equal.csv', index=False)
    old49 = pd.read_csv(RUN / 'historical_49_windows.csv')
    old250 = pd.read_csv(RUN / 'historical_250_windows.csv')
    noise = pd.concat([old49[old49.method != 'XPF'], old250[old250.method == '2048-noise']])
    noise.to_csv(OUT / 'noise_per_window.csv', index=False)
    current = windows[(windows.method == 'XPF') & (windows['mode'] == 'online')]
    original = old49[(old49.method == 'XPF') & (old49['mode'] == 'online')]
    joined = current.merge(original, on=['start', 'trajectory_id', 'channel'], suffixes=('_new', '_old'))
    assert len(joined) == 41 * 12 * 2
    maxdiff = float((joined.crps - joined.standardized_crps).abs().max())
    mean_old = float(original.standardized_crps.mean())
    print('Legacy replay max absolute difference:', maxdiff, 'historical mean:', mean_old, flush=True)
    if maxdiff > 2e-5:
        raise RuntimeError('Original XPF replay did not match archived results; investigate before reporting.')
    curves = windows[windows['mode'] == 'online'].groupby(['method', 'window_index']).crps.mean().unstack(0)
    assert curves.shape == (41, 7) and np.isfinite(curves.to_numpy()).all()
    # Publication figures are generated from the exact exported CSV artifacts.
    subprocess.run([sys.executable, str(ROOT / 'legacy_xpf16_feature_ablation/draw_from_csv.py'), str(OUT)], check=True)
    # Resample whole trajectories; all frames and channels remain paired.
    online = windows[windows['mode'] == 'online']
    scores = online.groupby(['method', 'trajectory_id']).crps.mean().unstack(0)
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(scores), size=(10000, len(scores)))
    comparisons = []
    for base in ['XPF', 'F']:
        for arm in ARMS:
            if arm == base:
                continue
            delta = (scores[arm] - scores[base]).to_numpy()
            bootstrap = delta[samples].mean(axis=1)
            a = online[online.method == arm].groupby('window_index').crps.mean()
            b = online[online.method == base].groupby('window_index').crps.mean()
            comparisons.append(dict(method=arm, reference=base, mean_difference=delta.mean(),
                ci95_low=np.quantile(bootstrap,.025), ci95_high=np.quantile(bootstrap,.975),
                trajectory_win_rate=float((delta < 0).mean()), window_win_rate=float((a < b).mean()),
                relative_percent=100 * delta.mean() / scores[base].mean()))
    for reference in ['X', 'P', 'XP', 'XF', 'PF']:
        delta = (scores['F'] - scores[reference]).to_numpy()
        bootstrap = delta[samples].mean(axis=1)
        comparisons.append(dict(method='F', reference=reference, mean_difference=delta.mean(),
            ci95_low=np.quantile(bootstrap,.025), ci95_high=np.quantile(bootstrap,.975),
            trajectory_win_rate=float((delta < 0).mean()),
            window_win_rate=float((curves['F'] < curves[reference]).mean()),
            relative_percent=100 * delta.mean() / scores[reference].mean()))
    for method in ['8-noise', '32-noise', '256-noise', '2048-noise']:
        n = noise[(noise.method == method) & (noise['mode'] == 'online') & (noise.estimator == 'empirical')]
        noise_traj = n.groupby('trajectory_id').standardized_crps.mean()
        noise_window = n.groupby('window_index').standardized_crps.mean()
        for arm in ['XPF', 'F']:
            paired = pd.concat([scores[arm], noise_traj], axis=1, join='inner').dropna()
            assert len(paired) == 12
            delta = (paired.iloc[:, 0] - paired.iloc[:, 1]).to_numpy()
            bootstrap = delta[samples].mean(axis=1)
            comparisons.append(dict(method=arm, reference=method, mean_difference=delta.mean(),
                ci95_low=np.quantile(bootstrap,.025), ci95_high=np.quantile(bootstrap,.975),
                trajectory_win_rate=float((delta < 0).mean()),
                window_win_rate=float((curves[arm] < noise_window).mean()),
                relative_percent=100 * delta.mean() / paired.iloc[:, 1].mean()))
    pd.DataFrame(comparisons).to_csv(OUT / 'paired_comparison.csv', index=False)
    timings = []
    projections = []
    for arm in ARMS:
        run = json.loads((RUN / 'evaluation' / arm / 'run.json').read_text())
        t = [x for x in run['timings'] if x['start'] > 0]
        projections.extend(dict(method=arm, **x) for x in run['online_projection'])
        timings.append(dict(method=arm, heads=49, parameters_per_head=34004,
            total_head_parameters=49*34004, timing_batch_size=run['timing_batch_size'],
            mean_batch_predict_seconds=np.mean([x['predict_seconds'] for x in t]),
            mean_batch_heads_seconds=np.mean([x['head49_seconds'] for x in t]),
            peak_gpu_allocated_bytes=run['peak_gpu_allocated_bytes'], timing_scope=run['timing_scope']))
    pd.DataFrame(timings).to_csv(OUT / 'timing.csv', index=False)
    pd.DataFrame(projections).to_csv(OUT / 'online_pair_projection.csv', index=False)
    (OUT / 'replay_audit.json').write_text(json.dumps(dict(xpf_replay_max_abs_diff=maxdiff,
        historical_xpf_online_crps=mean_old, expected_trajectory_windows=41*12*2,
        paired_bootstrap_unit='trajectory', paired_bootstrap_samples=10000,
        caveat='2048 and 49-grid noise used different original runs and calibration protocols; no claim of identical protocols.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
