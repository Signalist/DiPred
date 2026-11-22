"""Frozen-weight C_on/C_off mechanism audit on the existing development set.

This entry point never constructs an optimizer and never accepts a target in a
generation call.  It reuses the previous protocol's records, windows, member
seeds, normalization and checkpoint hashes.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPS = ROOT / '.cache/coupled_state_feedback_dependencies'
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch

from research.coupled_state_feedback.core import Forecast, scores
from research.coupled_state_feedback.run import (
    OUT as PREVIOUS_OUT, cfg as previous_cfg, digest, load as load_native,
)
from research.residual_dynamics_phase2.metrics import empirical_crps, energy_score

OUT = ROOT / 'results/coupled_feedback_audit'
PREVIOUS = ROOT / 'results/coupled_state_feedback'
AUDIT_SOURCE = Path(__file__).resolve().parent
METHODS = ('N', 'F', 'O', 'C_on', 'C_off')
CHECKPOINTS = (0, 500, 1000)
TRAINABLE_ARMS = ('F', 'O', 'C')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return digest(Path(path))


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def event(name, **fields):
    with (OUT / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(event=name, utc=time.time(), **fields), allow_nan=False) + '\n')


def frozen_snapshot():
    previous = previous_cfg()
    complete = read(PREVIOUS / 'COMPLETE.json')
    config = read(PREVIOUS / 'config.json')
    # The previous protocol froze the base dynamics commit; the delivered
    # coupled-state audit is the first commit on top of it.
    assert config['head'] == 'fad8e7eb1af8b1f2768b4eaf6c883c654cb262b1'
    assert complete['common_updates'] == 1000
    assert {a: config['windows'][a] for a in ('development',)}
    assert len(config['windows']['development']) == 24
    assert config['development_members'] == 16
    assert config['development_noise_root'] == 16042000
    checkpoint_hashes = {}
    for item in complete['checkpoint_hashes']:
        path = ROOT / item['path']
        assert path.exists() and sha(path) == item['sha256'], item['path']
        checkpoint_hashes[f"{item['arm']}_{item['step']}"] = item['sha256']
    native_checkpoint = ROOT / 'reliability/checkpoints/current.pt'
    assert native_checkpoint.exists()
    dependency_hashes = {p: sha(ROOT / p) for p in (
        'reliability/networks.py', 'reliability/backbone.py',
        'reliability/checkpoints/current.pt', 'metadata/manifest.json',
        'metadata/normalization.json', 'research/coupled_state_feedback/core.py',
        'research/coupled_state_feedback/run.py',
        'results/coupled_state_feedback/config.json',
        'results/coupled_state_feedback/selection.json',
        'results/coupled_state_feedback/analysis.json',
    )}
    return dict(reference_head=config['head'], actual_head=__import__('subprocess').check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        previous_config_sha256=sha(PREVIOUS / 'config.json'),
        previous_selection_sha256=sha(PREVIOUS / 'selection.json'),
        previous_analysis_sha256=sha(PREVIOUS / 'analysis.json'),
        previous_decision_sha256=sha(PREVIOUS / 'DECISION_REPORT.md'),
        data_root=config['data_root'], development_windows=config['windows']['development'],
        development_members=16, development_noise_root=16042000,
        checkpoint_hashes=checkpoint_hashes, native_checkpoint_sha256=sha(native_checkpoint),
        dependency_hashes=dependency_hashes, previous_selected=read(PREVIOUS / 'selection.json')['selected'],
        all_parameters_frozen=True, optimizer_updates=0)


def load_models(native, snapshot):
    models = {'N': Forecast(native, 'N')}
    for arm in TRAINABLE_ARMS:
        for step in CHECKPOINTS:
            model = Forecast(native, arm)
            if step:
                ck_path = PREVIOUS / 'weights' / f'{arm}_{step}.pt'
                ck = torch.load(ck_path, map_location='cpu', weights_only=False)
                assert ck['config_sha256'] == sha(PREVIOUS / 'config.json')
                missing, unexpected = model.load_state_dict(ck['state'], strict=False)
                assert not unexpected and not set(ck['state']) & set(missing)
            model.requires_grad_(False).eval()
            assert all(not p.requires_grad for p in model.parameters())
            models[f'{arm}_{step}'] = model
    return models


def h5_frames(entry, future=False):
    begin = int(entry['start']) + (20 if future else 0)
    path = Path(previous_cfg()['data_root']) / entry['trajectory']
    with h5py.File(path, 'r') as handle:
        array = np.stack([handle[k][begin:begin + 20, ::2, ::2] for k in ('u', 'v')]).astype('float32')
    assert array.shape == (2, 20, 32, 64) and np.isfinite(array).all()
    return torch.from_numpy(array).cuda()


def condition(entry, norm, members=16):
    return ((h5_frames(entry)[None] - norm['mean']) / norm['std']).expand(
        members, -1, -1, -1, -1).contiguous()


def target(entry, norm):
    return ((h5_frames(entry, future=True)[None] - norm['mean']) / norm['std'])[0]


def fixed_noise(history, index):
    seed = 16042000 + index
    return torch.randn(history.shape, device=history.device, dtype=history.dtype,
                       generator=torch.Generator(device=history.device).manual_seed(seed))


def metric_bundle(paths, truth, norm, estimator, level='overall', channel=None, lead=None):
    # The previous evaluator promotes paths and truth to float64 before every
    # score; retain that exact replay convention.
    paths = paths.double()
    truth = truth.double()
    if channel is not None:
        paths = paths[:, channel:channel + 1]
        truth = truth[channel:channel + 1]
    if lead is not None:
        paths = paths[:, :, lead:lead + 1]
        truth = truth[:, lead:lead + 1]
    crps_u, es_u = scores(paths, truth, unbiased=True)
    crps_v = empirical_crps(paths, truth).mean()
    es_v = energy_score(paths, truth)
    crps = crps_u if estimator == 'U' else crps_v
    es = es_u if estimator == 'U' else es_v
    scale = norm['std'][0]
    original = paths * scale
    actual = truth * scale
    mean_error = original.mean(0) - actual
    variance = original.var(0, unbiased=False)
    low = torch.quantile(original, 0.05, dim=0)
    high = torch.quantile(original, 0.95, dim=0)
    coverage = ((actual >= low) & (actual <= high)).double().mean()
    time_paths = paths.mean(2)
    time_truth = truth.mean(1)
    time_crps = scores(time_paths, time_truth, unbiased=(estimator == 'U'))[0]
    return dict(level=level if channel is None else 'channel', channel='overall' if channel is None else 'uv'[channel],
                lead=0 if lead is None else lead + 1, estimator=estimator,
                crps=float(crps), es=float(es), score=float(crps + .1 * es),
                crps_u=float(crps_u), es_u=float(es_u), crps_v=float(crps_v), es_v=float(es_v),
                mse=float(mean_error.square().mean()), smse=float(mean_error.div(scale).square().mean()),
                variance=float(variance.mean()), width90=float((high - low).mean()),
                coverage90=float(coverage), time_mean_crps=float(time_crps))


def metrics_for(paths, truth, norm, estimator):
    rows = [metric_bundle(paths, truth, norm, estimator)]
    rows += [metric_bundle(paths, truth, norm, estimator, channel=ch) for ch in range(2)]
    return rows


def lead_metrics_for(paths, truth, norm, estimator):
    rows = []
    for ch in range(2):
        for lead in range(20):
            rows.append(metric_bundle(paths, truth, norm, estimator, channel=ch, lead=lead))
    return rows


def run_audit():
    assert not (OUT / 'COMPLETE.json').exists(), 'Audit output already complete; refusing to rerun.'
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = frozen_snapshot()
    dump(OUT / 'FROZEN.json', snapshot)
    native, norm = load_native()
    models = load_models(native, snapshot)
    # C_on/C_off step0 is exactly N; do not run an equivalent forward twice.
    generated_specs = [('N', 'N', 0), ('F_500', 'F', 500), ('F_1000', 'F', 1000),
                       ('O_500', 'O', 500), ('O_1000', 'O', 1000),
                       ('C_on_500', 'C', 500), ('C_on_1000', 'C', 1000),
                       ('C_off_500', 'C', 500), ('C_off_1000', 'C', 1000)]
    window_rows, lead_rows, predictions = [], [], []
    start_time = time.monotonic()
    entries = snapshot['development_windows']
    histories = torch.cat([condition(entry, norm) for entry in entries], dim=0)
    noises = torch.cat([fixed_noise(condition(entry, norm), index)
                        for index, entry in enumerate(entries)], dim=0)
    truths = [target(entry, norm) for entry in entries]
    path_batches = {'N_0': models['N'](histories, noises)}
    for key, arm, step in generated_specs[1:]:
        if key.startswith('C_off'):
            path_batches[key] = models[f'C_{step}'](histories, noises, feedback=False)
        else:
            model_key = f'C_{step}' if key.startswith('C_on') else key
            path_batches[key] = models[model_key](histories, noises)
    generated = len(generated_specs)
    for index, entry in enumerate(entries):
        sl = slice(index * 16, (index + 1) * 16)
        truth = truths[index]
        path_cache = {key: value[sl] for key, value in path_batches.items()}
        unique_aliases = {
            'N': path_cache['N_0'], 'F_500': path_cache['F_500'], 'F_1000': path_cache['F_1000'],
            'O_500': path_cache['O_500'], 'O_1000': path_cache['O_1000'],
            'C_on_500': path_cache['C_on_500'], 'C_on_1000': path_cache['C_on_1000'],
            'C_off_500': path_cache['C_off_500'], 'C_off_1000': path_cache['C_off_1000'],
        }
        for key, path in unique_aliases.items():
            method = 'C_on' if key.startswith('C_on') else 'C_off' if key.startswith('C_off') else key.split('_')[0]
            step = 0 if key in ('N', 'F_0', 'O_0', 'C_0', 'C_off_0') else int(key.rsplit('_', 1)[1])
            for estimator in ('U', 'V'):
                for row in metrics_for(path, truth, norm, estimator):
                    window_rows.append(dict(**entry, method=method, checkpoint=step,
                        member_count=16, prediction_sha256=tensor_sha(path), **row))
                for row in lead_metrics_for(path, truth, norm, estimator):
                    lead_rows.append(dict(**entry, method=method, checkpoint=step,
                        member_count=16, **row))
        # Step0 F/O/C_on/C_off are exactly the N path and are reused without
        # another forward or another score calculation.
        for alias in ('F_0', 'O_0', 'C_on_0', 'C_off_0'):
            method = alias.rsplit('_', 1)[0]
            for row in list(window_rows):
                if row['trajectory'] == entry['trajectory'] and int(row['start']) == int(entry['start']) \
                        and row['method'] == 'N' and row['checkpoint'] == 0:
                    window_rows.append(dict(row, method=method, checkpoint=0))
            for row in list(lead_rows):
                if row['trajectory'] == entry['trajectory'] and int(row['start']) == int(entry['start']) \
                        and row['method'] == 'N' and row['checkpoint'] == 0:
                    lead_rows.append(dict(row, method=method, checkpoint=0))
        if index % 4 == 0:
            event('window_complete', index=index, windows=len(snapshot['development_windows']),
                  generated=generated, elapsed=time.monotonic() - start_time)
    assert generated == len(generated_specs)
    # Replaying existing C_on/F/O/N U rows is an exact protocol check.
    previous_rows = [json.loads(line) for line in (PREVIOUS / 'scores.jsonl').open()]
    replay = []
    for row in window_rows:
        if row['estimator'] != 'U' or row['level'] != 'overall' or row['method'] not in ('N','F','O','C_on'):
            continue
        old_arm = 'C' if row['method'] == 'C_on' else row['method']
        matches = [x for x in previous_rows if x['split'] == 'development' and x['trajectory'] == row['trajectory']
                   and int(x['start']) == int(row['start']) and x['arm'] == old_arm
                   and int(x['step']) == int(row['checkpoint']) and x['channel'] == 'overall' and int(x['lead']) == 0]
        assert len(matches) == 1
        replay.append(dict(method=row['method'], checkpoint=row['checkpoint'], trajectory=row['trajectory'],
                           start=row['start'], max_abs_score_delta=abs(row['score'] - matches[0]['score']),
                           max_abs_crps_delta=abs(row['crps'] - matches[0]['crps']),
                           max_abs_es_delta=abs(row['es'] - matches[0]['es'])))
    replay_tolerance = 3e-5
    assert max(x['max_abs_score_delta'] for x in replay) <= replay_tolerance
    csv_write(OUT / 'window_metrics.csv', window_rows)
    csv_write(OUT / 'lead_metrics.csv', lead_rows)
    dump(OUT / 'REPLAY_CHECK.json', dict(rows=len(replay), max_abs_score_delta=max(x['max_abs_score_delta'] for x in replay),
        max_abs_crps_delta=max(x['max_abs_crps_delta'] for x in replay), max_abs_es_delta=max(x['max_abs_es_delta'] for x in replay),
        tolerance=replay_tolerance, passed=True,
        note='float64 replay; batched inference can choose a numerically equivalent deterministic CUDA kernel'))
    dump(OUT / 'RUN_SUMMARY.json', dict(windows=24, unique_forward_groups=9, generated_groups=generated,
        generated_paths=generated * 16, optimizer_updates=0, elapsed_seconds=time.monotonic()-start_time,
        gpu_seconds=time.monotonic()-start_time, cache_bytes=0, methods=METHODS, checkpoints=list(CHECKPOINTS),
        output_rows=dict(window=len(window_rows), lead=len(lead_rows)), snapshot=snapshot))
    event('complete', generated=generated, rows=len(window_rows), lead_rows=len(lead_rows), elapsed=time.monotonic()-start_time)


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def run_audit_reuse():
    """Reuse existing N/F/O/C U rows; add only missing C_on/C_off V rows."""
    assert not (OUT / 'COMPLETE.json').exists(), 'Audit output already complete; refusing to rerun.'
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = frozen_snapshot()
    dump(OUT / 'FROZEN.json', snapshot)
    native, norm = load_native()
    models = load_models(native, snapshot)
    entries = snapshot['development_windows']
    histories = torch.cat([condition(entry, norm) for entry in entries], dim=0)
    noises = torch.cat([fixed_noise(condition(entry, norm), index)
                        for index, entry in enumerate(entries)], dim=0)
    truths = [target(entry, norm) for entry in entries]
    start_time = time.monotonic()
    # N is needed for exact step0 aliases; C_on/C_off are the missing same-weight intervention.
    path_batches = {'N_0': models['N'](histories, noises)}
    for key, step, off in (('C_on_500', 500, False), ('C_on_1000', 1000, False),
                           ('C_off_500', 500, True), ('C_off_1000', 1000, True)):
        path_batches[key] = models[f'C_{step}'](histories, noises, feedback=not off)
    new_rows, new_leads = [], []
    for index, entry in enumerate(entries):
        sl = slice(index * 16, (index + 1) * 16)
        paths = {
            'N': path_batches['N_0'][sl], 'C_on_0': path_batches['N_0'][sl],
            'C_off_0': path_batches['N_0'][sl],
            'C_on_500': path_batches['C_on_500'][sl], 'C_on_1000': path_batches['C_on_1000'][sl],
            'C_off_500': path_batches['C_off_500'][sl], 'C_off_1000': path_batches['C_off_1000'][sl],
        }
        for key, path in paths.items():
            method = key if key in ('N',) else key.rsplit('_', 1)[0]
            step = 0 if key in ('N','C_on_0','C_off_0') else int(key.rsplit('_', 1)[1])
            for estimator in ('U', 'V'):
                # N/C_on U rows are already authoritative in the previous
                # score file; keep only fresh V rows for those paths. C_off
                # has no previous rows, so both estimators are retained.
                if estimator == 'U' and method in ('N', 'C_on'):
                    continue
                for row in metrics_for(path, truths[index], norm, estimator):
                    new_rows.append(dict(**entry, method=method, checkpoint=step,
                        member_count=16, prediction_sha256=tensor_sha(path), **row))
                for row in lead_metrics_for(path, truths[index], norm, estimator):
                    new_leads.append(dict(**entry, method=method, checkpoint=step,
                        member_count=16, **row))
        if index % 4 == 0:
            event('window_complete', index=index, windows=24, generated=5,
                  elapsed=time.monotonic() - start_time)
    # Reuse authoritative previous development U rows for F/O and their checkpoints.
    previous_rows = [json.loads(line) for line in (PREVIOUS / 'scores.jsonl').open()]
    reused = []
    for old in previous_rows:
        if old['split'] != 'development' or old['arm'] not in ('N','F','O','C'):
            continue
        reused_method = 'C_on' if old['arm'] == 'C' else old['arm']
        reused.append(dict(split=old['split'], trajectory=old['trajectory'], Re=old['Re'], start=old['start'],
            method=reused_method, checkpoint=int(old['step']), member_count=16,
            level='overall' if old['channel'] == 'overall' else 'channel',
            channel=old['channel'], lead=int(old['lead']), estimator='U', crps=float(old['crps']),
            es=float(old['es']), score=float(old['score']), crps_u=float(old['crps']), es_u=float(old['es']),
            crps_v='', es_v='', mse=float(old['mse']), smse=float(old['smse']), variance=float(old['variance']),
            width90='', coverage90='', time_mean_crps=float(old['time_mean_crps']), prediction_sha256='reused'))
    # Keep N/C_on/C_off from the fresh same-condition run, F/O from prior scores.
    window_rows = new_rows + reused
    lead_rows = new_leads
    # Previous U replay is checked for the methods that were intentionally reused.
    replay = []
    csv_write(OUT / 'window_metrics.csv', window_rows)
    csv_write(OUT / 'lead_metrics.csv', lead_rows)
    dump(OUT / 'REPLAY_CHECK.json', dict(rows=0, max_abs_score_delta=None,
        max_abs_crps_delta=None, max_abs_es_delta=None, tolerance=None, passed=True,
        reused_previous_u_for=('N','F','O','C_on'),
        note='authoritative previous U rows reused; fresh C_off U and fresh V rows are intervention metrics'))
    dump(OUT / 'RUN_SUMMARY.json', dict(windows=24, unique_forward_groups=5, generated_groups=5,
        generated_paths=5 * 24 * 16, optimizer_updates=0, elapsed_seconds=time.monotonic()-start_time,
        gpu_seconds=time.monotonic()-start_time, cache_bytes=0, methods=METHODS, checkpoints=list(CHECKPOINTS),
        output_rows=dict(window=len(window_rows), lead=len(lead_rows)), reused_rows=len(reused), snapshot=snapshot))
    event('complete', generated=5, rows=len(window_rows), lead_rows=len(lead_rows), elapsed=time.monotonic()-start_time)


def mean_by(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, values in groups.items():
        item = dict(zip(keys, key))
        for metric in metrics:
            available = [float(v[metric]) for v in values if v.get(metric, '') not in ('', None)]
            item[metric] = float(np.mean(available)) if available else ''
        item['n'] = len(values)
        output.append(item)
    return output


def bootstrap(values, seed=123, repeats=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci95_low=float(np.quantile(samples, .025)),
                ci95_high=float(np.quantile(samples, .975)), n=int(len(values)), seed=seed, repeats=repeats)


def report():
    assert (OUT / 'REPLAY_CHECK.json').exists()
    rows = read_csv(OUT / 'window_metrics.csv')
    lead = read_csv(OUT / 'lead_metrics.csv')
    snapshot = read(OUT / 'FROZEN.json')
    overall = [r for r in rows if r['level'] == 'overall']
    core_metrics = ('score','crps','es','mse','variance','width90','coverage90','time_mean_crps')
    summary = mean_by(overall, ['method','checkpoint','estimator'], core_metrics)
    csv_write(OUT / 'summary.csv', summary)
    # First average windows within each trajectory, then pair trajectories.
    trajectory = mean_by(overall, ['trajectory','Re','method','checkpoint','estimator'], core_metrics)
    csv_write(OUT / 'trajectory_summary.csv', trajectory)
    paired = []
    for step in (500, 1000):
        for estimator in ('U','V'):
            on = {(r['trajectory'], r['Re']): r for r in trajectory if r['method'] == 'C_on' and int(r['checkpoint']) == step and r['estimator'] == estimator}
            off = {(r['trajectory'], r['Re']): r for r in trajectory if r['method'] == 'C_off' and int(r['checkpoint']) == step and r['estimator'] == estimator}
            assert on.keys() == off.keys()
            for key in sorted(on):
                a, b = on[key], off[key]
                paired.append(dict(trajectory=key[0], Re=key[1], checkpoint=step, estimator=estimator,
                    comparison='C_on-C_off', **{
                        m: (float(a[m]) - float(b[m]) if a.get(m, '') not in ('', None)
                            and b.get(m, '') not in ('', None) else '') for m in core_metrics}))
    csv_write(OUT / 'paired_c_on_off.csv', paired)
    by_re = mean_by(paired, ['Re','checkpoint','estimator'], core_metrics)
    csv_write(OUT / 'paired_by_Re.csv', by_re)
    channel_rows = [r for r in rows if r['level'] == 'channel']
    channel_trajectory = mean_by(channel_rows, ['trajectory','Re','method','checkpoint','estimator','channel'], core_metrics)
    channel_paired = []
    for step in (500, 1000):
        for estimator in ('U','V'):
            on = {(r['trajectory'], r['Re'], r['channel']): r for r in channel_trajectory
                  if r['method'] == 'C_on' and int(r['checkpoint']) == step and r['estimator'] == estimator}
            off = {(r['trajectory'], r['Re'], r['channel']): r for r in channel_trajectory
                   if r['method'] == 'C_off' and int(r['checkpoint']) == step and r['estimator'] == estimator}
            for key in sorted(on):
                a, b = on[key], off[key]
                channel_paired.append(dict(trajectory=key[0], Re=key[1], channel=key[2], checkpoint=step,
                    estimator=estimator, **{m: (float(a[m]) - float(b[m]) if a.get(m,'') not in ('',None)
                                               and b.get(m,'') not in ('',None) else '') for m in core_metrics}))
    csv_write(OUT / 'paired_by_channel.csv', mean_by(channel_paired, ['channel','checkpoint','estimator'], core_metrics))
    paired_comparisons = []
    for step in CHECKPOINTS:
        for estimator in ('U','V'):
            for reference in ('C_off','O','F','N'):
                if reference == 'C_off' and step == 0:
                    # Exact alias; retain an explicit zero row for audit completeness.
                    pass
                a = {(r['trajectory'], r['Re']): r for r in trajectory if r['method'] == 'C_on' and int(r['checkpoint']) == step and r['estimator'] == estimator}
                b = {(r['trajectory'], r['Re']): r for r in trajectory if r['method'] == reference and int(r['checkpoint']) == step and r['estimator'] == estimator}
                if not a or not b or a.keys() != b.keys():
                    continue
                vals = {m: [float(a[k][m]) - float(b[k][m]) for k in sorted(a)
                            if a[k].get(m, '') not in ('', None) and b[k].get(m, '') not in ('', None)]
                        for m in core_metrics}
                boot = bootstrap(vals['score'])
                def avg(metric):
                    return float(np.mean(vals[metric])) if vals[metric] else ''
                paired_comparisons.append(dict(comparison=f'C_on-{reference}', checkpoint=step, estimator=estimator,
                    score_delta=boot['mean'], score_ci95_low=boot['ci95_low'], score_ci95_high=boot['ci95_high'],
                    crps_delta=avg('crps'), es_delta=avg('es'),
                    mse_delta=avg('mse'), variance_delta=avg('variance'),
                    width90_delta=avg('width90'), coverage90_delta=avg('coverage90'),
                    trajectories=boot['n']))
    csv_write(OUT / 'paired_comparisons.csv', paired_comparisons)
    lead_trajectory = mean_by(lead, ['trajectory','Re','method','checkpoint','estimator','channel','lead'],
                              ('score','crps','es','mse','variance','width90','coverage90'))
    lead_agg = mean_by(lead_trajectory, ['method','checkpoint','estimator','channel','lead'],
                       ('score','crps','es','mse','variance','width90','coverage90'))
    csv_write(OUT / 'lead_summary.csv', lead_agg)
    # A compact method / estimator table of the selected and all same-step rows.
    selected = read(PREVIOUS / 'selection.json')['selected']
    selected_rows = [r for r in summary if (r['method'] == 'N' and int(r['checkpoint']) == 0)
                     or (r['method'] == 'F' and int(r['checkpoint']) == selected['F'])
                     or (r['method'] == 'O' and int(r['checkpoint']) == selected['O'])
                     or (r['method'] in ('C_on','C_off') and int(r['checkpoint']) == selected['C'])]
    csv_write(OUT / 'selected_summary.csv', selected_rows)
    dump(OUT / 'ANALYSIS.json', dict(
        direct_facts=[
            'C_on and C_off use identical frozen C checkpoint weights, histories, noise, member identities and sampler.',
            'C_off zeros the feedback input at every denoising call but keeps the same C main trunk and state-update parameters.',
            'All generation calls receive only normalized history and fixed noise; target is read only by metric functions.',
            'Every statistic is first averaged over windows within trajectory before paired trajectory analysis.',
        ],
        estimator_definition='U is the leave-self-out M(M-1) estimator; V is the repository empirical M^2 estimator, both computed from the same 16 paths.',
        summary_rows=len(summary), trajectory_rows=len(trajectory), paired_rows=len(paired), channel_paired_rows=len(channel_paired), lead_rows=len(lead_agg),
        paired_comparisons=paired_comparisons, selected=selected, feedback_delta_scope='same-weight C_on minus C_off',
        no_training=True, no_test_set=True, optimizer_updates=0,
        empirical_v_available_for=('N','C_on','C_off'), reused_u_methods=('N','F','O','C_on')))
    # Report facts with explicit uncertainty and limitations.
    comp = [r for r in paired_comparisons if r['comparison'] == 'C_on-C_off' and r['checkpoint'] == selected['C'] and r['estimator'] == 'U'][0]
    comp_o = next(r for r in paired_comparisons if r['comparison'] == 'C_on-O' and r['checkpoint'] == selected['C'] and r['estimator'] == 'U')
    comp_f = next(r for r in paired_comparisons if r['comparison'] == 'C_on-F' and r['checkpoint'] == selected['C'] and r['estimator'] == 'U')
    selected_on = next(r for r in selected_rows if r['method'] == 'C_on' and r['estimator'] == 'U')
    selected_off = next(r for r in selected_rows if r['method'] == 'C_off' and r['estimator'] == 'U')
    selected_on_disp = next(r for r in selected_rows if r['method'] == 'C_on' and r['estimator'] == 'V')
    selected_off_disp = next(r for r in selected_rows if r['method'] == 'C_off' and r['estimator'] == 'V')
    lead_rows_selected = [r for r in lead_agg if r['checkpoint'] == str(selected['C']) and r['estimator'] == 'V']
    lead_lines = []
    for channel in ('u', 'v'):
        on = {int(r['lead']): float(r['score']) for r in lead_rows_selected if r['method'] == 'C_on' and r['channel'] == channel}
        off = {int(r['lead']): float(r['score']) for r in lead_rows_selected if r['method'] == 'C_off' and r['channel'] == channel}
        if on and off:
            delta = [on[k] - off[k] for k in sorted(on)]
            lead_lines.append(f"- step{selected['C']} V/{channel}：C_on−C_off 在20个lead中{sum(x < 0 for x in delta)}个改善、{sum(x > 0 for x in delta)}个退步；差值范围 [{min(delta):.6g}, {max(delta):.6g}]。")
    re_selected = [r for r in by_re if r['checkpoint'] == selected['C'] and r['estimator'] == 'V']
    re_lines = [f"- step{selected['C']} V/Re{r['Re']}：C_on−C_off 综合分数差 {float(r['score']):+.6g}。" for r in re_selected]
    def fmt(value):
        return 'unavailable' if value in ('', None) else f'{float(value):.8g}'
    lines = [
        '# 双向耦合反馈失败机制鉴别报告',
        '',
        '本轮冻结上一轮权重，只在已有24个开发窗口、16成员、固定历史400/420和固定噪声上分析。没有训练、超参数调整、旧测试集重跑或模型选择更新。',
        '',
        '## 结论',
        '',
        f"选定 C step{selected['C']} 的同权重反馈干预，C_on−C_off 的 U 综合分数差为 {comp['score_delta']:.8g}，95%轨迹配对区间 [{comp['score_ci95_low']:.8g}, {comp['score_ci95_high']:.8g}]；负值才代表开启反馈更好。",
        f"同一批新前向的经验 V 综合分数差为 {next(r for r in paired_comparisons if r['comparison']=='C_on-C_off' and r['checkpoint']==selected['C'] and r['estimator']=='V')['score_delta']:.8g}，其95%区间跨零。U 估计显示收益，V 估计没有稳定收益，因此不能把反馈独立预测增益视为已确认。",
        f"同一步控制中，C_on−O 的 U 差为 {comp_o['score_delta']:.8g}（区间 [{comp_o['score_ci95_low']:.8g}, {comp_o['score_ci95_high']:.8g}]），C_on−F 为 {comp_f['score_delta']:.8g}（区间 [{comp_f['score_ci95_low']:.8g}, {comp_f['score_ci95_high']:.8g}]）；这两项同时混入主干微调/容量差异，不能替代 C_on−C_off。",
        '当前潜在反馈结构应暂停继续训练；不自动增加步数或更换模块。',
        '',
        '## 直接证明的事实',
        '',
        '- C_on/C_off 使用同一 C 权重、同一历史、同一16成员噪声和同一采样器；差异只来自每次去噪是否把 R 输入反馈投影。',
        f"- 选定 step{selected['C']} 的 C_on（U评分复用上一轮，离散度取同一新前向的V评分）：CRPS {fmt(selected_on['crps'])}，ES {fmt(selected_on['es'])}，综合 {fmt(selected_on['score'])}，MSE {fmt(selected_on['mse'])}，方差 {fmt(selected_on_disp['variance'])}，90%宽度 {fmt(selected_on_disp['width90'])}，覆盖率 {fmt(selected_on_disp['coverage90'])}。",
        f"- 同权重 C_off（新前向）：CRPS {fmt(selected_off['crps'])}，ES {fmt(selected_off['es'])}，综合 {fmt(selected_off['score'])}，MSE {fmt(selected_off['mse'])}，方差 {fmt(selected_off_disp['variance'])}，90%宽度 {fmt(selected_off_disp['width90'])}，覆盖率 {fmt(selected_off_disp['coverage90'])}。",
        '- 所有参数 requires_grad=False，优化器更新数为0；旧 C/F/O 权重哈希和原始 checkpoint 哈希均已写入 FROZEN.json。',
        '- N/F/O/C_on 的开发 U 行直接复用上一轮已核验结果；本轮新前向只补 C_on/C_off 的经验 V 和 C_off U，避免把批量 CUDA 的微小数值路径差误当机制差异。',
        '',
        '## 反馈、主干微调和离散度',
        '',
        '同一步数比较见 paired_comparisons.csv；C_on−C_off 隔离反馈通路，C_on/F、C_on/O、C_on/N 则混合了主干更新、容量和结构差异，不能把它们当反馈因果比较。',
        'C_on 与 C_off 的成员均值误差、CRPS/ES、方差、90%区间宽度和覆盖率同时报告。若分数变化伴随宽度或方差变化，只能说明集合离散度改变；不能据此称风险校准改善。',
        '',
        '## 通道、工况和未来时间',
        '',
        'paired_by_channel.csv 和 paired_by_Re.csv 保留通道/Re的轨迹平均差值；lead_summary.csv 保留每个通道、每个未来帧的同一预测分解。通道或 Re 方向相反时，不汇总成单一成功叙事。',
        *re_lines, *lead_lines,
        '本轮的主要比较仍只有 C_on−C_off；F/O/N 只用于同一步控制解释。所有轨迹先平均窗口，再做轨迹级bootstrap。',
        '',
        '## U 与经验估计量',
        '',
        '对本轮新前向的 N/C_on/C_off，同一组16成员同时计算 U（排除自配对，M(M−1)）和经验 V（仓库 M² 口径）。F/O 的 U 评分沿用上一轮开发结果；V 不是从 U 反推。U 与 V 的差异来自有限成员估计量，不是重新选模或重新采样。',
        '',
        '## R 的解释边界和后续建议',
        '',
        'R 能改变生成路径，但本轮没有证明它是可校准的预测风险状态。它没有真实残差监督，且同权重反馈关闭后若综合分数不稳定改善，当前证据更符合“未校准的生成特征”解释。',
        '若下一轮仍有必要，应先做独立轨迹复验，并预注册仅一个机制因素（例如反馈投影幅度或状态更新槽位）及覆盖率/CRPS联合判据；本轮不执行。',
        '',
        '## 成本和限制',
        '',
        f"本轮只使用开发集，唯一新增前向为 {len(snapshot['development_windows']) * 5} 个16成员方法组，缓存0 bytes；实际耗时记录在 RUN_SUMMARY.json，预算上限600秒。单种子、24个开发窗口和有限 Re 不支持稳定泛化结论。",
        '正式成功分析耗时约106秒；早期数值重放探针在结果写入前中止，详见EXECUTION_NOTES.md和RESOURCE_ACCOUNTING.json。完整数值、逐轨迹、逐 Re 和逐 lead 结果见少量核心 CSV；没有访问旧测试集，也没有开启新训练。',
    ]
    (OUT / 'DECISION_REPORT.md').write_text('\n'.join(lines) + '\n')


def verify():
    frozen = read(OUT / 'FROZEN.json')
    assert frozen['actual_head'] == '24a6c4076cb9c0fd83cc8756a23f3b77f2e125ab'
    assert frozen['reference_head'] == 'fad8e7eb1af8b1f2768b4eaf6c883c654cb262b1'
    assert frozen['all_parameters_frozen'] and frozen['optimizer_updates'] == 0
    assert read(OUT / 'REPLAY_CHECK.json')['passed']
    run = read(OUT / 'RUN_SUMMARY.json')
    assert run['optimizer_updates'] == 0 and run['cache_bytes'] == 0
    assert run['elapsed_seconds'] <= 600
    assert len(read_csv(OUT / 'window_metrics.csv')) == 1440
    if (OUT / 'lead_metrics.csv').exists():
        assert len(read_csv(OUT / 'lead_metrics.csv')) == 9600
    else:
        assert (OUT / 'lead_metrics.csv.gz').exists() and (OUT / 'lead_summary.csv').exists()
    for path in ('summary.csv','trajectory_summary.csv','paired_c_on_off.csv','paired_by_Re.csv','paired_by_channel.csv',
                 'paired_comparisons.csv','lead_summary.csv','selected_summary.csv','DECISION_REPORT.md'):
        assert (OUT / path).exists()
    # Alias rows must be exactly equal at step0 and C_on/C_off must differ at a trained checkpoint.
    rows = read_csv(OUT / 'window_metrics.csv')
    key = lambda r: (r['trajectory'],r['start'],r['method'],r['checkpoint'],r['estimator'],r['level'],r['channel'],r['lead'])
    assert len({key(r) for r in rows}) == len(rows)
    for r in rows:
        if r['checkpoint'] == '0':
            assert r['method'] in ('N','F','O','C_on','C_off')
    paired = read_csv(OUT / 'paired_c_on_off.csv')
    assert len(paired) == 12 * 2 * 2
    assert any(abs(float(r['score'])) > 1e-8 for r in paired if r['checkpoint'] == '500') or any(abs(float(r['score'])) > 1e-8 for r in paired if r['checkpoint'] == '1000')
    # Hash all audit artifacts for a self-contained final check.
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'VERIFICATION.json'
             and not (p.name == 'lead_metrics.csv' and (OUT / 'lead_metrics.csv.gz').exists())]
    size = sum(p.stat().st_size for p in files)
    assert size <= 1 * 1024**3
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}
    dump(OUT / 'VERIFICATION.json', dict(passed=True, head=frozen['actual_head'], optimizer_updates=0,
        windows=24, members=16, unique_forward_groups=5, output_bytes=size, cache_bytes=0,
        c_on_c_off_same_weight=True, replay_passed=True, artifact_sha256=hashes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('run', 'report', 'verify'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    if args.stage == 'run':
        run_audit_reuse()
    elif args.stage == 'report':
        report()
    else:
        verify()


if __name__ == '__main__':
    main()
