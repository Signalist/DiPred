"""A0/AR/AH/AHR train-only fits and one frozen test evaluation."""
import gzip
import json
import time

import numpy as np

from research.residual_information_study.information_probe.native_cache import (
    NativeCache, feature_matrices, previous_native_events,
)
from research.residual_information_study.information_probe.probe import (
    BASELINES, COMPARISONS, MODELS, Probe, features, fit, labels, losses, paired_differences, points,
)
from research.observed_residual_feedback.run import dump, read, sha


def paths():
    from research.residual_information_study.common.context import CACHE, OUT, PREVIOUS, ROOT
    return ROOT, OUT, PREVIOUS, OUT / 'information_probe', CACHE / 'information_probe'


def cache_controller():
    from research.residual_information_study.common.context import load_native
    root, out, _, _, cache = paths()
    native, norm, config = load_native()
    return NativeCache(native, norm, root, config['data_root'], cache, sha(out / 'FROZEN.json'))


def write_receipts(path, receipts):
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for receipt in receipts:
                stream.write((json.dumps(receipt, allow_nan=False, separators=(',', ':')) + '\n').encode())


def validate_training_entries(entries):
    from research.residual_information_study.common.context import schedule, windows
    prior = schedule()
    known = {}
    for row in prior['train']:
        known.setdefault((row['trajectory'], row['start']), row['noise_seeds'])
    expected = []
    for row in windows()['windows']['train']:
        for start in row['starts']:
            index = len(expected)
            seeds = known.get((row['trajectory'], start), [52000000 + 3 * index + i for i in range(3)])
            expected.append(dict(trajectory=row['trajectory'], start=start, Re=row['Re'], noise_seeds=seeds,
                                 episode_index=index))
    assert entries['train'] == expected
    assert len(expected) == 183 and len({x['trajectory'] for x in expected}) == 46
    assert all(entries[split] == prior[split] for split in ('development', 'test'))
    seeds = [seed for row in expected for seed in row['noise_seeds']]
    assert len(seeds) == len(set(seeds))
    return len(known)


def fit_probes(cache, cases, output):
    matrices, targets, trajectory_ids, receipts = [], [], [], []
    max_identity, max_half_identity = 0., 0.
    for index, entry in enumerate(cases):
        arrays, receipt, _ = cache.case('train', index, entry)
        receipts.append(receipt)
        for window in (2, 3):
            grid = points(entry['episode_index'], window)
            prev = arrays['observed'][window - 1]
            previous_mean = arrays['mean'][window - 2]
            residual = prev - previous_mean
            identity = np.max(np.abs(prev.astype(np.float64) - residual - previous_mean))
            half_identity = np.max(np.abs((arrays['first8'][window - 2].astype(np.float64)
                                          + arrays['last8'][window - 2]) / 2 - previous_mean))
            max_identity = max(max_identity, float(identity))
            max_half_identity = max(max_half_identity, float(half_identity))
            matrices.append(features(prev, arrays['mean'][window - 1], arrays['std'][window - 1],
                                     residual, arrays['observed'][window - 2], grid))
            targets.append(labels(arrays['observed'][window], arrays['mean'][window - 1], grid))
            trajectory_ids.extend([entry['trajectory']] * len(grid))
        if (index + 1) % 30 == 0:
            print(f'A: cached {index + 1}/{len(cases)} training episodes', flush=True)
    assert len(set(trajectory_ids)) == 46
    assert max_identity < 2e-6 and max_half_identity < 2e-6
    x, y = np.concatenate(matrices), np.concatenate(targets)
    probe = fit(x, y, trajectory_ids)
    write_receipts(output / 'train_native_receipts.jsonl.gz', receipts)
    np.savez_compressed(output / 'probes.npz', **probe.arrays())
    return probe, dict(episodes=len(cases), trajectories=len(set(trajectory_ids)), grid_rows=len(x),
                       feature_slots=x.shape[1], targets=y.shape[1], cache_bytes=sum(x['cache_bytes'] for x in receipts),
                       reconstruction_max_abs=max_identity, half_mean_identity_max_abs=max_half_identity)


def evaluate(cache, split, cases, output, probe):
    _, _, previous, _, _ = paths()
    old = previous_native_events(previous / 'publication.jsonl.gz', split)

    def predict(matrices):
        full = matrices['full']
        result = {model: probe.predict(model, full) for model in (*MODELS, *BASELINES)}
        for model in ('AR', 'AHR'):
            for half in ('first8', 'last8'):
                result[f'{model}:{half}'] = probe.predict(model, matrices[half])
        return result

    receipts = []
    rows = {}
    stability_by_track = {}
    max_identity = 0.
    for index, entry in enumerate(cases):
        arrays, receipt, forecasts = cache.case(split, index, entry, predict=predict, archived=old)
        receipts.append(receipt)
        trajectory = entry['trajectory']
        assert trajectory not in rows
        rows[trajectory] = {}
        stability_by_track[trajectory] = {}
        for window in (2, 3):
            truth = arrays['observed'][window]
            mean = arrays['mean'][window - 1]
            target = labels(truth, mean)
            previous = arrays['observed'][window - 1]
            previous_mean = arrays['mean'][window - 2]
            previous_residual = previous - previous_mean
            max_identity = max(max_identity, float(np.max(np.abs(previous.astype(np.float64)
                                                                 - previous_residual - previous_mean))))
            estimates, stability = forecasts[window]
            rows[trajectory][window] = {key: losses(values, target) for key, values in estimates.items()}
            for model in ('AR', 'AHR'):
                diff = estimates[f'{model}:first8'] - estimates[f'{model}:last8']
                stability[f'{model}_signed_split_output_mse'] = float(np.mean(diff[:, :40]**2))
                stability[f'{model}_second_moment_split_output_mse'] = float(np.mean(diff[:, 40:]**2))
            stability_by_track[trajectory][window] = stability
    assert len(rows) == 12 and max_identity < 2e-6
    write_receipts(output / f'{split}_native_receipts.jsonl.gz', receipts)
    metrics = {}
    track_metrics = {}
    for name in (*MODELS, *BASELINES, 'AR:first8', 'AR:last8', 'AHR:first8', 'AHR:last8'):
        track_metrics[name] = {metric: {trajectory: float(np.mean([windows[n][name][metric] for n in (2, 3)]))
                                       for trajectory, windows in rows.items()}
                               for metric in ('signed_mse', 'second_moment_mse')}
        metrics[name] = {key: float(np.mean(list(values.values()))) for key, values in track_metrics[name].items()}
    compare = {}
    relative = {}
    for model in MODELS:
        relative[model] = {baseline: {metric: (metrics[model][metric] - metrics[baseline][metric])
                                     / metrics[baseline][metric] for metric in metrics[model]}
                           for baseline in (*BASELINES, 'A0')}
    for first, second in COMPARISONS:
        compare[f'{first}-{second}'] = {
            short: paired_differences(track_metrics[first][metric], track_metrics[second][metric])
            for short, metric in [('signed', 'signed_mse'), ('second_moment', 'second_moment_mse')]}
    stability_summary = {key: float(np.mean([np.mean([w[n][key] for n in (2, 3)])
                                            for w in stability_by_track.values()]))
                         for key in next(iter(stability_by_track.values()))[2]}
    stability_summary['half_average_identity_max_abs'] = max(
        win['half_average_identity_max_abs'] for windows in stability_by_track.values() for win in windows.values())
    assert stability_summary['half_average_identity_max_abs'] < 2e-6
    stability_summary['split_input_rms'] = stability_summary['split_input_mse']**.5
    stability_summary['full_residual_rms'] = stability_summary['full_residual_mse']**.5
    result = dict(trajectories=12, windows=[2, 3], points_per_window=2048, metrics=metrics,
                  relative_change=relative, comparisons=compare, trajectory_metrics=track_metrics,
                  sampling_stability=stability_summary, source_checks=dict(
                      max_abs_native_previous_mean_reconstruction=max_identity,
                      archived_history_noise_truth_and_zero_correction_match=True,
                      archived_native_backbone_hash_matches=sum(sum(case['archive_identity']) for case in receipts),
                      archived_native_backbone_hash_expected=len(receipts) * 3),
                  cache_bytes=sum(case['cache_bytes'] for case in receipts))
    return result


def development():
    from research.residual_information_study.information_probe.checks import run as self_check
    _, out, _, output, _ = paths()
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'FROZEN_A.json'
    assert not frozen.exists(), 'A development is already frozen'
    self_check()
    entries = read(out / 'A_SCHEDULE.json')
    reused_seed_cases = validate_training_entries(entries)
    cache = cache_controller()
    probe, info = fit_probes(cache, entries['train'], output)
    evaluation = evaluate(cache, 'development', entries['development'], output, probe)
    evaluation['fit'] = dict(**info, seed_cases_from_original_schedule=reused_seed_cases,
                             fallback_noise_seed_cases=len(entries['train']) - reused_seed_cases,
                             sampling_seed='42600000 + 2 * episode_index + (window - 2)',
                             intercept='weighted target mean on training rows', normal_equation_lambda=1.)
    evaluation['executed_native_groups'] = dict(generated=cache.generated_groups, read_from_cache=cache.reused_groups)
    dump(output / 'development_summary.json', evaluation)
    files = ['probes.npz', 'development_summary.json', 'train_native_receipts.jsonl.gz',
             'development_native_receipts.jsonl.gz']
    dump(frozen, dict(utc_ns=time.time_ns(), config_sha256=sha(out / 'FROZEN.json'),
                      schedule_sha256=sha(out / 'A_SCHEDULE.json'), fits=4, models=list(MODELS),
                      hashes={name: sha(output / name) for name in files}))
    print('A: froze four probes and development, native groups:', cache.generated_groups, flush=True)
    return evaluation


def test():
    _, out, _, output, _ = paths()
    final = output / 'summary.json'
    assert not final.exists(), 'A test has already been evaluated'
    gate = read(out / 'FROZEN_SELECTION.json')
    frozen_path = output / 'FROZEN_A.json'
    frozen = read(frozen_path)
    assert sha(frozen_path) == gate['probe_selection_sha256']
    assert sha(out / 'FROZEN.json') == gate['config_sha256'] == frozen['config_sha256']
    assert frozen['schedule_sha256'] == sha(out / 'A_SCHEDULE.json')
    assert frozen['utc_ns'] < gate['utc_ns']
    assert all(sha(output / name) == expected for name, expected in frozen['hashes'].items())
    with np.load(output / 'probes.npz', allow_pickle=False) as learned:
        probe = Probe(**dict(learned))
    entries = read(out / 'A_SCHEDULE.json')['test']
    cache = cache_controller()
    result = evaluate(cache, 'test', entries, output, probe)
    result['executed_native_groups'] = dict(generated=cache.generated_groups, read_from_cache=cache.reused_groups)
    dump(final, dict(development=read(output / 'development_summary.json'), test=result,
                     frozen_selection_sha256=sha(out / 'FROZEN_SELECTION.json'),
                     test_receipts_sha256=sha(output / 'test_native_receipts.jsonl.gz')))
    print('A: scored exploratory test once, native groups:', cache.generated_groups, flush=True)
    return result
