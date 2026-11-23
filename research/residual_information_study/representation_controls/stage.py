"""Train fixed representation controls and evaluate after coordinated selection."""
import gzip
import time

import numpy as np
import torch

from research.observed_residual_feedback.core import Forecast
from research.observed_residual_feedback.reporting import avg, contrast, json_lines, trajectory_values
from research.observed_residual_feedback.run import (
    Observations, dump, frozen_tensors, jsonl, noise_for, optimize, read,
    score_rows, selection, sha, tensor_hash, trainable, write_scores,
)
from research.residual_information_study.common.context import (
    OUT, PREVIOUS, ROOT, load_native, remaining_gpu_seconds, schedule, windows,
)
from research.residual_information_study.representation_controls.checks import check
from research.residual_information_study.representation_controls.model import (
    ARMS, ControlForecast, ControlPublication, fit_update,
)

DEST = OUT / 'representation_controls'
STEPS = (0, 250, 500)
COMPARISONS = (('R_raw', 'C0'), ('CH', 'R_raw'), ('CS', 'R_raw'))


def archive_path(name, manifest):
    path = PREVIOUS / name
    assert sha(path) == manifest['artifact_sha256'][str(path.relative_to(ROOT))], name
    return path


def archive_manifest():
    manifest = read(PREVIOUS / 'VERIFICATION.json')
    for key in ('passed', 'weights_verified', 'own_revealed_state_chain_verified',
                'test_selection_precedes_issue'):
        assert manifest[key], key
    return manifest


def previous_identity(native, plan):
    """Verify the comparison model without evaluating or reading any test scores."""
    manifest = archive_manifest()
    config_path = archive_path('FROZEN.json', manifest)
    config = read(config_path)
    for path, checksum in config['sha256'].items():
        assert sha(ROOT / path) == checksum, path
    assert config['seed'] == 42 and config['members'] == 16 and config['updates_per_arm'] == 500
    assert config['candidates'] == list(STEPS)
    assert config['lr_backbone'] == 1e-5 and config['lr_new'] == 1e-4
    assert config['clip_grad_norm'] == 1.
    assert read(archive_path('schedule.json', manifest)) == plan
    assert read(archive_path('windows.json', manifest)) == windows()
    completed = read(archive_path('RUN.json', manifest))
    assert completed['common_updates'] == 500 and completed['frozen_backbone_unchanged']
    assert completed['native_calls_per_forecast'] == 4
    selected = read(archive_path('SELECTION.json', manifest))
    assert selected['selected']['J'] == 500 and selected['config_sha256'] == sha(config_path)
    model = Forecast(native, 'J')
    initialization = {key: tensor_hash(value) for key, value in trainable(model).items()}
    old_train = read(archive_path('TRAIN_STARTED.json', manifest))
    assert old_train['parameters']['J'] == dict(
        trainable=sum(value.numel() for value in trainable(model).values()),
        total=sum(value.numel() for value in model.parameters()))
    for record, step in zip(completed['weights']['J'], STEPS):
        path = archive_path(f'weights/J_{step}.pt', manifest)
        assert record['path'] == str(path.relative_to(ROOT)) and record['sha256'] == sha(path)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        assert (ck['arm'], ck['step'], ck['config_sha256']) == ('J', step, sha(config_path))
        assert set(ck['state']) == set(initialization)
        if step == 0:
            assert initialization == {key: tensor_hash(value) for key, value in ck['state'].items()}
    updates = [event for event in json_lines(archive_path('training.jsonl.gz', manifest))
               if event['arm'] == 'J']
    assert len(updates) == 500
    for step, (row, entry) in enumerate(zip(updates, plan['train']), 1):
        assert row['step'] == step and len(row['warm_losses']) == 2
        assert all(row[key] == value for key, value in entry.items())
    dump(DEST / 'RAW_IDENTITY.json', dict(
        prior_J_checkpoint=completed['weights']['J'][-1], prior_updates=len(updates),
        prior_frozen_hash=sha(config_path), archived_step_zero_exact=True,
        schedule_and_windows_identical=True, loss_and_model_source_hashes_verified=True,
        native_calls_per_forecast=4, parameters=old_train['parameters']['J']))
    return initialization


def read_archive_scores(split):
    manifest = archive_manifest()
    if split == 'development':
        files = [('development_0.jsonl.gz', 'N', 'N'), ('development_500.jsonl.gz', 'J', 'R_raw')]
        return [dict(row, arm=arm) for file, name, arm in files
                for row in json_lines(archive_path(file, manifest)) if row['arm'] == name]
    archive = json_lines(archive_path('test_selected.jsonl.gz', manifest))
    labels = {'N': 'N', 'J': 'R_raw'}
    return [dict(row, arm=labels[row['arm']]) for row in archive if row['arm'] in labels]


def archive_publications(split, plan, scores):
    """Authenticate archive identity and historical issue chains for N and J500."""
    verified = {}
    manifest = archive_manifest()
    entries = {row['trajectory']: row for row in plan[split]}
    seen = set()
    for event in json_lines(archive_path('publication.jsonl.gz', manifest)):
        if event['split'] != split or event['arm'] not in ('N', 'J'):
            continue
        wanted = 'selected' if split == 'test' else (0 if event['arm'] == 'N' else 500)
        if event['stage'] != wanted:
            continue
        name = event['trajectory']
        arm = 'R_raw' if event['arm'] == 'J' else 'N'
        entry = entries[name]
        n = event['window']
        assert (event['start'], event['Re'], event['seed']) == (
            entry['start'], entry['Re'], entry['noise_seeds'][n - 1])
        key = (name, n, arm)
        token = (*key, event['event'])
        assert token not in seen
        seen.add(token)
        verified.setdefault(key, {})[event['event']] = event
    expected = {(name, n, arm) for name in entries for n in (1, 2, 3) for arm in ('N', 'R_raw')}
    assert expected == set(verified)
    selected = read(archive_path('SELECTION.json', manifest))
    if split == 'test':
        assert all(selected['utc_ns'] < pair['issued']['utc_ns'] for pair in verified.values())
    shared_inputs = {}
    for (name, n, arm), pair in verified.items():
        issue, reveal = pair['issued'], pair['revealed']
        assert issue['utc_ns'] < reveal['utc_ns'] and issue['availability'] == (n > 1)
        if n > 1:
            prior = verified[(name, n - 1, arm)]['revealed']
            assert issue['history_sha256'] == prior['truth_sha256']
            assert issue['previous_residual_sha256'] == prior['residual_sha256']
            assert issue['previous_correction_sha256'] == prior['mean_correction_sha256']
        hashes = {key: issue[key] for key in ('history_sha256', 'noise_sha256')}
        hashes['truth_sha256'] = reveal['truth_sha256']
        assert shared_inputs.setdefault((name, n), hashes) == hashes
    identities = {(r['trajectory'], r['window'], r['arm'], r['field'], r['channel'], r['lead'])
                  for r in scores}
    expected_scores = {(*key, field, channel, lead) for key in expected
                       for field in ('backbone', 'final') for channel in ('overall', 'u', 'v')
                       for lead in range(21)}
    assert len(scores) == len(identities) and identities == expected_scores
    for row in scores:
        entry = entries[row['trajectory']]
        event = verified[(row['trajectory'], row['window'], row['arm'])]['issued']
        assert (row['split'], row['stage'], row['start'], row['Re'], row['seed']) == (
            split, event['stage'], entry['start'], entry['Re'], event['seed'])
    return shared_inputs


def config_sha():
    return sha(OUT / 'FROZEN.json')


def save(model, arm, step):
    path = DEST / 'weights' / f'{arm}_{step}.pt'
    assert not path.exists() and step in STEPS
    path.parent.mkdir(exist_ok=True)
    torch.save(dict(arm=arm, step=step, config_sha256=config_sha(),
                    state={key: value.detach().cpu() for key, value in trainable(model).items()}), path)
    return dict(step=step, path=str(path.relative_to(ROOT)), sha256=sha(path))


def restore(native, arm, item):
    path = ROOT / item['path']
    assert sha(path) == item['sha256']
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert (ck['arm'], ck['step'], ck['config_sha256']) == (arm, item['step'], config_sha())
    model = ControlForecast(native)
    assert set(ck['state']) == set(trainable(model))
    model.load_state_dict(ck['state'], strict=False)
    return model.eval().requires_grad_(False)


def publication_event(common, **fields):
    jsonl(DEST / 'publication.jsonl', dict(**common, **fields))


@torch.no_grad()
def evaluate(models, split, stage, entries, norm, data_root, expected):
    method_rows = {arm: [] for arm in models}
    for entry in entries:
        obs = Observations(entry, norm, data_root)
        history = obs.read_window(0)
        streams = {arm: ControlPublication(history, arm) for arm in models}
        for n, seed in enumerate(entry['noise_seeds'], 1):
            if not streams:
                break
            noise = noise_for(history, seed)
            expected_hashes = expected[(entry['trajectory'], n)]
            noise_sha = tensor_hash(noise)
            assert noise_sha == expected_hashes['noise_sha256']
            common = dict(split=split, stage=stage, trajectory=entry['trajectory'], Re=entry['Re'],
                          start=entry['start'], window=n, seed=seed)
            issued = {}
            for arm, model in models.items():
                stream = streams[arm]
                assert not model.training and not any(p.requires_grad for p in model.parameters())
                h_sha = tensor_hash(stream.history)
                assert h_sha == expected_hashes['history_sha256']
                encoded = stream.encoded_memory()
                backbone, correction = stream.issue(model, noise)
                issued[arm] = (backbone, correction)
                publication_event(common, arm=arm, event='issued', utc_ns=time.time_ns(),
                    history_sha256=h_sha, noise_sha256=noise_sha, availability=encoded.available,
                    backbone_sha256=tensor_hash(backbone), correction_sha256=tensor_hash(correction),
                    previous_residual_sha256=tensor_hash(stream.memory.residual),
                    previous_correction_sha256=tensor_hash(stream.memory.correction),
                    first_slot_sha256=tensor_hash(encoded.first),
                    second_slot_sha256=tensor_hash(encoded.second),
                    first_slot_semantics={'C0': 'zero', 'CH': 'older_observed_history',
                                          'CS': 'spatial_mean_previous_residual'}[arm])
            truth = obs.read_window(n)
            truth_sha = tensor_hash(truth)
            assert truth_sha == expected_hashes['truth_sha256']
            reveal_ns = time.time_ns()
            for arm, (backbone, correction) in issued.items():
                stream = streams[arm]
                stream.reveal(truth)
                publication_event(common, arm=arm, event='revealed', utc_ns=reveal_ns,
                    truth_sha256=truth_sha, residual_sha256=tensor_hash(stream.memory.residual),
                    mean_correction_sha256=tensor_hash(stream.memory.correction))
                for field, prediction in [('backbone', backbone), ('final', backbone + correction)]:
                    method_rows[arm].extend(dict(**common, arm=arm, field=field, **metrics)
                                           for metrics in score_rows(prediction, truth, norm))
    merged = []
    for arm, rows in method_rows.items():
        assert len(rows) == 12 * 3 * 2 * 3 * 21
        path = DEST / f'{split}_{stage}_{arm}.jsonl.gz'
        assert not path.exists()
        write_scores(path, rows)
        merged.extend(rows)
    return merged


def copy_log(name):
    path = DEST / f'{name}.jsonl'
    if path.exists():
        compressed_path = DEST / f'{name}.jsonl.gz'
        plain = path.read_bytes()
        with compressed_path.open('wb') as target:
            with gzip.GzipFile(fileobj=target, mode='wb', mtime=0) as compressed:
                compressed.write(plain)
        assert gzip.decompress(compressed_path.read_bytes()) == plain
        path.unlink()


def warm_value(rows, arm):
    return selection([row for row in rows if row['arm'] == arm])


def development():
    DEST.mkdir(parents=True, exist_ok=True)
    assert not (DEST / 'TRAIN_STARTED.json').exists(), 'Formal representation run already started.'
    check()
    native, norm, cfg = load_native()
    plan = schedule()
    initial = previous_identity(native, plan)
    archived = read_archive_scores('development')
    observed = archive_publications('development', plan, archived)
    baseline = warm_value(archived, 'N')
    raw_baseline = warm_value(archived, 'R_raw')
    old_selected = read(PREVIOUS / 'SELECTION.json')
    assert abs(raw_baseline - dict((step, value) for value, step in
                                 old_selected['candidates']['J'])[500]) < 1e-14
    models = {arm: ControlForecast(native).train() for arm in ARMS}
    assert all(initial == {key: tensor_hash(p) for key, p in trainable(model).items()}
               for model in models.values())
    frozen = {arm: frozen_tensors(model) for arm, model in models.items()}
    optimizers = {arm: optimize(model) for arm, model in models.items()}
    paths = {arm: [save(model, arm, 0)] for arm, model in models.items()}
    dump(DEST / 'TRAIN_STARTED.json', dict(utc_ns=time.time_ns(), config_sha256=config_sha(),
        schedule_sha256=sha(PREVIOUS / 'schedule.json'), seed=42, arms=ARMS,
        same_J0_initialization=True, augmented_parameters=sum(p.numel() for p in trainable(models['C0']).values())))

    candidates = {arm: [(baseline, 0)] for arm in ARMS}
    common = 0
    slowest = 1.
    for index, entry in enumerate(plan['train'], 1):
        # Preserve the common-step design and reserve time for final evaluation.
        reserve = 120. + (90. if index in STEPS else 0.)
        if remaining_gpu_seconds() <= reserve + 2 * slowest:
            break
        started = time.monotonic()
        for arm in ARMS:
            losses, grad = fit_update(arm, models[arm], optimizers[arm],
                                      Observations(entry, norm, cfg['data_root']))
            jsonl(DEST / 'training.jsonl', dict(step=index, arm=arm, warm_losses=losses,
                gradient_norm=grad, utc_ns=time.time_ns(), **entry))
        slowest = max(slowest, time.monotonic() - started)
        common = index
        if index % 25 == 0:
            print('C common updates', index, 'remaining GPU seconds', round(remaining_gpu_seconds()), flush=True)
        if index in STEPS:
            for arm, model in models.items():
                paths[arm].append(save(model, arm, index))
            rows = evaluate({arm: restore(native, arm, paths[arm][-1]) for arm in ARMS},
                            'development', index, plan['development'], norm, cfg['data_root'], observed)
            for arm in ARMS:
                candidates[arm].append((warm_value(rows, arm), index))

    changes = {}
    for arm, model in models.items():
        assert frozen[arm] == frozen_tensors(model)
        changes[arm] = [key for key, p in trainable(model).items() if tensor_hash(p) != initial[key]]
        if common:
            assert all(any(key.startswith(group) for key in changes[arm])
                       for group in ('base.net.up2', 'base.net.out', 'encoder', 'projection', 'head'))
    selected = {arm: min(candidates[arm])[1] for arm in ARMS}
    dump(DEST / 'SELECTION.json', dict(utc_ns=time.time_ns(), config_sha256=config_sha(),
        selected=selected, candidates=candidates, archived_R_raw_step=500,
        archived_development=dict(N=baseline, R_raw=raw_baseline)))
    copy_log('training')
    dump(DEST / 'RUN.json', dict(config_sha256=config_sha(), common_updates=common, weights=paths,
        stop='completed' if common == 500 else 'GPU budget guard', frozen_parameters_identical=True,
        changed_parameters=changes, cache_bytes=0,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
    print('C fixed training ended', common, 'selected', selected, flush=True)


METRICS = ('score_u', 'score_v', 'crps_u', 'crps_v', 'es_u', 'es_v', 'mse', 'smse',
           'variance', 'standardized_variance', 'width90', 'residual_mae',
           'residual_signed_mean', 'residual_physical_mae', 'residual_physical_signed_mean')
SCORED_ARMS = ('N', 'R_raw', *ARMS)


def summarize(rows):
    summary = {}
    for phase, physical_windows in [('warm', (2, 3)), ('cold', (1,))]:
        summary[phase] = {}
        for field in ('backbone', 'final'):
            summary[phase][field] = {}
            for metric in METRICS:
                values = trajectory_values(rows, SCORED_ARMS, physical_windows, field, metric)
                summary[phase][field][metric] = dict(
                    means={arm: avg(values, arm) for arm in SCORED_ARMS},
                    contrasts={f'{first}-{second}': contrast(values, first, second)
                               for first, second in COMPARISONS})
    return summary


def exploration(rows):
    """Descriptive differences by channel, lead and Re, with trajectory weighting."""
    values = {}
    for row in rows:
        if row['window'] == 1 or row['field'] != 'final':
            continue
        for group in ('all', str(row['Re'])):
            key = (group, row['channel'], row['lead'])
            record = values.setdefault(key, {}).setdefault(row['trajectory'], {})
            record.setdefault(row['arm'], []).append(row)
    strata = []
    for key, trajectories in sorted(values.items()):
        lead = key[2]
        metrics = ('crps_u', 'crps_v', 'mse', 'smse', 'variance')
        if not lead:
            metrics += ('score_u', 'score_v', 'es_u', 'es_v', 'width90')
        assert all(set(methods) == set(SCORED_ARMS) for methods in trajectories.values())
        assert all(len(windows) == 2 for methods in trajectories.values() for windows in methods.values())
        for metric in metrics:
            means = {arm: float(np.mean([np.mean([r[metric] for r in methods[arm]])
                                         for methods in trajectories.values()])) for arm in SCORED_ARMS}
            strata.append(dict(Re=key[0], channel=key[1], lead=lead, metric=metric, means=means,
                contrasts={f'{a}-{b}': means[a] - means[b] for a, b in COMPARISONS},
                n=len(trajectories)))
    return strata


def verify_fresh_publications(plan, completed, selected, gate, observed):
    logs = json_lines(DEST / 'publication.jsonl') if (DEST / 'publication.jsonl').exists() else []
    pairs = {}
    for row in logs:
        key = (row['split'], row['stage'], row['trajectory'], row['window'], row['arm'])
        pair = pairs.setdefault(key, {})
        assert row['event'] not in pair
        pair[row['event']] = row
    expected = set()
    for step in (250, 500):
        if step <= completed:
            expected.update(('development', step, e['trajectory'], n, arm)
                            for e in plan['development'] for n in (1, 2, 3) for arm in ARMS)
    expected.update(('test', 'selected', e['trajectory'], n, arm)
                    for e in plan['test'] for n in (1, 2, 3) for arm in ARMS if selected[arm])
    assert set(pairs) == expected
    zero_sha = tensor_hash(torch.zeros((2, 20, 32, 64), dtype=torch.float32))
    for key, pair in pairs.items():
        split, stage, trajectory, window, arm = key
        issue, reveal = pair['issued'], pair['revealed']
        assert issue['utc_ns'] < reveal['utc_ns']
        assert issue['availability'] == (window > 1)
        if split == 'test':
            assert gate['utc_ns'] <= issue['utc_ns']
        source = observed[split][(trajectory, window)]
        for field in ('history_sha256', 'noise_sha256'):
            assert issue[field] == source[field]
        assert reveal['truth_sha256'] == source['truth_sha256']
        if window > 1:
            prior = pairs[(split, stage, trajectory, window - 1, arm)]
            assert issue['history_sha256'] == prior['revealed']['truth_sha256']
            assert issue['previous_residual_sha256'] == prior['revealed']['residual_sha256']
            assert issue['previous_correction_sha256'] == prior['revealed']['mean_correction_sha256']
            if arm == 'CH':
                assert issue['first_slot_sha256'] == prior['issued']['history_sha256']
        if arm == 'C0' or window == 1:
            assert issue['first_slot_sha256'] == zero_sha
        if arm != 'CS' or window == 1:
            assert issue['second_slot_sha256'] == zero_sha
    return len(pairs)


def test():
    gate = read(OUT / 'FROZEN_SELECTION.json')
    selected_path = DEST / 'SELECTION.json'
    chosen = read(selected_path)
    selected = chosen['selected']
    assert gate['config_sha256'] == chosen['config_sha256'] == config_sha()
    assert gate['representation_selection_sha256'] == sha(selected_path)
    assert gate['representation_selected'] == selected
    assert chosen['utc_ns'] <= gate['utc_ns']
    assert not (DEST / 'TEST_STARTED.json').exists(), 'Representation test already started.'
    run = read(DEST / 'RUN.json')
    assert run['config_sha256'] == config_sha() and run['frozen_parameters_identical']
    plan = schedule()
    # Evidence that all logged updates follow the frozen schedule.
    train_path = DEST / 'training.jsonl.gz'
    updates = json_lines(train_path) if train_path.exists() else []
    assert len(updates) == len(ARMS) * run['common_updates']
    for index, row in enumerate(updates):
        i, j = divmod(index, len(ARMS))
        assert row['step'] == i + 1 and row['arm'] == ARMS[j]
        assert all(row[key] == value for key, value in plan['train'][i].items())
        assert len(row['warm_losses']) == 2 and np.isfinite(row['warm_losses']).all()

    development = read_archive_scores('development')
    dev_observed = archive_publications('development', plan, development)
    baseline = warm_value(development, 'N')
    verified_candidates = {}
    for arm in ARMS:
        computed = [(baseline, 0)]
        for item in run['weights'][arm]:
            assert sha(ROOT / item['path']) == item['sha256']
            step = item['step']
            if step:
                rows = json_lines(DEST / f'development_{step}_{arm}.jsonl.gz')
                computed.append((warm_value(rows, arm), step))
        assert [list(item) for item in computed] == chosen['candidates'][arm]
        assert min(computed)[1] == selected[arm]
        verified_candidates[arm] = computed

    native, norm, cfg = load_native()
    dump(DEST / 'TEST_STARTED.json', dict(utc_ns=time.time_ns(),
        selection_sha256=sha(OUT / 'FROZEN_SELECTION.json'), selected=selected))
    archived = read_archive_scores('test')
    test_observed = archive_publications('test', plan, archived)
    models = {}
    aliases = []
    for arm in ARMS:
        if selected[arm]:
            item = next(item for item in run['weights'][arm] if item['step'] == selected[arm])
            models[arm] = restore(native, arm, item)
        else:
            aliases.append(arm)
    new = evaluate(models, 'test', 'selected', plan['test'], norm, cfg['data_root'], test_observed)
    rows = archived + new
    rows.extend(dict(row, arm=arm) for arm in aliases for row in archived if row['arm'] == 'N')
    assert len(rows) == len(SCORED_ARMS) * 12 * 3 * 2 * 3 * 21
    issued = verify_fresh_publications(plan, run['common_updates'], selected, gate,
                                      dict(development=dev_observed, test=test_observed))
    summary = summarize(rows)
    summary.update(selected={**selected, 'R_raw': 500, 'N': 0},
                   comparable_500_step_completion=run['common_updates'] == 500,
                   updates_per_arm=run['common_updates'], candidates=verified_candidates,
                   publications=issued, archive_reused=['N', 'R_raw'], zero_step_aliases=aliases,
                   new_test_forward_groups=len(models) * 12 * 3,
                   test_rows=len(rows), protocol_sha256=config_sha(),
                   bootstrap=dict(seed=123, replicates=10000, unit='trajectory'),
                   selection_sha256=sha(OUT / 'FROZEN_SELECTION.json'))
    dump(DEST / 'summary.json', summary)
    write_scores(DEST / 'test_exploration.jsonl.gz', exploration(rows))
    copy_log('publication')
    print('C verified fresh publications', issued, 'and test groups', summary['new_test_forward_groups'],
          flush=True)
