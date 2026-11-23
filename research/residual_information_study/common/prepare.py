"""Freeze the datasets, budgets, random schedules and initial dependencies."""
import gzip
import json

import h5py

from research.observed_residual_feedback import checks, run as observed
from .context import CACHE, CACHE_BYTES, GPU_SECONDS, OUT, PREVIOUS, REFERENCE, ROOT, SOURCE, read, schedule, sha, windows


def prepare():
    assert observed.git('rev-parse', 'HEAD') == REFERENCE
    assert not (OUT / 'FROZEN.json').exists(), 'The fixed protocol is already present'
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    previous = read(PREVIOUS / 'FROZEN.json')
    previous_sha256 = sha(PREVIOUS / 'FROZEN.json')
    previous_run = read(PREVIOUS / 'RUN.json')
    previous_selection = read(PREVIOUS / 'SELECTION.json')
    assert previous_run['common_updates'] == 500 and previous_selection['selected']['J'] == 500
    assert previous_selection['config_sha256'] == previous_sha256
    for name, digest in previous['sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert previous['seed'] == 42 and previous['members'] == 16 and previous['candidates'] == [0, 250, 500]
    seq, listed = schedule(), windows()['windows']
    assert {split: len(rows) for split, rows in listed.items()} == dict(train=46, development=12, test=12)
    records = read(ROOT / 'results/residual_dynamics/protocol.json')['records']
    manifest = read(ROOT / 'metadata/manifest.json')['files']
    groups = []
    data = ROOT.__class__(previous['data_root'])
    for split, rows in listed.items():
        groups.append({row['Re'] for row in rows})
        files = {row['file']: row for row in records[split]}
        assert set(files) == {row['trajectory'] for row in rows}
        for row in rows:
            original = files[row['trajectory']]
            assert row['Re'] == original['group'] and original in manifest[split]
            with h5py.File(data / row['trajectory'], 'r') as handle:
                assert all(len(handle[key]) == original['frames'] for key in 'uv')
            eligible = (200, 300, 400, 500) if split == 'train' else (400,)
            assert row['starts'] == [start for start in eligible if start + 80 <= original['frames']]
        legal = {(row['trajectory'], start) for row in rows for start in row['starts']}
        actual = {(row['trajectory'], row['start']) for row in seq[split]}
        assert actual <= legal
        if split != 'train':
            assert actual == legal and len(seq[split]) == 12
    assert all(not group & other for i, group in enumerate(groups) for other in groups[:i])
    assert len(seq['train']) == 500
    seeds = [seed for rows in seq.values() for row in rows for seed in row['noise_seeds']]
    assert len(seeds) == len(set(seeds))
    first = {}
    for row in seq['train']:
        first.setdefault((row['trajectory'], row['start']), row)
    train_A = []
    for row in listed['train']:
        for start in row['starts']:
            k = len(train_A)
            item = first.get((row['trajectory'], start))
            replacement = list(range(52000000 + 3*k, 52000000 + 3*k + 3))
            train_A.append(dict(trajectory=row['trajectory'], Re=row['Re'], start=start,
                                episode_index=k, noise_seeds=item['noise_seeds'] if item else replacement))
    assert len(train_A) == 183
    all_a_seeds = [seed for item in train_A for seed in item['noise_seeds']]
    assert len(all_a_seeds) == len(set(all_a_seeds))
    eligible_B = [row for row in listed['train'] if 400 in row['starts']][:4]
    legal_A = {(row['trajectory'], row['start']): row for row in train_A}
    train_B = [first.get((row['trajectory'], 400), legal_A[(row['trajectory'], 400)])
               for row in eligible_B]
    train_B = [{k: v for k, v in row.items() if k != 'episode_index'} for row in train_B]
    assert len(train_B) == 4
    observed.dump(OUT / 'A_SCHEDULE.json', dict(seq, train=train_A))
    observed.dump(OUT / 'B_GRADIENT_SCHEDULE.json', train_B)
    checks.numerical_checks()
    checks.publication_check()
    dependencies = [PREVIOUS / file for file in ('schedule.json', 'windows.json', 'FROZEN.json',
                    'RUN.json', 'VERIFICATION.json', 'SELECTION.json', 'PREFLIGHT.json', 'TRAIN_STARTED.json',
                    'DECISION_REPORT.md', 'weights/J_0.pt', 'weights/J_500.pt', 'weights/L_500.pt',
                    'old_audit_repaired.csv', 'old_audit_repaired_paired.json', 'training.jsonl.gz')]
    inputs = ['PROTOCOL.md', 'A_SCHEDULE.json', 'B_GRADIENT_SCHEDULE.json']
    paths = dependencies + [OUT / item for item in inputs] + list((SOURCE / 'common').glob('*.py'))
    checksums = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    archive = read(PREVIOUS / 'VERIFICATION.json')['artifact_sha256']
    assert all(checksums[name] == archive[name] for name in set(checksums) & set(archive))
    observed.dump(OUT / 'FROZEN.json', dict(reference=REFERENCE, actual_head=observed.git('rev-parse', 'HEAD'),
        data_root=str(data), members=16, seed=42, windows=dict(train=46, development=12, test=12),
        neural_updates=500, neural_max_arms=3, candidates=[0, 250, 500], ridge_max_models=4,
        gpu_budget_seconds=GPU_SECONDS, cache_budget_bytes=CACHE_BYTES, B_group_limit=120,
        selection='warm windows 2 and 3, trajectory equal final S_U; earliest exact tie',
        uncertainty=dict(unit='trajectory', repeats=10000, seed=123),
        archive_expected_sha256=archive, sha256=checksums))
    observed.dump(OUT / 'SHARED_CHECKS.json', dict(passed=True, explicit_U_V_member_pairs=True,
        original_normalized_identity=True, channelwise_correct_scale=True,
        issuance_before_reveal=True, train_rows=500, legal_A_train_fragments=len(train_A),
        reused_A_train_first_occurrence=len(first), B_train_fragments=len(train_B),
        config_sha256=sha(OUT / 'FROZEN.json')))
    print('Frozen protocol, shared window identities, and scoring checks passed', flush=True)
