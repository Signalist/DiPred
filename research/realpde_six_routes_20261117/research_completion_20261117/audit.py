"""Read-only integration checks; recompute gains without importing experiment code.

Run after refresh_mechanics.py. Writes only its own integration outputs.
This checks numerical/provenance consistency, not statistical confirmation.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

W = Path(__file__).resolve().parent.parent
OUT = W / 'runs/research_completion_20261117'
ROUTES = ['error_structure', 'lowcost_joint', 'online_signed_state',
          'active_observation', 'compute_allocation', 'goal_task_error',
          'direct_correction_controls']
SPLITS = ['development', 'late', 'future']
checks, gains, hashes = [], [], []


def read(path):
    return json.loads((W / path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def require(condition, label, **details):
    assert bool(condition), (label, details)
    checks.append(dict(check=label, passed=True, **details))


def verify_hash(path, expected, record):
    p = W / path
    actual = sha(p)
    require(actual == expected, 'hash:' + str(path), record=str(record))
    hashes.append(dict(file=str(p.relative_to(W)), sha256=actual, bytes=p.stat().st_size,
                       verified_against=str(record)))


def paired(frame, method, reference, metric='mse', column='method', label='', split=''):
    data = frame.pivot(index='file', columns=column, values=metric)
    a, b = data[method].to_numpy(), data[reference].to_numpy()
    require(len(a) == 12 and np.isfinite(a).all() and np.isfinite(b).all(),
            'trajectory_count:' + label + '/' + split, n=len(a))
    # Independent aggregation; deterministic bootstrap is descriptive, not a new seed trial.
    rng = np.random.default_rng(42)
    ids = rng.integers(0, len(a), (4000, len(a)))
    boot = 100 * (1 - a[ids].mean(1) / b[ids].mean(1))
    row = dict(route=label, split=split, method=method, reference=reference,
               metric=metric, method_mean=float(a.mean()), reference_mean=float(b.mean()),
               gain_percent=float(100 * (1 - a.mean() / b.mean())),
               descriptive_ci_low=float(np.quantile(boot, .025)),
               descriptive_ci_high=float(np.quantile(boot, .975)),
               wins=int((a < b).sum()), trajectories=len(a))
    gains.append(row)
    return row


def compare_scalar(actual, expected, label, tol=1e-7):
    require(abs(actual - expected) < tol, label, absolute_difference=float(abs(actual - expected)))


def provenance():
    # Only latest manifests are treated as current declarations. Old stage hashes remain historical.
    records = {
        'lowcost_joint': 'audit.json', 'compute_allocation': 'provenance.json',
        'online_signed_state': 'provenance.json', 'active_observation': 'completion_manifest.json',
        'goal_task_error': 'completion_manifest.json', 'direct_correction_controls': 'completion_manifest.json',
    }
    for route, filename in records.items():
        root = 'runs/' + route + '_20261117/'
        record = root + filename
        j = read(record)
        for key in ['sources', 'source_files', 'source_hashes']:
            for name, expected in j.get(key, {}).items():
                path = name if (W / name).is_file() else route + '_20261117/' + name
                verify_hash(path, expected, record)
        for name, expected in j.get('protocols', {}).items():
            verify_hash(route + '_20261117/' + name, expected, record)
        if 'protocol_sha256' in j:
            verify_hash(route + '_20261117/PROTOCOL.md', j['protocol_sha256'], record)
    r = 'runs/compute_allocation_20261117/'
    j = read(r + 'provenance.json')
    for key, name in [('model_sha256', 'models.pt'), ('selection_sha256', 'selection.json')]:
        verify_hash(r + name, j[key], r + 'provenance.json')
    r = 'runs/online_signed_state_20261117/'
    j = read(r + 'provenance.json')
    for key, name in [('main_model_sha256', 'models.pt'), ('geometric_model_sha256', 'geometric/model.pt'),
                      ('linear_geometric_model_sha256', 'linear_geometry/model.pt')]:
        verify_hash(r + name, j[key], r + 'provenance.json')
    verify_hash('../Shared/online_residual_subspace.py', j['shared_memory_sha256'], r + 'provenance.json')
    for route, sub in [('active_observation', ''), ('active_observation', 'predictive/'),
                       ('active_observation', 'critic/'), ('goal_task_error', '')]:
        r = 'runs/' + route + '_20261117/' + sub
        j = read(r + 'selection.json')
        name = 'models.pt' if route == 'goal_task_error' else 'model.pt'
        verify_hash(r + name, j['model_sha256'], r + 'selection.json')
    r = 'runs/direct_correction_controls_20261117/'
    j = read(r + 'selection.json')
    for name, expected in j['model_hashes'].items():
        verify_hash(r + name + '.pt', expected, r + 'selection.json')
    for name, expected in j['source_hashes'].items():
        path = ('runs/error_structure_20261117/local/' + name) if name.endswith('.pt') else ('../Shared/' + name)
        verify_hash(path, expected, r + 'selection.json')
    r = 'runs/error_structure_20261117/'
    verify_hash(r + 'deterministic4/model6000.pt', read(r + 'deterministic4/complete.json')['endpoint_sha256'],
                r + 'deterministic4/complete.json')
    verify_hash('error_structure_20261117/structure.py', read(r + 'mechanism_audit.json')['source_sha256'],
                r + 'mechanism_audit.json')
    r = 'runs/lowcost_joint_20261117/conditional_scores/scorefit/'
    verify_hash(r + 'compact_generator.pt', read(r + 'compact_manifest.json')['compact_sha256'], r + 'compact_manifest.json')
    fresh = read('runs/research_completion_20261117/mechanics_refresh.json')
    require(fresh['passed'] and len(fresh['stages']) == 4, 'current_dependence_mechanics')
    verify_hash('lowcost_joint_20261117/study.py', fresh['source_sha256'], 'mechanics_refresh.json')
    verify_hash('lowcost_joint_20261117/audit.py', fresh['audit_source_sha256'], 'mechanics_refresh.json')


def audits():
    names = {
        'error_structure': ['mechanism_audit', 'inference_audit'],
        'lowcost_joint': ['audit', 'inference_audit', 'conditional_scores/scorefit/inference_audit',
                         'common_calibration/development/audit', 'common_calibration/late/audit', 'common_calibration/future/audit'],
        'online_signed_state': ['audit', 'geometric/inference_audit', 'linear_geometry/inference_audit',
                                'probability/audit', 'temporal_intervals/audit', 'geometric_drift/audit'],
        'active_observation': ['audit', 'predictive/audit', 'critic/audit'],
        'compute_allocation': ['accounting_audit', 'dispatch_audit', 'public_api_audit'],
        'goal_task_error': ['audit'], 'direct_correction_controls': ['audit'],
    }
    for route, files in names.items():
        for name in files:
            path = 'runs/' + route + '_20261117/' + name + '.json'
            require(read(path)['passed'], 'recorded_audit:' + path)


def aggregate():
    r = W / 'runs/direct_correction_controls_20261117'
    d = pd.read_csv(r / 'all_trajectories.csv')
    saved = pd.read_csv(r / 'paired_gains.csv')
    for split in SPLITS:
        q = d[d.split == split]
        for ref in ['base', 'rescrct_block', 'rescrct_original']:
            row = paired(q, 'warp_F_last', ref, label='structured', split=split)
            if ref != 'rescrct_original':
                old = saved[(saved.split == split) & (saved.method == 'warp_F_last') & (saved.reference == ref)].iloc[0]
                compare_scalar(row['gain_percent'], old.gain, 'stored_structure_gain')
    roll = pd.read_csv(r / 'rollout_trajectories.csv')
    for method in roll.method.unique():
        paired(roll[roll['mode'] == 'free'], method, 'base', label='free_200', split='continuous')
    # Apply exactly the two predeclared latency constraints, then calibration selection.
    t = pd.read_csv(r / 'timing.csv', index_col=0)
    s = read('runs/direct_correction_controls_20261117/selection.json')
    eligible = [m for m, v in s['models'].items() if v['gain'] > 0 and
                t.loc[m, 'full_ms'] <= 1.1 * t.loc['warp_F_last', 'full_ms'] and
                t.loc[m, 'active_post_ms'] <= 1.1 * t.loc['warp_F_last', 'active_post_ms']]
    winner = min(eligible, key=lambda m: s['models'][m]['calibration_mse'])
    old = read('runs/direct_correction_controls_20261117/budget_selection.json')
    require(set(eligible) == set(old['eligible']) and winner == old['selected'], 'actual_cost_selection', selected=winner)
    r = W / 'runs/goal_task_error_20261117'
    d = pd.read_csv(r / 'all_trajectories.csv')
    selected = read('runs/goal_task_error_20261117/selection.json')['chosen']
    old = pd.read_csv(r / 'paired_gains.csv')
    for split in SPLITS:
        q = d[d.split == split]
        for method in [selected, 'direct/X+P/16', 'task_affine', 'direct/task_stats/8']:
            row = paired(q, method, 'base', metric='score', column='model', label='task_error', split=split)
            expected = old[(old.split == split) & (old.method == method) & (old.reference == 'base')].iloc[0].gain
            compare_scalar(row['gain_percent'], expected, 'stored_task_gain')
        for task in ['kinetic', 'fluctuation', 'enstrophy', 'regional_u']:
            paired(q, selected, 'base', metric=task + '_mse', column='model', label='task_component', split=split)
    for split in SPLITS:
        r = W / 'runs/lowcost_joint_20261117/conditional_scores/scorefit'
        d = pd.read_csv(r / split / 'trajectories.csv')
        for metric in ['es', 'vs', 'functional_crps', 'crps_speed_max']:
            row = paired(d, 'ar__mixed', 'smooth', metric=metric, label='lowcost', split=split)
            if metric != 'crps_speed_max':
                require(row['gain_percent'] >= -2, 'predeclared_macro_quality_gate:' + metric + '/' + split)
        r = W / 'runs/compute_allocation_20261117'
        d = pd.read_csv(r / (split + '_trajectories.csv'))
        q = d[(d['mode'] == 'raw') & (d.action == 'smooth') & (d.quota == .5)]
        old = pd.read_csv(r / 'paired.csv')
        for ref in ['random_expected', 'risk', 'width']:
            row = paired(q, 'benefit', ref, metric='functional_crps', column='policy', label='allocation_half', split=split)
            expected = old[(old.split == split) & (old['mode'] == 'raw') & (old.action == 'smooth') &
                           (old.quota == .5) & (old.baseline == ref)].iloc[0].gain_percent
            compare_scalar(row['gain_percent'], expected, 'stored_allocation_gain')
    d = pd.read_csv(W / 'runs/online_signed_state_20261117/all_trajectories.csv')
    for split in SPLITS:
        q = d[(d.split == split) & (d.scenario == 'natural')]
        for method in ['geo_rolling', 'lin_rolling', 'rw_kalman', 'rnn', 'rolling_bias', 'memory_gated']:
            paired(q, method, 'base', label='online_natural', split=split)
    d = pd.read_csv(W / 'runs/online_signed_state_20261117/probability/all_trajectories.csv')
    for split in SPLITS:
        q = d[(d.split == split) & (d['mode'] == 'raw')]
        paired(q, 'geo_shift', 'base', metric='es', label='online_probability', split=split)
    d = pd.read_csv(W / 'runs/online_signed_state_20261117/geometric_drift/postchange_trajectories.csv')
    for scenario in ['step', 'ramp']:
        paired(d[d.scenario == scenario], 'linear_rolling', 'faulted_base', label='injected_geometry', split=scenario)
    r = W / 'runs/active_observation_20261117'
    d = pd.read_csv(r / 'predictive_and_critic_trajectories.csv')
    old = pd.read_csv(r / 'observation_efficiency.csv')
    for split in SPLITS:
        q = d[(d.split == split) & (d.budget == 8)]
        for method in ['voi', 'critic']:
            paired(q, method, 'risk', label='observation_equal8', split=split)
        q = d[(d.split == split) & (((d.method == 'risk') & (d.budget == 8)) |
                                   ((d.method == 'random_expectation16') & (d.budget == 32)))]
        row = paired(q, 'risk', 'random_expectation16', label='observation_8_vs_32', split=split)
        expected = old[(old.split == split) & (old.method == 'risk') & (old.budget == 8) &
                       (old.reference == 'random_expectation16') & (old.reference_budget == 32)].iloc[0].gain_percent
        compare_scalar(row['gain_percent'], expected, 'stored_observation_efficiency')
        require(row['descriptive_ci_low'] < 0 < row['descriptive_ci_high'], 'no_confirmed_75pct_sensor_savings:' + split)


def requirements():
    frame = pd.read_csv(Path(__file__).with_name('requirements.tsv'), sep='\t')
    require(frame.id.is_unique and len(frame) >= 40, 'requirements_complete_inventory', rows=len(frame))
    for row in frame.to_dict('records'):
        for path in row['evidence'].split(';'):
            require((W / path).is_file(), 'evidence:' + row['id'], path=path)
    lines = ['# 原附件逐项验收', '', '“完成”指在原数据和简单模型约束下完成探索；科学结果单列，不能将验收通过解释为假设成立。', '',
             '| ID | 路线 / 要求 | 探索结果 | 证据 | 结论限制 |', '|---|---|---|---|---|']
    for row in frame.to_dict('records'):
        evidence = '；'.join('[%s](../../%s)' % (Path(p).name, p) for p in row['evidence'].split(';'))
        lines.append('| {id} | {route}：{requirement} | {outcome} | {evidence} | {limit} |'.format(**dict(row, evidence=evidence)))
    (OUT / 'REQUIREMENTS.md').write_text('\n'.join(lines) + '\n')
    frame.to_csv(OUT / 'requirements.csv', index=False)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    provenance()
    audits()
    aggregate()
    requirements()
    pd.DataFrame(gains).to_csv(OUT / 'recomputed_gains.csv', index=False)
    pd.DataFrame(hashes).to_csv(OUT / 'verified_hashes.csv', index=False)
    # Bind the integration result to its actual inputs, including source and historical tables.
    inventory = []
    for route in ROUTES:
        for root in [W / (route + '_20261117'), W / 'runs' / (route + '_20261117')]:
            for p in sorted(root.rglob('*')):
                if p.is_file() and p.suffix in ('.py', '.md', '.json', '.csv'):
                    inventory.append(dict(path=str(p.relative_to(W)), bytes=p.stat().st_size, sha256=sha(p)))
    pd.DataFrame(inventory).to_csv(OUT / 'evidence_inventory.csv', index=False)
    result = dict(passed=True, checks=checks, verified_existing_hashes=len(hashes),
                  recomputed_comparisons=len(gains), evidence_metadata_files=len(inventory),
                  claims='Bounded original-data exploration; not independent statistical confirmation.',
                  historical_records='Early progress/full_goal_complete=false fields and pre-optimization source hashes are preserved snapshots, superseded by the final integration report.',
                  source_sha256=sha(Path(__file__)), requirements_sha256=sha(Path(__file__).with_name('requirements.tsv')))
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print('INTEGRATION_AUDIT', {k: v for k, v in result.items() if k != 'checks'}, flush=True)
