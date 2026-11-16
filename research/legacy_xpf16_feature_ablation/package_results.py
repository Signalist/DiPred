"""Package completed experiments and their provenance; fail on incomplete data."""
import ast
import hashlib
import json
import os
import shutil
import sys
import subprocess
import tarfile
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from clean_reliability_common import manifest
from experiment_reliability_transport_e12 import normalization
RUN = ROOT / 'runs/legacy_xpf16_feature_ablation_20261116'
REPO = ROOT / 'github/DiPred'
DEST = REPO / 'results/reliability/legacy_xpf16_feature_ablation'
CODE = REPO / 'research/legacy_xpf16_feature_ablation'
ARMS = ['X', 'P', 'F', 'XP', 'XF', 'PF', 'XPF']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    trained = json.loads((RUN / 'training_complete.json').read_text())
    assert trained['complete'] and len(trained['weights']) == 294
    assert all((RUN / 'evaluation' / arm / 'run.json').exists() for arm in ARMS)
    audit = json.loads((RUN / 'publication/replay_audit.json').read_text())
    assert audit['xpf_replay_max_abs_diff'] < 2e-5
    checks = json.loads((RUN / 'publication/audit_checks.json').read_text())
    assert checks['passed']
    subprocess.run([sys.executable, str(ROOT / 'legacy_xpf16_feature_ablation/benchmark.py')], cwd=ROOT, check=True, env=dict(os.environ, TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0'))
    DEST.mkdir(parents=True, exist_ok=False)
    CODE.mkdir(parents=True, exist_ok=False)
    for source in (ROOT / 'legacy_xpf16_feature_ablation').glob('*.py'):
        shutil.copy2(source, CODE / source.name)
    shutil.copy2(ROOT / 'legacy_xpf16_feature_ablation/run_after_training.sh', CODE)
    shutil.copy2(ROOT / 'legacy_xpf16_feature_ablation/FEATURE_SOURCES.md', CODE)
    shutil.copy2(ROOT / 'legacy_xpf16_feature_ablation/REPRODUCE.md', CODE)
    # Archive the statically referenced legacy workspace modules as well.
    dependencies = CODE / 'workspace_dependencies'
    dependencies.mkdir()
    pending = ['clean_reliability_common', 'experiment_selected16_quantile',
               'experiment_reliability_transport_e12', 'temporal_mean_memory_control',
               'experiment_fullblock_memory_risk', 'selected16_quantile_head',
               'export_aligned_test_distribution', 'xpf_crps_metrics']
    seen = set()
    while pending:
        name = pending.pop()
        source = ROOT / (name + '.py')
        if name in seen or not source.exists():
            continue
        seen.add(name)
        tree = ast.parse(source.read_text())
        shutil.copy2(source, dependencies / (name + '.py'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                pending.append(node.module.split('.')[0])
            elif isinstance(node, ast.Import):
                pending.extend(alias.name.split('.')[0] for alias in node.names)
    for source in (RUN / 'publication').iterdir():
        shutil.copy2(source, DEST / source.name)
    for name in ['historical_recovery.json', 'training_complete.json', 'historical_49_windows.csv', 'historical_250_windows.csv', 'historical_noise_timing.csv', 'feature_test.log', 'score_test.log', 'restore_250.log', 'protocol250_audit.json']:
        shutil.copy2(RUN / name, DEST / name)
    shutil.copy2(ROOT / 'runs/legacy_xpf16_feature_ablation_train.log', DEST / 'training_execution.log')
    old = ROOT / 'runs/xpf16_fullpercent_20261111/models'
    old_weights = {str(p.relative_to(old)): sha(p) for p in sorted(old.glob('XPF16-Q*/model.pt'))}
    assert len(old_weights) == 49
    datafiles = {}
    for protocol, folder in [('49', ROOT / 'runs/xpf16_fullpercent_20261111/evaluation/test'),
                             ('250', ROOT / 'runs/xpf16_02grid_2048_20261112/evaluation/test')]:
        for start in range(20, 821, 20):
            f = folder / f'{start:04d}.npz'
            datafiles[f'{protocol}/{f.name}'] = sha(f)
    protocol = dict(baseline_main_commit='f15759823acd9c7bbdb02653e50cf4fe12fafcce', head_parameters=34004, heads_per_arm=49, training_steps=2000,
        training_windows=819, batch_size=4, seed=42, optimizer='AdamW; lr 3e-4 cosine to 1e-6; weight decay 1e-5; gradient clip 1',
        torch_version=torch.__version__, cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(0), tf32_cublas_override=0,
        target='Y-P for all seven arms; P is a common output anchor even when not directly input',
        X='historical channelwise standard deviation; t=0 P change also depends on last X',
        P='gradient and temporal difference; P-change at t=0 is neutralized when X is absent',
        F='48-channel actual final denoising feature, original trainable 8-channel projection',
        auxiliaries='two spatial coordinates; masked normalized inputs use zero; F masked after projection',
        prediction='four-step frozen DiPred; entire 20-frame window issued before truth feedback',
        primary_aggregation='12 trajectories and 2 channels equally weighted, 41 windows after cold start',
        normalization_mean=normalization()['mean'].flatten().cpu().tolist(),
        normalization_std=normalization()['std'].flatten().cpu().tolist(),
        crps='original 98 quantiles; sorting, interpolated median, constant outer 1% tails, trapezoid pinball integration',
        noise='actual saved empirical and fair ensemble CRPS; online is legacy proxy-scale; 2048 from separate 250-grid run',
        protocol_limit='XPF uses two-tail online correction; noise uses proxy-scale correction; 2048 is a different run',
        temporal_noise_limit='49-grid archives contain window, channel, and trajectory scores but not per-lead noise scores',
        old_xpf_weights=old_weights, old_xpf_weights_repository='results/reliability/xpf16_independent49/weights/XPF16-Q01.pt through XPF16-Q49.pt', old_xpf_initial_offsets_sha256=sha(ROOT / 'runs/xpf16_fullpercent_20261111/initial_offsets.pt'), historical_window_npz_hashes=datafiles,
        initial_weights_sha256=sha(ROOT / 'runs/selected16_quantile_20261110/compact16_initial.pt'),
        frozen_dipred_checkpoint_sha256=sha(ROOT / 'github/DiPred/reliability/checkpoints/current.pt'),
        selected16_comparison_checkpoint_sha256=sha(ROOT / 'github/DiPred/reliability/checkpoints/selected16_comparison.pt'),
        dynamic_memory_checkpoint_sha256=sha(ROOT / 'github/DiPred/reliability/checkpoints/dynamic_memory.pt'),
        feature_source_sha256=sha(ROOT / 'github/DiPred/reliability/features.py'),
        legacy_head_source_sha256=sha(ROOT / 'selected16_quantile_head.py'))
    (DEST / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (DEST / 'data_split.json').write_text(json.dumps(manifest(), indent=2)+'\n')
    durations = []
    for arm in ARMS:
        for k in range(1, 50):
            path = (old / f'XPF16-Q{k:02d}' / 'model.pt' if arm == 'XPF'
                    else RUN / 'models' / arm / f'Q{k:02d}.pt')
            ck = torch.load(path, map_location='cpu', weights_only=False)
            durations.append(dict(method=arm, k=k, steps=ck['config']['steps'], parameters=ck['parameters'],
                batch_wall_seconds=ck.get('group_seconds', ck.get('elapsed')), model_sha256=sha(path),
                initial_sha256=ck.get('initial_sha256', protocol['initial_weights_sha256'])))
    pd.DataFrame(durations).to_csv(DEST / 'training_heads.csv', index=False)
    for arm in ARMS:
        folder = DEST / 'evaluation_metadata' / arm
        folder.mkdir(parents=True)
        for name in ['run.json', 'events.jsonl']:
            shutil.copy2(RUN / 'evaluation' / arm / name, folder / name)
        if arm != 'XPF':
            shutil.copy2(RUN / 'calibration' / f'{arm}.pt', folder / 'initial_offsets.pt')
    with tarfile.open(DEST / 'new_weights.tar.gz', 'w:gz') as archive:
        archive.add(RUN / 'models', arcname='models')
    archive_path = DEST / 'new_weights.tar.gz'
    archive_sha = sha(archive_path)
    part_dir = DEST / 'weight_parts'
    part_dir.mkdir()
    parts = []
    with archive_path.open('rb') as source:
        while True:
            chunk = source.read(768 * 1024)
            if not chunk:
                break
            part = part_dir / f'{len(parts):04d}.part'
            part.write_bytes(chunk)
            parts.append(dict(path=str(part.relative_to(DEST)), sha256=sha(part)))
    (DEST / 'weights_parts.json').write_text(json.dumps(dict(archive='new_weights.tar.gz',
        sha256=archive_sha, parts=parts), indent=2)+'\n')
    rebuilt = RUN / 'verify_reconstructed_weights.tar.gz'
    with rebuilt.open('wb') as target:
        for item in parts:
            target.write((DEST / item['path']).read_bytes())
    assert sha(rebuilt) == archive_sha
    with tarfile.open(rebuilt, 'r:gz') as check:
        assert len([x for x in check.getmembers() if x.isfile() and x.name.endswith('.pt')]) == 294
    rebuilt.unlink()
    archive_path.unlink()
    summary = pd.read_csv(DEST / 'summary_trajectory_channel_equal.csv')
    paired = pd.read_csv(DEST / 'paired_comparison.csv')
    hist = json.loads((RUN / 'historical_recovery.json').read_text())
    text = ['# Legacy XPF16 feature ablation', '',
        'All reported scores use actual test windows. Test trajectories have been used in earlier exploratory work.',
        'The shared public P anchor means the F-only head is not an end-to-end P-free predictor. In this unchanged architecture X-only receives history variability, not the raw 20-frame history; with P present the first-step P change also uses the last X frame.',
        'The first-frame P-change feature depends jointly on P and X and is masked whenever either source is missing; the seven arms therefore do not form an additive feature-factor decomposition.',
        '2048-noise comes from the separate 250-grid run. Noise proxy-scale and XPF two-tail calibration differ.', '',
        '## Online trajectory/channel equally weighted standardized results', '',
        '| Method | CRPS | 90% coverage | width | interval score |', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        row = summary[(summary.method == arm) & (summary['mode'] == 'online')].iloc[0]
        text.append(f'| {arm} | {row.crps:.8f} | {row.coverage:.5f} | {row.width:.6f} | {row.interval_score:.6f} |')
    text += ['', '## Paired comparisons', '',
        'Differences are method minus reference in standardized CRPS; negative values favor the method. Bootstrap resamples the 12 test trajectories.']
    for method, reference in [('F','XPF'), ('XP','XPF'), ('XF','XPF'), ('PF','XPF'), ('F','XF'), ('F','PF'), ('XPF','256-noise'),
                              ('XPF','2048-noise'), ('F','256-noise'), ('F','2048-noise')]:
        selected = paired[(paired.method == method) & (paired.reference == reference)]
        if selected.empty:
            row = paired[(paired.method == reference) & (paired.reference == method)].iloc[0]
            delta, low, high = -row.mean_difference, -row.ci95_high, -row.ci95_low
            window_win = 1 - row.window_win_rate
        else:
            row = selected.iloc[0]
            delta, low, high = row.mean_difference, row.ci95_low, row.ci95_high
            window_win = row.window_win_rate
        text.append(f'- {method} vs {reference}: {delta:.8f}; 95% trajectory bootstrap [{low:.8f}, {high:.8f}]; window win fraction {window_win:.3f}.')
    by_re = pd.read_csv(DEST / 'by_re_channel.csv')
    f = by_re[(by_re.method == 'F') & (by_re['mode'] == 'online')]
    xpf = by_re[(by_re.method == 'XPF') & (by_re['mode'] == 'online')]
    fx = f.merge(xpf, on=['Re','channel'], suffixes=('_f','_xpf'))
    text += ['', 'F minus XPF by Re and channel (negative favors F):', '']
    for _, row in fx.sort_values(['Re','channel']).iterrows():
        text.append(f'- Re {row.Re:g}, {row.channel}: {row.crps_f-row.crps_xpf:.8f}.')
    old250 = pd.read_csv(DEST / 'historical_250_windows.csv')
    q = old250[(old250.method == 'XPF') & (old250['mode'] == 'online')]
    n = old250[(old250.method == '2048-noise') & (old250['mode'] == 'online') & (old250.estimator == 'empirical')]
    qmean = q.groupby(['Re','channel']).standardized_crps.mean().mean()
    nmean = n.groupby(['Re','channel']).standardized_crps.mean().mean()
    text += ['', f'In the separate historical 250-head protocol, Re/channel-equally weighted XPF online CRPS is {qmean:.8f} and 2048-noise proxy-online empirical CRPS is {nmean:.8f}.',
        'Other feature sets were not trained under the 250-head protocol, so ranking consistency across protocols is not established.',
        'One training seed per feature set cannot measure seed-to-seed stability; trajectory bootstrap intervals describe only this exploratory test set. Several trajectories share the same Re group, so these intervals do not establish generalization to new Re groups.']
    text += ['', 'Paired differences and 95% trajectory bootstrap intervals are in paired_comparison.csv.',
        f"Original 49-head online CRPS: {hist['history49_online_mean']:.9f}; replay maximum per-trajectory/channel/window difference: {audit['xpf_replay_max_abs_diff']:.3g}.",
        'The 250-grid XPF/2048 comparison is archived separately; other feature groups were not trained in that protocol.',
        'Window indices 2–42 are scored; cold-start window 1 is used only to initialize online feedback. Online pair-projection counters include all 42 issued windows.',
        'Original stored PDE scale is reported as physical; source data do not establish an SI unit.',
        'Different calibrators and empirical versus quantile-integral estimators limit direct equivalence claims.',
        'timing_benchmark.csv measures all 49 heads and four-step DiPred after training on one GPU; timing.csv records evaluation under variable load. Historical noise sampling times are in historical_noise_timing.csv; these exclude calibration and cannot be represented as complete end-to-end latency.',
        'Training-head wall times are shared within each parallel batch and must not be summed as independent elapsed times. The execution log records the actual batch groupings and a resumed group; all heads retain independent parameters and optimizer states.', '']
    (DEST / 'REPORT.md').write_text('\n'.join(text))
    (DEST / 'README.md').write_text('See REPORT.md, protocol.json and research/legacy_xpf16_feature_ablation/REPRODUCE.md. The two figures are reproducible from the uploaded CSVs using research/legacy_xpf16_feature_ablation/draw_from_csv.py. Run restore_weights.py in this directory to reconstruct and SHA-256-check the archive containing six times 49 independently trained heads. Existing XPF weights are referenced by SHA-256 in protocol.json. Historical compact scores are preserved in noise_per_window.csv. No synthetic window scores are used. Full retraining/replay requires the original workspace layout: copy this research directory to the workspace root, place workspace_dependencies/*.py in that root, and supply the legacy xpf16_fullpercent_study code, cached predictor features/statistics and RealPDE data. The caches and PDE data are not included here.\n')
    shutil.copy2(ROOT / 'legacy_xpf16_feature_ablation/restore_weights.py', DEST)
    (DEST / 'audit.json').write_text(json.dumps(dict(passed=True, checks=checks, replay=audit, expected_new_models=294,
        actual_new_models=len(trained['weights']), input_mask_test='see test_features.py',
        no_claim_of_strictly_equal_calibration=True), indent=2)+'\n')
    # Every model and every published artifact except the manifest itself is checksummed.
    files = {str(p.relative_to(DEST)): sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    code_hashes = {str(q.relative_to(CODE)): sha(q) for q in sorted(CODE.rglob('*')) if q.is_file()}
    (DEST / 'manifest.json').write_text(json.dumps(dict(files=files, code_sha256=code_hashes, new_model_sha256=trained['weights']), indent=2) + '\n')
    print('PACKAGED', DEST, flush=True)


if __name__ == '__main__':
    main()
