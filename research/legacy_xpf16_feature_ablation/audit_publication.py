"""Audit model completeness, data provenance, masks, causal events and CSV aggregation."""
import json
import hashlib
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from clean_reliability_common import manifest

RUN = ROOT / 'runs/legacy_xpf16_feature_ablation_20261116'
OUT = RUN / 'publication'
ARMS = ['X', 'P', 'F', 'XP', 'XF', 'PF', 'XPF']


def main():
    m = manifest()['files']
    sets = {k: {item['file'] for item in v} for k, v in m.items()}
    for a in sets:
        for b in sets:
            if a < b:
                assert sets[a].isdisjoint(sets[b]), (a, b)
    assert len(sets['test']) == 12 and len(sets['calibration']) == 10
    for field in ('group', 'sha256_first200'):
        identifiers = {k: {item[field] for item in v} for k, v in m.items()}
        for a in identifiers:
            for b in identifiers:
                if a < b:
                    assert identifiers[a].isdisjoint(identifiers[b]), (field, a, b)
    model_count = 0
    initialization = hashlib.sha256((ROOT / 'runs/selected16_quantile_20261110/compact16_initial.pt').read_bytes()).hexdigest()
    for arm in ARMS:
        for k in range(1, 50):
            p = (ROOT / f'runs/xpf16_fullpercent_20261111/models/XPF16-Q{k:02d}/model.pt' if arm == 'XPF'
                 else RUN / f'models/{arm}/Q{k:02d}.pt')
            if arm == 'XPF':
                archived = ROOT / f'github/DiPred/results/reliability/xpf16_independent49/weights/XPF16-Q{k:02d}.pt'
                assert hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256(archived.read_bytes()).digest()
            c = torch.load(p, map_location='cpu', weights_only=False)
            assert c['parameters'] == 34004 and c['config']['k'] == k
            assert c['config']['steps'] == 2000 and c['config']['batch'] == 4 and c['config']['seed'] == 42
            assert abs(c['config']['lower'] - k/100) < 1e-12 and abs(c['config']['upper'] - (1-k/100)) < 1e-12
            if arm != 'XPF':
                assert c['initial_sha256'] == initialization and c['steps'] == 2000
                assert c['loss'][-1]['step'] == 2000 and c['training_pairs'] == 819
            assert all(torch.isfinite(t).all() for t in c['model'].values())
            model_count += 1
        run = json.loads((RUN / 'evaluation' / arm / 'run.json').read_text())
        stats = run['feature_stats']
        prohibited = ([] if 'X' in arm else [0, 1]) + ([] if 'P' in arm else [2, 3, 4, 5]) + ([] if 'F' in arm else list(range(6, 14)))
        assert all(stats['mean'][i] == 0 and stats['std'][i] == 0 for i in prohibited), arm
        events = [json.loads(x) for x in (RUN / 'evaluation' / arm / 'events.jsonl').read_text().splitlines()]
        assert len(events) == 84
        for j in range(42):
            a, b = events[2*j:2*j+2]
            assert a['event'] == 'issued' and b['event'] == 'revealed_full20'
            assert a['start'] == b['start'] == j*20 and a['time'] <= b['time']
    frame = pd.read_csv(OUT / 'seven_features_per_window.csv')
    assert len(frame) == 7 * 2 * 41 * 12 * 2
    assert not frame.duplicated(['method', 'mode', 'start', 'trajectory_id', 'channel']).any()
    assert set(frame.window_index) == set(range(2,43)) and np.isfinite(frame.crps).all()
    numeric = ['crps','wis','crossing_before','crossing_any_before','within_pair_crossing','coverage','width','interval_score','lower_miss','upper_miss']
    assert np.isfinite(frame[numeric].to_numpy()).all()
    assert ((frame.coverage >= 0) & (frame.coverage <= 1)).all()
    assert (frame.width >= 0).all() and (frame.interval_score >= 0).all()
    summary = pd.read_csv(OUT / 'summary_trajectory_channel_equal.csv').set_index(['method','mode'])
    means = frame.groupby(['method','mode']).crps.mean()
    assert np.allclose(means.sort_index(), summary.crps.sort_index(), atol=1e-12)
    noise = pd.read_csv(OUT / 'noise_per_window.csv')
    for name in ['8-noise','32-noise','256-noise','2048-noise']:
        rows = noise[(noise.method == name) & (noise['mode'] == 'online') & (noise.estimator == 'empirical')]
        assert len(rows) == 41*12*2 and rows.window_index.nunique() == 41
        assert rows.groupby('window_index').standardized_crps.mean().nunique() > 1
    redraw = RUN / 'redraw_check'
    redraw.mkdir(exist_ok=True)
    for name in ('seven_features_per_window.csv', 'noise_per_window.csv'):
        shutil.copy2(OUT / name, redraw / name)
    subprocess.run([sys.executable, str(ROOT / 'legacy_xpf16_feature_ablation/draw_from_csv.py'), str(redraw)], check=True)
    for stem in ('Seven_Features_CRPS_by_Window', 'Noise_XPF_F_CRPS_by_Window'):
        assert (OUT / (stem + '.pdf')).exists() and (redraw / (stem + '.pdf')).exists()
        original_image, redrawn_image = mpimg.imread(OUT / (stem + '.png')), mpimg.imread(redraw / (stem + '.png'))
        assert original_image.shape == redrawn_image.shape
        assert float(np.abs(original_image - redrawn_image).mean()) < 1e-6, stem
    result = dict(passed=True, tested_head_count=model_count, legacy_weight_hashes_match_existing_repository=True, split_file_ids_disjoint=True, split_groups_and_first200_hashes_disjoint=True,
        test_trajectories=12, calibration_trajectories=10, window_rows=len(frame),
        missing_source_zero_after_normalization_and_projection=True,
        causal_issue_then_reveal_events=7*42, seven_summaries_match_window_rows=True,
        all_noise_curves_have_41_real_distinct_window_records=True, both_figures_redrawn_from_exported_csv=True,
        limitations=['historical test data are exploratory', 'event ordering audit does not independently prove model-input provenance',
                     'noise versus quantile CRPS estimators and online calibration differ'])
    (OUT/'audit_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
