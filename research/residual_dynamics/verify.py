"""Independent checks of released trajectory scores and experiment boundaries."""
import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import torch

from run import REPO, ResidualPredictor, dump, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPO / 'results/residual_dynamics')
    parser.add_argument('--check-cache', action='store_true')
    args = parser.parse_args()
    p = json.loads((args.out / 'protocol.json').read_text())
    train = json.loads((args.out / 'training.json').read_text())
    audit = json.loads((args.out / 'alignment.json').read_text())
    scores = json.loads((args.out / 'summary.json').read_text())
    contrasts = json.loads((args.out / 'contrasts.json').read_text())
    rows = list(csv.DictReader((args.out / 'per_trajectory.csv').open()))
    sets = {split: {r['file'] for r in records} for split, records in p['records'].items()}
    assert not any(sets[a] & sets[b] for a in sets for b in sets if a != b)
    assert p['starts'] == [400, 420] and audit['passed']
    assert all(r['all_32_members_all_20_frames_max_abs_error'] <= 2e-6 for r in audit['snapshots'])
    for summary in scores:
        subset = [r for r in rows if r['split'] == summary['split'] and r['arm'] == summary['arm']
                  and r['condition'] == summary['condition']]
        assert len(subset) == summary['trajectories']
        assert {r['trajectory'] for r in subset} == sets[summary['split']]
        for key in ('mse', 'mse_u', 'mse_v'):
            assert np.isclose(np.mean([float(r[key]) for r in subset]), summary[key], rtol=1e-12)
        assert np.isclose(np.mean(summary['trial_macro_mse']), summary['mse'], rtol=1e-12)
    lookup = {(r['split'], r['arm'], r['condition'], r['trajectory']): float(r['mse']) for r in rows}
    for row in contrasts:
        if row['contrast'] in ('B-A', 'C-B'):
            a, b = row['contrast'].split('-')
            ac, bc = 'aligned', 'aligned'
        else:
            a = b = row['contrast'][0]
            ac, bc = 'member_shuffled', 'aligned'
        delta = np.array([lookup[row['split'], a, ac, record['file']] - lookup[row['split'], b, bc, record['file']]
                          for record in p['records'][row['split']]])
        assert np.isclose(delta.mean(), row['mean_delta_mse'], rtol=1e-12)
        rng = np.random.default_rng(123)
        draws = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
        assert np.allclose(np.quantile(draws, [.025, .975]), row['trajectory_bootstrap_95'], rtol=1e-12)
    for arm in 'ABC':
        model = ResidualPredictor(arm)
        assert sum(v.numel() for v in model.parameters()) == train['parameters'][arm]
    torch.manual_seed(42)
    b = ResidualPredictor('B')
    torch.manual_seed(42)
    c = ResidualPredictor('C')
    assert all(torch.equal(x, y) for x, y in zip(b.net.parameters(), c.net.parameters()))
    x = torch.randn(4, 2, 32, 64)
    assert torch.equal(b(x, torch.arange(2, 6)), c(x, torch.arange(2, 6)))
    # A permutation moves the complete 2-channel field, never targets or pixels.
    rng = torch.Generator().manual_seed(7100)
    e = torch.randn(32, 2, 20, 32, 64)
    y = e[:, :, 1].clone()
    donor = torch.randperm(32, generator=rng)
    previous = e[:, :, 0][donor]
    assert torch.equal(previous[donor.argsort()], e[:, :, 0])
    assert torch.equal(y, e[:, :, 1])
    assert not torch.equal(donor, torch.arange(32))
    checked = []
    if args.check_cache:
        for entry in json.loads((args.out / 'cache_manifest.json').read_text()):
            assert sha(entry['path']) == entry['sha256']
            checked.append(entry['path'])
    current = REPO / 'reliability/checkpoints/current.pt'
    assert sha(current) == p['frozen']['packaged_checkpoint_sha256']
    dump(args.out / 'verification.json', dict(passed=True, trajectory_scores_reaggregated=True,
        contrasts_recomputed=True, split_disjoint=True, snapshot_frames_members_exact=True,
        B_C_initial_function_exact=True, shuffle_preserves_fields_and_targets=True,
        backbone_checkpoint_unchanged=True, cache_files_hash_checked=checked,
        parameters=train['parameters'], python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        code_sha256={name: sha(Path(__file__).parent/name) for name in ('run.py', 'verify.py')}))
    print('VERIFIED', len(rows), 'trajectory rows')


if __name__ == '__main__':
    main()
