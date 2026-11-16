"""Check all published 250-head archive parts against original window NPZs."""
import hashlib
import json
import io
import tarfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'runs/legacy_xpf16_feature_ablation_20261116'
RESTORED = OUT / 'restored_250'
manifest = json.loads((ROOT / 'github/DiPred/results/reliability/xpf16_02grid_2048/split_data_manifest.json').read_text())
for name, entry in manifest.items():
    p = RESTORED / name
    assert p.stat().st_size == entry['size'] and hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256']
weights = []
with tarfile.open(RESTORED / 'xpf16_grid_250_weights.tar.gz', 'r|gz') as archive:
    for member in archive:
        if not member.isfile() or not member.name.endswith('.pt'):
            continue
        k = len(weights) + 1
        assert member.name == f'XPF16-P{k:03d}.pt'
        ck = torch.load(io.BytesIO(archive.extractfile(member).read()), map_location='cpu', weights_only=False)
        assert ck['config']['k'] == k and bool(ck['config']['median_head']) == (k == 250)
        weights.append(member.name)
assert len(weights) == 250
compact = np.load(RESTORED / 'test_compact.npz')
differences = []
for i, start in enumerate(compact['starts']):
    old = np.load(ROOT / f'runs/xpf16_02grid_2048_20261112/evaluation/test/{int(start):04d}.npz')
    for key in ['noise_proxy_empirical', 'xpf_online_rearranged_crps']:
        differences.append(abs(float(old[key].mean()) - float(compact['window_' + key][i])))
assert len(differences) == 84 and max(differences) < 2e-7
result = dict(archive_files_verified=len(manifest), independent_head_checkpoints=len(weights), paired_heads=249, median_heads=1,
              test_windows_compared=42, max_abs_difference=max(differences),
              note='The new feature combinations are not trained in this separate 250-head protocol.')
(OUT / 'protocol250_audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
