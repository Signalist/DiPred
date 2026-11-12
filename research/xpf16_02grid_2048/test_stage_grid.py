"""Stage synthetically populated outputs in an isolated scratch directory."""
import json,shutil
from pathlib import Path
import numpy as np
import stage_grid
from train_grid import OUT,configs

root=OUT/'summary_smoketest'
(root/'full_test_cdf').mkdir(parents=True, exist_ok=True)
(root/'full_test_cdf/config.json').write_text('{"synthetic":true}')
for name in ('protocol.json','training_complete.json','evaluation_complete.json','independent_audit.json','portable_audit.json','latency.json'):
    (root/name).write_text('{"passed":true,"synthetic":true}')
shutil.copy2(OUT.parent/'xpf16_extreme_002_2048_20261112/calibration.pt',root/'calibration.pt')
model=(OUT/'models/XPF16-P005/model.pt').resolve()
for c in configs():
    directory=root/'models'/c['id'];directory.mkdir(parents=True,exist_ok=True)
    p=directory/'model.pt'
    if not p.exists():p.symlink_to(model)
folder=root/'evaluation/test';folder.mkdir(parents=True,exist_ok=True)
original=np.load(folder/'0000.npz')
snap={k:original[k] for k in original.files}
snap.update(pixels=np.arange(16),truth_snapshot=np.zeros((2,2,16)),point_snapshot=np.zeros((2,2,16)),
 sample_snapshot=np.zeros((2048,2,2,16)),noise_snapshot=np.zeros((499,2,2,16)),
 xpf_snapshot=np.zeros((499,2,2,16)),xpf_rearranged_snapshot=np.zeros((499,2,2,16)))
np.savez_compressed(folder/'0000.npz',**snap)
stage_grid.OUT=root
stage_grid.REPO=root/'staged_repo'
stage_grid.DEST=stage_grid.REPO/'results/reliability/xpf16_02grid_2048'
stage_grid.SRC=stage_grid.REPO/'research/xpf16_02grid_2048'
stage_grid.main()
assert (stage_grid.DEST/'split_data_manifest.json').exists()
print('GRID_STAGE_SMOKE_PASSED; synthetic artifacts stay outside repository')
