"""Exercise grid report aggregation on isolated development smoke data only."""
import json,shutil
import numpy as np
from train_grid import OUT
import summarize_grid

root=OUT/'summary_smoketest'
source=OUT/'smoke_v3/development'
meta=json.loads((source/'complete.json').read_text());meta['starts']=[0,20]
for split in ('development','test'):
    directory=root/'evaluation'/split;directory.mkdir(parents=True,exist_ok=True)
    (directory/'complete.json').write_text(json.dumps(meta))
    for start in (0,20):shutil.copyfile(source/'0000.npz',directory/f'{start:04d}.npz')
# The development smoke output has no test snapshot. Insert synthetic values
# exclusively in this isolated fixture to exercise the CDF plotting code.
snapshot=root/'evaluation/test/0000.npz'
with np.load(snapshot) as loaded:
    data={key:loaded[key] for key in loaded.files}
data.update(pixels=np.arange(16), truth_snapshot=np.zeros((2,2,16)),
            point_snapshot=np.zeros((2,2,16)),
            sample_snapshot=np.zeros((2048,2,2,16)),
            xpf_rearranged_snapshot=np.zeros((499,2,2,16)))
np.savez_compressed(snapshot,**data)
(root/'evaluation_complete.json').write_text('{"synthetic":true}')
summarize_grid.OUT=root
summarize_grid.main()
print('GRID_SUMMARY_SMOKE_PASSED: copied development data, not a study result')
