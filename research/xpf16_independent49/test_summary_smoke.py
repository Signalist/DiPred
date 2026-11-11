"""Check aggregation independently in a separate synthetic workspace."""
import json
import shutil
from pathlib import Path
import summarize49
from train49 import OUT

root=OUT/'summary_smoketest'
source=OUT/'smoke_evaluation_v3/development'
meta=json.loads((source/'run.json').read_text())
meta['starts']=[0,20]
for split in ('development','test'):
    dest=root/'evaluation'/split
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'run.json').write_text(json.dumps(meta))
    for start in (0,20):
        shutil.copyfile(source/'0000.npz',dest/f'{start:04d}.npz')
(root/'evaluation_complete.json').write_text('{}')
summarize49.OUT=root
summarize49.plot=lambda *x:None
summarize49.plot_extra=lambda *x:None
summarize49.main()
print('SUMMARY_SMOKE_PASSED: copied smoke data, not real test results')
