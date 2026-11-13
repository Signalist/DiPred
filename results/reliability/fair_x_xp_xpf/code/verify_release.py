"""Check released file hashes and the wide per-frame table without PDE data."""
import csv,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
manifest=json.loads((root/'manifest.json').read_text())
for name,expected in manifest['files_sha256'].items():
    digest=hashlib.sha256()
    with (root/name).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=expected:raise ValueError('Hash mismatch: '+name)
for name in ('audit.json','extra_audit.json','report_audit.json','recalculation_audit.json','wide_frame_audit.json'):
    if not json.loads((root/name).read_text())['passed']:raise ValueError('Failed audit: '+name)
keys=('method','seed','calibration','trajectory_id','Re','window_start','channel')
with (root/'per_window.csv').open(newline='') as handle:
    reference={tuple(row[k] for k in keys):float(row['crps']) for row in csv.DictReader(handle)}
seen=set();maximum=0.
with (root/'per_frame.csv').open(newline='') as handle:
    for row in csv.DictReader(handle):
        key=tuple(row[k] for k in keys)
        if key in seen or key not in reference:raise ValueError('Unmatched per-frame row')
        seen.add(key)
        value=sum(float(row[f'crps_frame_{i:02d}']) for i in range(1,21))/20
        maximum=max(maximum,abs(value-reference[key]))
if len(seen)!=len(reference) or maximum>1e-6:raise ValueError('Per-frame audit failed')
print('All hashes and audits pass; wide rows:',len(seen),'maximum window CRPS discrepancy:',maximum)
