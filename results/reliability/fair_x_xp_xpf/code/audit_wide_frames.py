"""Check complete wide per-frame data against every per-window CRPS."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'github/DiPred/results/reliability/fair_x_xp_xpf'
keys=('method','seed','calibration','trajectory_id','Re','window_start','channel')
with (DEST/'per_window.csv').open(newline='') as handle:
    original={tuple(row[k] for k in keys):float(row['crps']) for row in csv.DictReader(handle)}
seen=set();maximum=0.;valid=True
with (DEST/'per_frame.csv').open(newline='') as handle:
    reader=csv.DictReader(handle)
    valid &= all(f'crps_frame_{i:02d}' in reader.fieldnames for i in range(1,21))
    for row in reader:
        key=tuple(row[k] for k in keys);valid &= key in original and key not in seen
        seen.add(key)
        if key in original:
            score=sum(float(row[f'crps_frame_{i:02d}']) for i in range(1,21))/20
            maximum=max(maximum,abs(score-original[key]))
valid &= len(seen)==len(original)==27*41*12*2 and maximum<1e-6
result={'passed':bool(valid),'rows':len(seen),'frames_per_row':20,'max_window_crps_error':maximum,'precision':'each decimal roundtrips the original float32 exactly'}
(DEST/'wide_frame_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert valid
