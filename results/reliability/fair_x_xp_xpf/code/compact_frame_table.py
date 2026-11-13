"""Losslessly pivot per-frame long CSV to one row with 20 lead columns."""
import csv,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'github/DiPred/results/reliability/fair_x_xp_xpf'
path=DEST/'per_frame.csv';tmp=path.with_suffix('.tmp')
keys=('method','seed','calibration','trajectory_id','Re','window_start','channel')
count=0
with path.open(newline='') as source,tmp.open('w',newline='') as target:
    reader=csv.DictReader(source)
    columns=list(keys)+[f'crps_frame_{i:02d}' for i in range(1,21)]
    writer=csv.DictWriter(target,fieldnames=columns);writer.writeheader()
    current=None
    for row in reader:
        key=tuple(row[k] for k in keys);frame=int(row['future_frame'])
        if frame==1:
            if current is not None:raise ValueError('Incomplete frame block')
            current=dict(zip(keys,key))
        if current is None or tuple(current[k] for k in keys)!=key or frame!=1+sum(k.startswith('crps_frame_') for k in current):
            raise ValueError('Unexpected alignment or frame order')
        current[f'crps_frame_{frame:02d}']=row['crps']
        if frame==20:
            writer.writerow(current);current=None;count+=1
    if current is not None:raise ValueError('Incomplete final block')
assert count==27*41*12*2
os.replace(tmp,path)
print('Wrote',count,'complete 20-frame rows')
