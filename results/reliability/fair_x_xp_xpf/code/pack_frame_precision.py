"""Write enough decimal digits to reproduce each stored float32 exactly."""
import csv,os,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
path=ROOT/'github/DiPred/results/reliability/fair_x_xp_xpf/per_frame.csv'
tmp=path.with_suffix('.tmp')
with path.open(newline='') as source,tmp.open('w',newline='') as target:
    reader=csv.DictReader(source);writer=csv.DictWriter(target,fieldnames=reader.fieldnames);writer.writeheader()
    for row in reader:
        for name in reader.fieldnames:
            if name.startswith('crps_frame_'):
                value=float(row[name]);short=format(value,'.9g')
                assert struct.pack('f',value)==struct.pack('f',float(short))
                row[name]=short
        writer.writerow(row)
os.replace(tmp,path)
print('Float32 values preserved exactly.')
