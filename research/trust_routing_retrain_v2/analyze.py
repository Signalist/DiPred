"""Aggregate saved development rows; no model calls."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import OUT,csvwrite

METRICS=['crps','energy','energy_raw','mse','smse','variance','std_variance','width90','std_width90','time_mean_crps']

def load(path):return list(csv.DictReader(Path(path).open()))

def aggregate(rows,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,items in groups.items():
        result.append(dict(zip(keys,key),n=len(items),**{m:float(np.mean([float(x[m]) for x in items])) for m in METRICS}))
    return result

def main():
    rows=load(OUT/'development_native_fp32_scores.csv')
    for arm in ('G_context','G_trust'):rows+=load(OUT/f'development_{arm}.csv')
    overall=[r for r in rows if r['channel']=='overall']
    csvwrite(OUT/'development_per_trajectory.csv',aggregate(overall,['arm','step','trajectory_id','Re']))
    csvwrite(OUT/'development_by_Re.csv',aggregate(overall,['arm','step','Re']))
    csvwrite(OUT/'development_by_channel.csv',aggregate(rows,['arm','step','channel']))
    csvwrite(OUT/'development_by_Re_channel.csv',aggregate(rows,['arm','step','Re','channel']))
    csvwrite(OUT/'candidate_disclosure.csv',aggregate(overall,['arm','step']))
    print('ANALYZED saved development rows:',len(rows),flush=True)

if __name__=='__main__':main()
