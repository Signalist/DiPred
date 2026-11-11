"""Render all charts from copied synthetic smoke artifacts in isolation."""
import csv
from pathlib import Path
import numpy as np
import summarize49
from train49 import OUT

root=OUT/'summary_smoketest'
folder=root/'summary'
summarize49.OUT=root
fake=np.broadcast_to(np.arange(98,dtype=np.float32).reshape(98,1,1,1),(98,2,2,32))
np.savez_compressed(root/'evaluation/test/snapshot.npz',raw=fake,online=fake,truth=np.zeros((2,2,32)))
macro=list(csv.DictReader((folder/'macro49.csv').open()))
group=list(csv.DictReader((folder/'test_group_lead.csv').open()))
for row in macro:
    for key in ('alpha','target_coverage','coverage','width','lower_coverage','upper_coverage','lower_pinball','upper_pinball'):
        row[key]=float(row[key])
for row in group:
    for key in ('alpha','target_coverage','coverage','lower_coverage','upper_coverage'):
        row[key]=float(row[key])
    row['lead']=int(row['lead'])
summarize49.plot(folder,macro,group)
summarize49.plot_extra(folder,group)
print('PLOT_SMOKE_PASSED: synthetic input only')
