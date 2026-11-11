"""Independently verify compact replay snapshot, CRPS, and forecast order."""
import json
from pathlib import Path

import numpy as np
import torch

from xpf_crps_metrics import ensemble_crps
from xpf_quantiles129_head import quantile_levels

OUT = Path('runs/xpf_quantiles129_calibrated_20261111')
s = np.load(OUT / 'snapshot.npz')
point = s['point_normalized'].astype('float64')
y = s['truth_normalized'].astype('float64')
report = {}
for name, n in (('8-noise', 8), ('32-noise', 32)):
    raw = s['raw_samples32'][:n].astype('float64')
    lo = s[name + '_online_lo'].astype('float64')
    hi = s[name + '_online_hi'].astype('float64')
    radius = s[name + '_raw_radius'].astype('float64')
    observed = s[name + '_calibrated_samples'].astype('float64')
    predicted = point[None] + (raw - point[None]) * ((hi-lo) / np.maximum(2*radius, 1e-6))[None]
    scale_error = float(np.max(np.abs(predicted-observed)))
    assert scale_error < 2e-6
    first = np.mean(np.abs(observed-y[None]), axis=0)
    pair = np.abs(observed[:, None] - observed[None, :]).sum(axis=(0, 1))
    expected = {'empirical': first-pair/(2*n*n), 'fair': first-pair/(2*n*(n-1))}
    actual = ensemble_crps(torch.tensor(observed), torch.tensor(y))
    errors = {k: float(np.max(np.abs(actual[k].numpy()-v))) for k,v in expected.items()}
    assert max(errors.values()) < 1e-12
    report[name] = {'sample_transform_max_error': scale_error, 'crps_pairwise_max_error': errors}
levels = quantile_levels()
assert len(levels) == 129 and np.all(np.diff(levels) > 0)
for name in ('quantiles129_raw','quantiles129_online'):
    assert np.min(np.diff(s[name],axis=2)) >= -1e-7
    report[name+'_min_gap'] = float(np.min(np.diff(s[name],axis=2)))
events = [json.loads(line) for line in (OUT/'events.jsonl').read_text().splitlines()]
assert len(events) == 2*42*2
for issue, reveal in zip(events[::2], events[1::2]):
    assert issue['event'] == 'all_calibrated_forecasts_issued'
    assert reveal['event'] == 'full20_target_revealed'
    assert (issue['split'],issue['history_start']) == (reveal['split'],reveal['history_start'])
    assert issue['time'] <= reveal['time']
report['issue_reveal_pairs'] = len(events)//2
report['passed'] = True
(OUT/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
