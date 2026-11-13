"""Generate a facts-only report directly from the audited result tables."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'github/DiPred/results/reliability/fair_x_xp_xpf'
with (DEST/'summary.csv').open(newline='') as f: rows=list(csv.DictReader(f))
audit=json.loads((DEST/'audit.json').read_text())
manifest=json.loads((DEST/'manifest.json').read_text())
report=['# X / XP / XPF matched ablation', '',
'Training status: 9 of 9 heads completed (3 arms × 3 seeds; 2,000 steps each).',
'The same fixed slots, network, parameter count, sample batches, objective, seed-specific initialization and optimizer schedule were used. X has P/F zeroed; XP has F zeroed; XPF receives the last actual hidden state of the frozen four-step DiPred. Every head predicts Y relative to the final X frame.',
f"Each head has {rows[0]['head_parameters']} parameters; XP and XPF additionally use {manifest['frozen_backbone_parameters']} frozen backbone parameters.",
'Train, development, calibration and test files and Re groups are disjoint. The test trajectories have been used in previous research and the results are exploratory; there is no verified fresh independent test set.',
'Raw results are the principal comparison. Fixed CQR and online two-tail calibration use the same procedures across arms. Each cached input entry contains only that window’s observed history and its P/F; the evaluation replay loads only the current entry and issues the full 20-frame forecast before using its target as feedback. The input cache itself was prepared offline, so event times represent a causal replay rather than live deployment. Calibration pools correlated positions and windows and has no claimed trajectory-level coverage guarantee.',
'Raw heads are monotone by construction; calibrated quantiles are rearranged before distribution scoring. Crossing columns describe the calibrated quantiles before rearrangement. All CRPS values use the 0.2% grid, trapezoidal pinball integration and clamped endpoints. The CDF uses piecewise-linear quantile interpolation and endpoint atoms in original dataset units.',
'All inference latency measurements include 499 output levels; XP and XPF include the frozen four-call DiPred. Fixed-CQR and online timings include calibration and quantile sorting. All timings exclude one-time model loading and feedback updates. Recorded training time is the actual wall time of the concurrent group of three heads, not a per-head runtime.',
'', 'Raw results per seed and channel (full-test mean):', '',
'| Method | Seed | Channel | CRPS | Coverage 90% | Interval score |',
'|---|---:|---|---:|---:|---:|']
for r in rows:
 if r['calibration']=='raw' and r['channel'] in ('u','v'):
  report.append(f"| {r['method']} | {r['seed']} | {r['channel']} | {float(r['crps']):.8g} | {float(r['coverage90']):.6f} | {float(r['interval_score90']):.8g} |")
report += ['',f"Audits passed: {audit['passed']}. See `audit.json` for individual checks.",
'`per_window.csv` and the 20-column wide `per_frame.csv` contain full aligned observation summaries; `quantile_metrics.csv` and `cdf_u.csv` / `cdf_v.csv` cover the same windows. `paired_comparison.csv` uses 10,000 bootstrap resamples of trajectories, with seeds averaged within a trajectory for pooled results. Additional per-window paired intervals are descriptive and unadjusted for multiple comparisons. Several trajectories share the same Re group, so this bootstrap is not a Re-group-level confidence interval.',
'`prediction_snapshot.npz` is a small subset of actual aligned predictions and targets. Run `python code/recalculate.py .` from this result directory to independently check the published summary and sample CRPS. The full raw PDE data and full prediction tensors are not included.',
'The exact source scripts and trained weights are included. Full training/evaluation additionally require the original RealPDE files, the original workspace cache and its research dependencies; see `config.json` and `manifest.json` for splits, settings and hashes.', '']
(DEST/'REPORT.md').write_text('\n'.join(report))
