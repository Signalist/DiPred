# X / XP / XPF matched ablation

Training status: 9 of 9 heads completed (3 arms × 3 seeds; 2,000 steps each).
The same fixed slots, network, parameter count, sample batches, objective, seed-specific initialization and optimizer schedule were used. X has P/F zeroed; XP has F zeroed; XPF receives the last actual hidden state of the frozen four-step DiPred. Every head predicts Y relative to the final X frame.
Each head has 75310 parameters; XP and XPF additionally use 1616872 frozen backbone parameters.
Train, development, calibration and test files and Re groups are disjoint. The test trajectories have been used in previous research and the results are exploratory; there is no verified fresh independent test set.
Raw results are the principal comparison. Fixed CQR and online two-tail calibration use the same procedures across arms. Each cached input entry contains only that window’s observed history and its P/F; the evaluation replay loads only the current entry and issues the full 20-frame forecast before using its target as feedback. The input cache itself was prepared offline, so event times represent a causal replay rather than live deployment. Calibration pools correlated positions and windows and has no claimed trajectory-level coverage guarantee.
Raw heads are monotone by construction; calibrated quantiles are rearranged before distribution scoring. Crossing columns describe the calibrated quantiles before rearrangement. All CRPS values use the 0.2% grid, trapezoidal pinball integration and clamped endpoints. The CDF uses piecewise-linear quantile interpolation and endpoint atoms in original dataset units.
All inference latency measurements include 499 output levels; XP and XPF include the frozen four-call DiPred. Fixed-CQR and online timings include calibration and quantile sorting. All timings exclude one-time model loading and feedback updates. Recorded training time is the actual wall time of the concurrent group of three heads, not a per-head runtime.

Raw results per seed and channel (full-test mean):

| Method | Seed | Channel | CRPS | Coverage 90% | Interval score |
|---|---:|---|---:|---:|---:|
| X | 42 | u | 0.0070927145 | 0.860920 | 0.070204653 |
| X | 42 | v | 0.0035877403 | 0.814153 | 0.032507448 |
| XP | 42 | u | 0.0066396773 | 0.886890 | 0.065521017 |
| XP | 42 | v | 0.0029231919 | 0.832763 | 0.025661668 |
| XPF | 42 | u | 0.0063155149 | 0.883690 | 0.06324077 |
| XPF | 42 | v | 0.0022124282 | 0.792566 | 0.020344378 |
| X | 123 | u | 0.0071620907 | 0.834953 | 0.071995254 |
| X | 123 | v | 0.0037285571 | 0.856988 | 0.033220028 |
| XP | 123 | u | 0.0065793244 | 0.831138 | 0.064083154 |
| XP | 123 | v | 0.0029278419 | 0.843485 | 0.025311314 |
| XPF | 123 | u | 0.005918041 | 0.846656 | 0.057751318 |
| XPF | 123 | v | 0.0021676657 | 0.796285 | 0.020722962 |
| X | 2026 | u | 0.0071300755 | 0.812967 | 0.070902232 |
| X | 2026 | v | 0.0036512574 | 0.839324 | 0.032422817 |
| XP | 2026 | u | 0.0065936703 | 0.824004 | 0.067327462 |
| XP | 2026 | v | 0.0029139329 | 0.848308 | 0.025408339 |
| XPF | 2026 | u | 0.0060357607 | 0.812126 | 0.060149075 |
| XPF | 2026 | v | 0.002170521 | 0.804652 | 0.019999871 |

Audits passed: True. See `audit.json` for individual checks.
`per_window.csv` and the 20-column wide `per_frame.csv` contain full aligned observation summaries; `quantile_metrics.csv` and `cdf_u.csv` / `cdf_v.csv` cover the same windows. `paired_comparison.csv` uses 10,000 bootstrap resamples of trajectories, with seeds averaged within a trajectory for pooled results. Additional per-window paired intervals are descriptive and unadjusted for multiple comparisons. Several trajectories share the same Re group, so this bootstrap is not a Re-group-level confidence interval.
`prediction_snapshot.npz` is a small subset of actual aligned predictions and targets. Run `python code/recalculate.py .` from this result directory to independently check the published summary and sample CRPS. The full raw PDE data and full prediction tensors are not included.
The exact source scripts and trained weights are included. Full training/evaluation additionally require the original RealPDE files, the original workspace cache and its research dependencies; see `config.json` and `manifest.json` for splits, settings and hashes.
