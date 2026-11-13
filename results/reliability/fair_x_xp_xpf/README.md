# Fair X / XP / XPF result files

The primary comparison is the `raw` row. All three arms share the same head architecture, same common X-last-frame reference, matched seed and training minibatches. X zeroes the P and F slots; XP zeroes F; XPF uses the actual final up2 hidden state of the frozen four-call DiPred. The point and hidden features are computed without future observations.

Each score is computed on test starts 20, 40, ..., 820. The start-0 prediction is a cold-start window, excluded from reported metrics but revealed to initialize online feedback. History/P/F entries were precomputed offline for each window; the evaluation replay loads only the current entry and only then issues a prediction before reading that window’s target. Each row in `per_window.csv` represents one trajectory, start and velocity channel; `per_frame.csv` is a wide table: columns `crps_frame_01` through `crps_frame_20` contain all future-frame scores for the same row key. Each decimal roundtrips its original float32 value exactly. `summary.csv` contains all seeds, arms, calibration modes and channel-wise or two-channel means. Both CRPS and interval scores are in original dataset units. The parameter columns distinguish the shared head from the additional frozen DiPred parameters. The timing column reports full-grid prediction latency, including DiPred for XP/XPF and calibration plus rearrangement where applicable; model loading and feedback updates are excluded.

All scores use 499 quantiles, levels 0.002, 0.004, ..., 0.998. CRPS uses trapezoidal integration of twice pinball loss; the value at probability 0 or 1 is clamped to the nearest predicted quantile. Calibrated quantities are monotonically rearranged before scoring, and the adjacent-crossing and any-crossing columns record crossing before rearrangement. Raw quantiles are monotone by architecture. The empirical CQR calibration pools correlated positions and windows; no independent-trajectory or shift-robust coverage guarantee is claimed.

The `cdf_u.csv` and `cdf_v.csv` columns share the same original-scale x axis. At each position x is forecast value minus truth; CDF interpolation is linear between quantile points and missing tails are represented by endpoint atoms. `quantile_metrics.csv` gives the calibration and pinball results on the same 499-point grid. `paired_comparison.csv` defines improvement as baseline minus comparison and resamples trajectories 10,000 times; `pooled` first averages seeds within each trajectory. `paired_by_window.csv` additionally provides descriptive per-window bootstrap intervals; these multiple comparisons are unadjusted. Trajectories can share Re groups, so this is not a Re-group-level interval.

The NPZ snapshot contains actual quantile and target values on a small set of grids in normalized units; `normalization_std` maps errors and CRPS to original units. Run the following command **from this directory** to independently recompute aggregate and snapshot CRPS, plus paired comparisons and bootstrap intervals:

```bash
python code/recalculate.py .
python code/verify_release.py .
```

For raw inference from a PyTorch history tensor in original dataset units, run for example:

```bash
python code/predict.py --history history.pt --weights weights/XPF_42.pt --arm XPF --output quantiles.pt
```

The inference script uses `reliability/checkpoints/current.pt` from this repository and returns the 499 quantiles in original units. The model weights and fitted calibrators, their SHA-256 hashes, splits, feature statistics and normalization values are included. `code/` contains the original training and evaluation scripts; `research_dependencies.zip` archives their workspace Python dependencies. Reproducing the full training/evaluation additionally requires the original external RealPDE dataset and historical workspace caches, neither of which is uploaded. The test trajectories were previously used during research, so all test results are exploratory.
