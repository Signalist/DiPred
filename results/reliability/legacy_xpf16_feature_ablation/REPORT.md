# Legacy XPF16 feature ablation

All reported scores use actual test windows. Test trajectories have been used in earlier exploratory work.
The shared public P anchor means the F-only head is not an end-to-end P-free predictor. In this unchanged architecture X-only receives history variability, not the raw 20-frame history; with P present the first-step P change also uses the last X frame.
The first-frame P-change feature depends jointly on P and X and is masked whenever either source is missing; the seven arms therefore do not form an additive feature-factor decomposition.
2048-noise comes from the separate 250-grid run. Noise proxy-scale and XPF two-tail calibration differ.

## Online trajectory/channel equally weighted standardized results

| Method | CRPS | 90% coverage | width | interval score |
|---|---:|---:|---:|---:|
| X | 0.08054905 | 0.89371 | 0.472575 | 0.676713 |
| P | 0.08012599 | 0.89810 | 0.465334 | 0.666060 |
| F | 0.07949387 | 0.89963 | 0.485860 | 0.648716 |
| XP | 0.07967449 | 0.89525 | 0.459119 | 0.651008 |
| XF | 0.07888207 | 0.89512 | 0.468475 | 0.623980 |
| PF | 0.07870810 | 0.89861 | 0.469752 | 0.623673 |
| XPF | 0.07838271 | 0.89644 | 0.461630 | 0.612403 |

## Paired comparisons

Differences are method minus reference in standardized CRPS; negative values favor the method. Bootstrap resamples the 12 test trajectories.
- F vs XPF: 0.00111117; 95% trajectory bootstrap [0.00088351, 0.00134448]; window win fraction 0.000.
- XP vs XPF: 0.00129178; 95% trajectory bootstrap [0.00086394, 0.00179166]; window win fraction 0.000.
- XF vs XPF: 0.00049936; 95% trajectory bootstrap [0.00038472, 0.00061025]; window win fraction 0.000.
- PF vs XPF: 0.00032539; 95% trajectory bootstrap [0.00022900, 0.00042830]; window win fraction 0.000.
- F vs XF: 0.00061181; 95% trajectory bootstrap [0.00045415, 0.00078780]; window win fraction 0.000.
- F vs PF: 0.00078577; 95% trajectory bootstrap [0.00062999, 0.00094109]; window win fraction 0.000.
- XPF vs 256-noise: -0.00117277; 95% trajectory bootstrap [-0.00176472, -0.00060039]; window win fraction 1.000.
- XPF vs 2048-noise: -0.00091528; 95% trajectory bootstrap [-0.00145501, -0.00039507]; window win fraction 1.000.
- F vs 256-noise: -0.00006160; 95% trajectory bootstrap [-0.00055432, 0.00042212]; window win fraction 0.585.
- F vs 2048-noise: 0.00019589; 95% trajectory bootstrap [-0.00024299, 0.00063041]; window win fraction 0.293.

F minus XPF by Re and channel (negative favors F):

- Re 3750, u: 0.00050621.
- Re 3750, v: 0.00162842.
- Re 13977, u: 0.00098158.
- Re 13977, v: 0.00065717.
- Re 22926, u: 0.00182807.
- Re 22926, v: 0.00148385.

In the separate historical 250-head protocol, Re/channel-equally weighted XPF online CRPS is 0.08123503 and 2048-noise proxy-online empirical CRPS is 0.08218239.
Other feature sets were not trained under the 250-head protocol, so ranking consistency across protocols is not established.
One training seed per feature set cannot measure seed-to-seed stability; trajectory bootstrap intervals describe only this exploratory test set. Several trajectories share the same Re group, so these intervals do not establish generalization to new Re groups.

Paired differences and 95% trajectory bootstrap intervals are in paired_comparison.csv.
Original 49-head online CRPS: 0.078382707; replay maximum per-trajectory/channel/window difference: 2.57e-08.
The 250-grid XPF/2048 comparison is archived separately; other feature groups were not trained in that protocol.
Window indices 2–42 are scored; cold-start window 1 is used only to initialize online feedback. Online pair-projection counters include all 42 issued windows.
Original stored PDE scale is reported as physical; source data do not establish an SI unit.
Different calibrators and empirical versus quantile-integral estimators limit direct equivalence claims.
timing_benchmark.csv measures all 49 heads and four-step DiPred after training on one GPU; timing.csv records evaluation under variable load. Historical noise sampling times are in historical_noise_timing.csv; these exclude calibration and cannot be represented as complete end-to-end latency.
Training-head wall times are shared within each parallel batch and must not be summed as independent elapsed times. The execution log records the actual batch groupings and a resumed group; all heads retain independent parameters and optimizer states.
