# XPF16 / 8-noise / 32-noise — matched online spatiotemporal calibration

All three use the same frozen four-step State DiPred, common zero-noise point P,
and the same `MemoryQuantileCalibrator("memory_dynamic")` implementation.
Each method has separately fitted initialization and its own evolving state.
8-noise uses actual paths with seeds 500000–500007; 32-noise uses actual32-path std.
XPF16 outputs learned 05/95 residual quantiles. Their interval midpoints may differ,
although the base point forecast is common. Neural weights are unchanged.

Run from repository root:

```bash
python analyze_xpf16_noise8_noise32.py
```

This standard-library analysis independently recomputes group means, test macro
scores, both 96-cell comparisons, every counterexample, and both CRPS-completion
comparisons from `per_case.json`. It requires no original dataset or GPU.

- `per_case.json`: all5616 development/test trajectory-window-channel-method rows.
- `summary.json`, `test_macro.json`, `comparison.csv`: full and primary aggregates.
- `regression_counts.json`, `counterexamples.*`: original8metrics×12cells;
  XPF16 vs8-noise66wins/30regressions, vs32-noise70wins/26regressions.
- `crps_group_comparisons.json`: separate Gaussian/Laplace assumed-CDF diagnostics.
- `calibration.pt`: separately fitted initial state for all3 methods, no neural weights.
- `audit_snapshot.npz`: small actual8-path/interval/score snapshot for formula checking.
- `independent_audit.json`, `events.jsonl`, `protocol.json`, `complete.json`:
  provenance, full20-frame causal issue/reveal checks and exact frozen-control replay.
- `comparison.png/.pdf/.svg`: table figure.

The previous matched XPF16/32-noise interval comparison already used the same
online controller. This study explicitly verifies that equality and adds actual8-noise;
it does not claim to repair previously unequal calibration algorithms.

CRPS requires a full CDF. Gaussian and Laplace completions both preserve the issued
5/95% endpoints but impose different tail assumptions. These are not native full
CDFs learned by the quantile head; XPF16 does not consistently win across both.
Coverage is pointwise marginal, not simultaneous20-frame coverage. Test windows2–39,
equal Re/channel weights; all data reused, exploratory and no formal coverage guarantee.
Source-only replay code is in `research/xpf16_noise8_noise32/`; detailed report in `docs/`.
