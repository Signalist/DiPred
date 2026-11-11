# XPF16: 19 ordered marginal quantile levels

The original 5%/95% XPF16 endpoints are frozen. A separate 26,964-parameter head
learns the 17 interior levels from the same 16 inputs and actual prediction
residuals. Positive gaps guarantee ordered levels. Four-step DiPred and the
existing complete-20-frame `memory_dynamic` calibrator are unchanged.

`group_summary.json` and `trajectory_summary.json` aggregate warm windows by
group/channel and by trajectory/channel, respectively, for all 19 levels and
raw/online variants. `summary.json` and `quantile_table.csv` provide global
coverage and standardized pinball loss. The full 71,136 window rows remain local. `events`,
`protocol`, `frozen`, `training` and `independent_audit` document causality,
fit, frozen endpoints, matching checks and exploratory limitations. Original
and online endpoints reproduce the existing online XPF16 controls; the empirical
CDF uses <= and the prior lower miss uses <, so exact ties differ.

Portable inference (history is original-scale `[B,2,20,32,64]`):

```bash
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.quantiles19_predict \
  --input history.pt --output forecast.pt --geometry 0.0034 -0.0034 0.02
```

Replace geometry with actual dx,dy,dt. `quantiles` and `raw_quantiles` have shape
`[B,2,19,20,32,64]` and are in the original field scale. `quantile_levels` is
0.05..0.95. CLI is cold-start. For online streams, instantiate
`Quantiles19Predictor`, call `predict`, then `reveal` only after the complete20
truth frames, and `reset` for another trajectory/batch order.

The 5–95% nodes do not specify tails below 5%/above 95%, a continuous CDF or a
joint spatiotemporal distribution. The historical development/test trajectories
have been reused: results are exploratory, not fresh independent confirmation.
Report: `docs/研究结果_XPF16_19分位水平_20261111.md`.
