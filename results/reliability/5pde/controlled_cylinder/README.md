# Result layout

`summary.csv`, `paired_comparison.csv`, `paired_by_window.csv`, `quantile_metrics.csv`, channel-specific `cdf_*.csv`, `config.json`, `manifest.json` and the small prediction snapshot describe the complete held-out test evaluation. `per_window_index.csv` and `per_frame_index.csv` list all twelve model/seed shards (nine trained heads and three sampling baselines) under `runs/`. Per-frame tables have one row per window/channel and columns `crps_frame_01` through `crps_frame_20`; these values round-trip the original float32 scores.

The separate source code is under `research/5pde/`. To check a scenario without the large original PDE data, run `python research/5pde/verify_release.py results/reliability/5pde/<scenario>`. `REPORT.md` explains normalization, the evaluation protocol, and limitations.

Large CSV tables are losslessly gzip-compressed; use `pandas.read_csv(path)` directly. Indices name the compressed files.
