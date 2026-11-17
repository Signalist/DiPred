# Active observations on original RealPDE

Results: `runs/active_observation_20261117/REPORT.md`. Three retained mechanisms: global residual factors, observable future-error regression, and a2432-coefficient empirical benefit critic. The task explicitly adds observations at forecast lead5 and evaluates only future leads6–20. Analytic VOI did not stably beat risk selection; the critic's development advantage did not persist into the later window. No broad VOI or confirmed75% observation-saving claim is made.

Run from Workspace after `source env.sh`. The original frozen predictor/heads and cached train/cal/dev splits are dependencies; no original checkpoint or default mainline was replaced.

```bash
python active_observation_20261117/observations.py prepare --split train
python active_observation_20261117/observations.py fit
python active_observation_20261117/observations.py prepare --split calibration
python active_observation_20261117/observations.py calibrate
# For calibration/development/late/future, prepare missing caches then evaluate.
python active_observation_20261117/observations.py evaluate --split development
python active_observation_20261117/audit.py
python active_observation_20261117/report.py

python active_observation_20261117/predictive.py fit
python active_observation_20261117/predictive.py calibrate
python active_observation_20261117/predictive.py evaluate --split development
python active_observation_20261117/predictive_audit.py
python active_observation_20261117/report.py --predictive

python active_observation_20261117/critic.py features
python active_observation_20261117/critic.py fit
python active_observation_20261117/critic.py calibrate
python active_observation_20261117/critic.py evaluate --split development
python active_observation_20261117/critic_audit.py
python active_observation_20261117/benchmark.py
python active_observation_20261117/summary_report.py
```

Evaluate each of calibration/development/late/future before running the full audits/reports. `observations.py prepare` runs actual batch1 inference and stores full marginal scale fields, so it is intentionally not a tiny cache. Predictive/critic reuse exactly those same issued forecasts. Their fit/calibrate routines read only train/calibration labels respectively.

Public interfaces, with this source directory on `sys.path`:

```python
from active_stream import SparseUpdatePredictor
from predictive import make_predictor
from critic import make_critic_predictor

predictor = make_predictor(method="risk", budget=8)
# Float32 CPU physical history, shape[1,2,20,32,64].
request = predictor.predict(history, start=0, trajectory="example")
# Obtain only request["sites_yx"] at the specified observation_time25.
# selected_uv is CPU physical values, shape[8,2]; an entire target is rejected.
future15 = predictor.update(selected_uv, at=request["observation_time"])
```

`SparseUpdatePredictor` uses the first factor model; `make_predictor` uses the predictive-mode model; `make_critic_predictor` uses the learned selection policy with that same predictive-mode update. Outputs are GPU physical fields. Initialization/model loading and the five-frame wait are separate from measured per-request compute.

Known implementation fixes: an audit import initially resolved an older module with the same short name; explicit loading and the unique `active_stream.py` name remove that ambiguity. Empty greedy requests now retain integer index dtype. Normalization is cached at assimilator initialization rather than reloading the checkpoint for each update; all replay/label/mask audits passed again afterward. Original timing files before this optimization remain as `before_norm_cache_*` diagnostics. No statistical threshold was relaxed.
