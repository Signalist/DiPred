# Signed online state exploration

Report: `runs/online_signed_state_20261117/REPORT.md`. Point MSE, controlled forecast-side bias/geometry diagnostics, actual online inference, free rollouts, common raw/fixed/online probability calibration and a disclosed SPCI-QRF/AcMCP block-h1 task-interval comparison are implemented. This does not complete the six-route goal or establish physical-parameter identification or fresh external validation.

New extensions:

- `probability.py fit`, then `run --split calibration/development/late/future`; `probability_audit.py`; `probability_report.py`.
- `temporal_intervals.py collect`, `calibrate`, `run`; `temporal_audit.py`; `temporal_report.py`. Same12 trajectories,41 blocks each,10 common warm-up blocks. QRF uses sklearn's documented optional pkg_resources fallback because unrelated system package metadata is unreadable; no system packages changed.
- `geometric_drift.py run`, then `report`; `geometric_drift_audit.py`. Forecast-output warp fault only, with frozen natural-calibration settings.

Run after `source env.sh` from Workspace. The protocols and official reference-source hashes document all adaptations. AcMCP is a Python translation of the block-h1 recursion; R was unavailable. QRF uses standard leaf-weighted quantiles instead of the author's dependency stack. Neither is labeled a full frame-h20 original-package reproduction.

From Workspace, first `source env.sh`.

```bash
python online_signed_state_20261117/state.py prepare
python online_signed_state_20261117/state.py fit
python online_signed_state_20261117/state.py calibrate
for split in calibration development late future; do
  python online_signed_state_20261117/state.py evaluate --split "$split"
  python online_signed_state_20261117/controls.py "$split"
done
python online_signed_state_20261117/audit.py
python online_signed_state_20261117/predict.py

python online_signed_state_20261117/geometric.py fit
python online_signed_state_20261117/geometric.py calibrate
for split in calibration development late future; do
  python online_signed_state_20261117/geometric.py evaluate --split "$split"
  python online_signed_state_20261117/geo_inference.py stream --split "$split"
done
python online_signed_state_20261117/geo_inference.py audit
python online_signed_state_20261117/geo_inference.py rollout

python online_signed_state_20261117/geometric.py fit --linear
python online_signed_state_20261117/geometric.py calibrate --linear
for split in calibration development late future; do
  python online_signed_state_20261117/geo_inference.py stream --linear --split "$split"
done
python online_signed_state_20261117/geo_inference.py audit --linear
python online_signed_state_20261117/geo_inference.py rollout --linear
python online_signed_state_20261117/report.py
```

Public actual inference, receiving physical u/v tensors:

```python
import sys
sys.path.insert(0, 'online_signed_state_20261117')
from geo_inference import GeometricPredictor

stream = GeometricPredictor(linear=True)  # frozen calibration-selected rolling state
forecast = stream.predict(observed_history, start=0)
# Only when all20 actual future frames have arrived:
stream.reveal(observed_future, time=40)
next_forecast = stream.predict(next_observed_history, start=20)
```

History/feedback/output shape is `[1,2,20,32,64]`. `normalized=True` accepts/returns the inherited normalization. The7-dimensional current residual is identified inside reveal, never passed in as a future oracle. For unobserved long rollouts, call `skip_feedback()` after predict and feed the issued forecast as the next input; it never creates a residual observation. Use separate predictor instances for separate trajectories. `linear=False` runs the80-step identification control. `predict.OnlinePredictor` implements the8-mode POD state family, whose calibration-selected model failed to improve heldout natural MSE.

All data are original real flows, seed42. Protocols and complete calibration grids are preserved. Small natural gains do not establish an advantage over the existing local warp, physical-state identifiability, independent validation, or stable long-term dynamics. The old batch-based nonlinear registration failed a tight inference replay tolerance; final geometry results therefore use direct single-window streams throughout, with the failed diagnostic retained. No default predictor was replaced or externally published.
