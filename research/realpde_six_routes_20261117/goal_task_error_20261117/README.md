# Goal-oriented empirical task errors

Results: `runs/goal_task_error_20261117/REPORT.md`. This pilot predicts four scalar task errors for the original observed20→20 forecast. It does not change the field, add observations, or certify a closed PDE. Primary selection was frozen on calibration before development evaluation; subsequent simple task-statistic controls are explicitly documented in `SIMPLE_CONTROL_ADDENDUM.md`. Initial model artifacts remain under `initial_*`.

Run from Workspace with the original checkpoint and existing cached splits available:

```bash
source env.sh
python goal_task_error_20261117/task_error.py coordinates
python goal_task_error_20261117/task_error.py prepare --split train
python goal_task_error_20261117/task_error.py prepare --split calibration
python goal_task_error_20261117/task_error.py fit
# Repeat prepare/evaluate for development, late, future; evaluate calibration too.
python goal_task_error_20261117/task_error.py prepare --split development
python goal_task_error_20261117/task_error.py evaluate --split development
# Apply disclosed additional controls, then evaluate all four splits again.
python goal_task_error_20261117/simple_controls.py
python goal_task_error_20261117/task_audit.py
python goal_task_error_20261117/benchmark.py
python goal_task_error_20261117/report.py
```

The actual point-cache dependency is `runs/active_observation_20261117/{split}.pt`; only its original issued P is reused. No new observation or assimilation output is used. Source X/Y and fixed deterministic controls are in `runs/error_structure_20261117`. Feature preparation explicitly replays every predictor at batch1 and verifies P exactly. This may take minutes on the A100.

With this source directory on `sys.path`:

```python
from task_stream import TaskErrorPredictor
predictor = TaskErrorPredictor()  # frozen primary selection
result = predictor.predict(history)  # CPU float32 physical [1,2,20,32,64]
task_values = result['task_estimates']  # GPU [4]
signed_errors = result['signed_error_estimate']
# Order: kinetic, temporal fluctuation, enstrophy, right-quarter mean u.
```

Other methods are explicit constructor arguments, e.g. `TaskErrorPredictor('direct/X+P/16')` or `TaskErrorPredictor('task_affine')`. The primary deterministic-feature method needs an additional four-call backbone; the XP/F_last models reuse the original forecast computation. Timing includes CPU physical input normalization and CPU task-value output, excluding model construction. The API accepts only history and has no target or feedback input.

`coordinates.json` records all68 train/cal/dev trajectory grids, nominal stored time spacing and missing physical metadata. Units are not guessed. `bounds.pt` distinguishes exact target-assisted inequalities from empirical calibration envelopes. Neither high observed coverage nor an exact Taylor identity establishes an out-of-distribution error certificate.

The main metric trades off tasks: the selected model improves quadratic energy statistics but worsens regional mean u. The 12/36-coefficient task-statistic controls obtain substantial gains themselves. Taylor L+Q is identical to the direct linear readout when the nonnegative clamp is inactive; dropping Q and using learned POD field correction both have strong negative results. These limitations are part of the retained result.
