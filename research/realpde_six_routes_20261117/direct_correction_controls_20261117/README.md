# Direct residual and refinement controls

Results: `runs/direct_correction_controls_20261117/REPORT.md`. Frozen warp is compared with a stronger F-conditioned additive head, full-resolution XP additive network, exact imported original ResCrct, exact imported offline BlockResidualCorrector, and a small shared two-pass deterministic refiner. All use the same original RealPDE correction split and seed42. The original ResCrct is independently trained with zero feedback throughout for the no-feedback task; it is not a reproduction of the old online pipeline.

From Workspace:

```bash
source env.sh
python direct_correction_controls_20261117/benchmark.py --initial
python direct_correction_controls_20261117/controls.py fit
python direct_correction_controls_20261117/calibration_replay.py
for split in calibration development late future; do
  python direct_correction_controls_20261117/controls.py evaluate --split "$split"
done
python direct_correction_controls_20261117/rollout.py
python direct_correction_controls_20261117/audit.py
python direct_correction_controls_20261117/benchmark.py
python direct_correction_controls_20261117/report.py
```

`fit` skips completed frozen models. `calibration_replay` is a one-time step that must precede development scoring; it rejects an existing development result. Do not rerun it over a completed experiment. It verifies actual batch1 selection and preserves the prior batched choices. No gain changed in this run. To reproduce from scratch, use a separate output directory rather than overwriting the existing results.

Dependencies: original backbone checkpoint; `error_structure_20261117` X/Y, frozen warp/additive projections; `active_observation_20261117` actual original P; `goal_task_error_20261117` actual F_last. Only the issued point forecasts/features are reused—no sparse observation updates or task-corrected values. Both imported ResCrct classes come from `Shared`, and their source hashes are recorded.

With this source directory on `sys.path`:

```python
from controls import Predictor
predictor = Predictor('warp_F_last')
result = predictor.predict(history)  # CPU physical [1,2,20,32,64]
future = result['physical']         # GPU physical [1,2,20,32,64]
```

Methods: `base`, `warp_F_last`, `additive_F_last`, `dense_F`, `additive_XP`, `rescrct_original`, `rescrct_block`, `refine_two`. Each predictor consumes only the current history. There is no reveal or future-target argument; gain0 returns original P exactly. The dense_F model also depends on the shared frozen PCA projection; its tensor storage is counted separately in `storage_dependencies.json`.

Both complete flow latency and active correction latency determine the predeclared cost-comparable subset. Matching network-call counts, parameter counts, or optimizer updates is not claimed to equal matching FLOPs. All raw gain1 errors are retained. Free200-frame rollout is separately evaluated with each model's own predicted histories and no retuning. Its negative result prevents a long-rollout claim.
