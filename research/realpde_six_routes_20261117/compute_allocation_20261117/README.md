# Expected-benefit compute allocation

Run from Workspace after `source env.sh`. Main report: `runs/compute_allocation_20261117/REPORT.md`.

The frozen protocol selects on calibration only. Training uses92 existing real-data windows and seed42. The selected router predicts expensive-action benefit, using16 PCs of pooled X/P/F_last and17 supervised regression weights. PCA and normalization parameters are additional; all selected weights are in models.pt. No default mainline model is changed.

```bash
for split in train calibration development late future; do
  python compute_allocation_20261117/allocation.py prepare --split "$split"
done
python compute_allocation_20261117/allocation.py fit
for split in calibration development late future; do
  python compute_allocation_20261117/allocation.py issue --split "$split"
done
for split in calibration development late future; do
  python compute_allocation_20261117/allocation.py evaluate --split "$split"
done
python compute_allocation_20261117/benchmark.py
python compute_allocation_20261117/audit_report.py
```

Actual predictor:

```python
import sys
sys.path.insert(0, 'compute_allocation_20261117')
from predict import BudgetedPredictor

model = BudgetedPredictor()
# histories: physical u/v, [N,2,20,32,64], all available at the same issue time.
samples, info = model.predict_batch(histories, quota=0.5)
# samples is a list of N GPU tensors [32,2,20,32,64] in physical units.
# info['expensive'] records which requests used32 native diffusion paths.
```

`normalized=True` accepts/returns the inherited training normalization. `request_ids` provides stable unique IDs for tie handling; `sampling_seeds` fixes each cheap ensemble. `policy` can be benefit, risk, width, or the stateless seed42 random control. `expensive_action='secc3'` runs the frozen-router stronger-control sensitivity; the primary action is smooth. No targets, cached scores or future requests enter this API. Predicting a batch with N=1 cannot realize a fractional fixed quota; this API explicitly implements a synchronous batch service, not a streaming budget controller.

Raw/fixed/online evaluation reuses previously audited shared-calibration field scores and frozen masks. The public API above emits raw marginals; online calibration is not silently included. Physical max-speed, kinetic-energy and VS tradeoffs remain even where the primary mean improves. See REPORT.md and task_gains.csv.

Artifacts: selection_grid.csv (all calibration models), selection.json/models.pt (frozen choice), *_issued_masks.csv (target-free issue), *_windows.csv and *_trajectories.csv (complete scores), paired.csv (trajectory uncertainty), by_re.csv, representations.csv, priority_diagnostics.csv, timing.csv/timing_raw.json, independent accounting and dispatch audit JSONs. IMPLEMENTATION_NOTES.md records the random-control consistency fix and interleaved timing refinement without model refitting.
