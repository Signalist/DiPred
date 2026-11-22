# Coupled feedback audit

Run from the repository root:

```bash
PYTHONPATH=.cache/coupled_state_feedback_dependencies:$PYTHONPATH \
  python research/coupled_feedback_audit/run.py run
PYTHONPATH=.cache/coupled_state_feedback_dependencies:$PYTHONPATH \
  python research/coupled_feedback_audit/run.py report
PYTHONPATH=.cache/coupled_state_feedback_dependencies:$PYTHONPATH \
  python research/coupled_feedback_audit/run.py verify
```

The run is frozen-weight only. It reuses the prior development U rows for N/F/O/C_on and performs only five new 16-member forward groups per development window: N step0, C_on/C_off at steps500/1000, with step0 aliases reused. New C_off U and new C_on/C_off V metrics are computed from the same paths; no target is passed to generation. The entry creates no optimizer and never accesses the old test set.

`report` averages windows within trajectory before paired bootstrap and writes the compact comparison, channel, Re and lead tables under `results/coupled_feedback_audit/`. `verify` checks the frozen HEAD, checkpoint hashes, row identities, no updates, and the artifact hash manifest.

The raw 20-lead intermediate is published as `lead_metrics.csv.gz`; `lead_summary.csv` is the directly readable published table.
