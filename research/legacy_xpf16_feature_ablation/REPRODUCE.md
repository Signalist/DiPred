# Reproduction

The plotting script only needs the published CSV files and Python with pandas,
NumPy and Matplotlib:

```bash
python research/legacy_xpf16_feature_ablation/draw_from_csv.py results/reliability/legacy_xpf16_feature_ablation
```

Re-running training or raw-model evaluation additionally requires the original
workspace layout, RealPDE files, the frozen DiPred checkpoints, the original
819-example feature cache and legacy 49/250-window archives. The large dataset
and caches are not shipped. Source dependencies are copied into
`workspace_dependencies/`; place those files at the research workspace root,
place this directory at `workspace/legacy_xpf16_feature_ablation`, and make the
repository available at `workspace/github/DiPred`. The previous
`research/xpf16_independent49` code is also needed at
`workspace/xpf16_fullpercent_study`. Verify all paths and hashes in
`protocol.json` before running.

From the original workspace root:

```bash
source env.sh
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
python legacy_xpf16_feature_ablation/test_features.py
python legacy_xpf16_feature_ablation/train.py
for arm in X P F XP XF PF XPF; do
  python legacy_xpf16_feature_ablation/evaluate.py "$arm"
done
python legacy_xpf16_feature_ablation/recover_history.py
python legacy_xpf16_feature_ablation/summarize.py
python legacy_xpf16_feature_ablation/audit_publication.py
```

`train.py` resumes existing group checkpoints and never overwrites the original
XPF models. It writes only under its new run directory. Its historical runtime
started with groups of four and resumed with groups of sixteen; recorded group
wall times should not be added as independent head durations. Each head has
separate parameters, gradient clipping and optimizer moments. Online feedback
is applied only after all 20 frames of a window have been revealed.

To restore newly trained weights, run `restore_weights.py` from the result
folder; it verifies every part and the complete archive before reporting success. Extract it with `tar -xzf new_weights.tar.gz` to obtain the per-arm, per-quantile checkpoints.
The 250-head baseline's pre-existing split archive can be restored with its own
`research/xpf16_02grid_2048/restore_result_parts.py` script.
