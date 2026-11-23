# Residual information study

Use a worktree at the protocol's reference commit and copy this source directory and the study's `PROTOCOL.md` into it for a fresh run. Use the original HDF5 data location pinned in the previous experiment. From the repository root, with CUDA PyTorch, NumPy and h5py available:

```bash
source env.sh
python research/residual_information_study/run.py all
```

The same entry point supports `prepare`, `A`, `B`, `C`, `freeze`, `test-A`, `test-C` and `report`. Complete the three development phases before sealing test selection. The single GPU lease serializes stages. The shared time ledger counts synchronized phase wall time, including CPU and I/O in each phase, against the budget. Training and publication logs remain distinct. Run `report` on the completed study to verify the artifacts and rebuild the report without inference.

A stores observed windows and native member statistics in an ignored cache. Only its fitted ridge parameters and compact result and identity logs are committed. B retains local factual input snapshots and publishes their hashes and aggregate diagnostics. C publishes trainable-tensor overlays for the exact pinned native checkpoint plus publication, training, and score logs. C reuses the prior N and J scores on the fixed windows. Final evaluation reads the archived test scores only after the unified selection gate.
