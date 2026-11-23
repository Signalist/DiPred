# Observed residual feedback

`run.py` is the experiment entry point. For a fresh replay, use a worktree at the reference commit in the protocol and copy this source directory and the new result directory's `PROTOCOL.md` into it. Supply the original real HDF5 data directory. The environment needs CUDA PyTorch, NumPy and h5py.

```bash
source env.sh
python research/observed_residual_feedback/run.py prepare --data /path/to/train_real
python research/observed_residual_feedback/run.py preflight
python research/observed_residual_feedback/run.py train
python research/observed_residual_feedback/run.py report
```

Preparation freezes the training schedules, candidate selection rule and file hashes. Preflight checks same-noise native equivalence, residual sign and scaling, the gradient paths and publication order. Training uses saved checkpoints only at 0, 250 and 500. The training command selects from development, freezes that choice and evaluates the old test once. The run guard rejects a second training invocation in the same output directory. The report command verifies the existing logs and scores, then rebuilds summaries on CPU.

The weight files overlay the frozen `reliability/checkpoints/current.pt` predictor. Each contains the trainable tensors, arm, step and protocol hash. `publication.jsonl.gz` records the head and correction hashes at issue time, then the revealed target and resulting state hashes. At issue time the complete member forecasts are buffered in memory. Score files contain summaries derived from those same forecasts, with no additional inference for channel, Re or lead results. The old audit repair reads its archived metrics on CPU.
