# Coupled state feedback

Run from repository root with the repository PyTorch environment. One entry point:

```bash
source env.sh
python research/coupled_state_feedback/run.py prepare --data /path/to/train_real
python research/coupled_state_feedback/run.py preflight
python research/coupled_state_feedback/run.py train
python research/coupled_state_feedback/run.py report
python research/coupled_state_feedback/run.py verify
```

`prepare` freezes existing 46/12/12 records, frame-availability windows, seed42 schedule and hashes. `preflight` uses only a real training window, checks M8 memory and discards its two smoke updates. `train` refuses to restart after TRAIN_STARTED, runs F/O/C in common rounds, evaluates fixed candidates, freezes selection, then generates the old exploratory test once. Every sample resets R; generation APIs accept no truth. The complete U objective includes gradients of all member-pair terms. `report` and `verify` are CPU-only and never regenerate predictions.

No old source, weights, results, caches or untracked work is overwritten. New prediction caches are unnecessary. Saved weights contain all trainable tensors; reconstruct with the hash-verified original checkpoint plus `Forecast` and overlay the saved state. Checkpoint file and config hashes are recorded in events/COMPLETE. Full formal training is deliberately not automatically resumable or rerunnable. The fixed protocol and limitations are in `results/coupled_state_feedback/PROTOCOL.md`.
