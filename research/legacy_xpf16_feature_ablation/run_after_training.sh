#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
RUN=runs/legacy_xpf16_feature_ablation_20261116
while [[ ! -f "$RUN/training_complete.json" ]]; do
  if ! pgrep -f '^python legacy_xpf16_feature_ablation/train.py$' >/dev/null; then
    echo 'Training process stopped before completion; resume train.py.' >&2
    exit 1
  fi
  sleep 30
done
for arm in X P F XP XF PF XPF; do
  python legacy_xpf16_feature_ablation/evaluate.py "$arm"
done
python legacy_xpf16_feature_ablation/recover_history.py
python legacy_xpf16_feature_ablation/summarize.py
