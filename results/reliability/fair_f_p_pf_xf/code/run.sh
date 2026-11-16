#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
while [[ ! -f runs/fair_f_p_pf_xf_20261116/training_complete.json ]]; do
  if ! pgrep -f 'python fair_f_p_pf_xf/train.py' >/dev/null; then
    echo 'TRAINER_STOPPED_BEFORE_COMPLETION' >&2
    exit 1
  fi
  sleep 20
done
python fair_f_p_pf_xf/validate.py
python fair_f_p_pf_xf/evaluate.py
python fair_f_p_pf_xf/export.py
python fair_f_p_pf_xf/plot_cdf.py
python fair_f_p_pf_xf/audit.py
python fair_f_p_pf_xf/stage.py
printf 'ALL_COMPLETED_AND_AUDITED\n'
