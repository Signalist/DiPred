#!/usr/bin/env bash
# Resume-friendly full comparison. Failure leaves logs and completed outputs intact.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f env.sh ]]; then source env.sh; fi
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
out=runs/five_pde_reliability_20261114
scenarios=(cylinder controlled_cylinder fsi foil combustion)
for name in "${scenarios[@]}"; do
  for seed in 42 123 2026; do
    for arm in X XP XPF; do
      checkpoint="$out/heads/$name/seed_$seed/$arm.pt"
      while [[ ! -f "$checkpoint" ]]; do
        if ! pgrep -f "python five_pde_reliability/train_heads.py $name" > /dev/null; then
          printf 'MISSING_CHECKPOINT_AND_TRAINER_STOPPED %s\n' "$checkpoint" >&2
          exit 1
        fi
        sleep 30
      done
    done
  done
done
for name in "${scenarios[@]}"; do
  python five_pde_reliability/audit_preflight.py "$name"
  python five_pde_reliability/audit_noise.py "$name"
  python five_pde_reliability/validate.py "$name"
  python five_pde_reliability/evaluate.py "$name"
  for arm in 8-noise 32-noise 256-noise; do
    python five_pde_reliability/evaluate.py "$name" --arm "$arm"
  done
  python five_pde_reliability/benchmark.py "$name"
  python five_pde_reliability/audit_results.py "$name"
  python five_pde_reliability/summarize.py "$name"
  python five_pde_reliability/pair_by_window.py "$name"
  python five_pde_reliability/release.py "$name"
  python five_pde_reliability/verify_release.py "../../results/reliability/5pde/$name"
done
printf 'ALL_FIVE_SCENARIOS_COMPLETED\n'
