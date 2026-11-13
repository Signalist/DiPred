#!/usr/bin/env bash
set -euo pipefail
source env.sh
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
python fair_x_xp_xpf/benchmark_modes.py
python fair_x_xp_xpf/summarize.py
cp fair_x_xp_xpf/README_RESULTS.md github/DiPred/results/reliability/fair_x_xp_xpf/README.md
cp fair_x_xp_xpf/finalize.sh github/DiPred/results/reliability/fair_x_xp_xpf/code/finalize.sh
python fair_x_xp_xpf/pair_by_window.py
python fair_x_xp_xpf/publish_extras.py
python fair_x_xp_xpf/audit_runner.py
python fair_x_xp_xpf/extra_audit.py
python fair_x_xp_xpf/report.py
python fair_x_xp_xpf/audit_report.py
python github/DiPred/results/reliability/fair_x_xp_xpf/code/recalculate.py github/DiPred/results/reliability/fair_x_xp_xpf
python fair_x_xp_xpf/compact_frame_table.py
python fair_x_xp_xpf/pack_frame_precision.py
python fair_x_xp_xpf/audit_wide_frames.py
python fair_x_xp_xpf/finalize_hashes.py
