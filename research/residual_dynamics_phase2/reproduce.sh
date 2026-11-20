#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../../env.sh
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
parent=$(realpath ../../runs)
run_dir=$(mktemp -d "$parent/phase2_replay.XXXXXXXX")
export DIPRED_PHASE2_OUT="$run_dir/results" DIPRED_PHASE2_CACHE="$run_dir/cache"
mkdir -p "$DIPRED_PHASE2_OUT"
printf 'Replay directory: %s\n' "$run_dir"
stage() {
    local log=$1
    shift
    python "research/residual_dynamics_phase2/$1.py" "${@:2}" 2>&1 | tee "$DIPRED_PHASE2_OUT/$log.log"
}
stage unit_tests tests
stage audit audit
stage mechanism_fit mechanisms fit
stage mechanism_score mechanisms score
stage decomposition decomposition
stage causal_fit causal fit
stage causal_score causal score
stage consistency consistency
stage coupling coupling
stage verify verify
