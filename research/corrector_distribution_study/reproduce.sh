#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 1
output=$(realpath -m -- "$1")
test ! -e "$output/FROZEN.json"
cd "$(git rev-parse --show-toplevel)"
source env.sh
export PYTHONPATH="$PWD/../Forecast/Runtime/.deps${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p -- "$output"
module=research.corrector_distribution_study.run
python -u -m "$module" --phase development --out "$output" 2>&1 | tee "$output/development.log"
cp -- results/corrector_distribution_study/EXPLORATION_LOG.md "$output/EXPLORATION_LOG.md"
python -u -m "$module" --phase exploration --out "$output" 2>&1 | tee "$output/exploration.log"
python -u -m "$module" --phase final --out "$output" 2>&1 | tee "$output/final.log"
python -m research.corrector_distribution_study.plot "$output"
