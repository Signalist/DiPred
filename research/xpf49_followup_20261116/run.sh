#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
python xpf49_followup_20261116/prepare.py
python xpf49_followup_20261116/audit_xpf.py
python xpf49_followup_20261116/audit_grid250.py
python xpf49_followup_20261116/query_xpf.py train
python xpf49_followup_20261116/query_xpf.py evaluate
python xpf49_followup_20261116/ecc.py
python xpf49_followup_20261116/local.py
python xpf49_followup_20261116/benchmark.py
python xpf49_followup_20261116/analysis.py
python xpf49_followup_20261116/verify.py
python xpf49_followup_20261116/report.py
python xpf49_followup_20261116/completion_audit.py
