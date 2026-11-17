#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
python f_risk_query_20261116/prepare.py
python f_risk_query_20261116/risk.py
python f_risk_query_20261116/evaluate_risk.py
python f_risk_query_20261116/query.py prepare
python f_risk_query_20261116/query.py train
python f_risk_query_20261116/query.py evaluate
python f_risk_query_20261116/query_ensemble.py
python f_risk_query_20261116/query_decisions.py
python f_risk_query_20261116/local_calibration.py
python f_risk_query_20261116/curves.py
python f_risk_query_20261116/symmetric_control.py
python f_risk_query_20261116/benchmark.py
python f_risk_query_20261116/verify.py
python f_risk_query_20261116/report.py
python f_risk_query_20261116/completion_audit.py
