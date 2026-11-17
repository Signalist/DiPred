"""Recheck dependence mechanics with current code, preserving historical audits."""
from pathlib import Path
import importlib.util
import json
import sys

W = Path(__file__).resolve().parent.parent
OUT = W / 'runs/research_completion_20261117'
sys.path.insert(0, str(W / 'lowcost_joint_20261117'))
import study
spec = importlib.util.spec_from_file_location('dependence_audit', W / 'lowcost_joint_20261117/audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    base = W / 'runs/lowcost_joint_20261117'
    rows = {}
    for label, suffix, ar, scorefit in [
        ('raw', '', False, False),
        ('conditional', 'conditional_scores', False, False),
        ('moment_ar', 'conditional_scores/ar', True, False),
        ('scorefit', 'conditional_scores/scorefit', False, True),
    ]:
        study.OUT = base / suffix
        study.AR_STAGE = ar
        study.SCOREFIT_STAGE = scorefit
        audit.OUT = study.OUT
        # Only redirect the audit output; fitting, selections, scores stay untouched.
        def capture(path, value, label=label):
            assert Path(path).name == 'mechanism_audit.json'
            rows[label] = value
            (OUT / (label + '_mechanism_audit.json')).write_text(json.dumps(value, indent=2) + '\n')
        audit.dump = capture
        print('CURRENT_MECHANICS', label, flush=True)
        audit.mechanics()
    assert len(rows) == 4 and all(v['passed'] for v in rows.values())
    (OUT / 'mechanics_refresh.json').write_text(json.dumps({
        'passed': True, 'stages': rows,
        'original_audit_files_preserved': True,
        'source_sha256': study.digest(W / 'lowcost_joint_20261117/study.py'),
        'audit_source_sha256': study.digest(W / 'lowcost_joint_20261117/audit.py'),
    }, indent=2) + '\n')
