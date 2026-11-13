"""Verify every numeric table entry in the generated report against CSV."""
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'github/DiPred/results/reliability/fair_x_xp_xpf'
with (DEST/'summary.csv').open(newline='') as f:
    rows={(r['method'],r['seed'],r['channel']):r for r in csv.DictReader(f) if r['calibration']=='raw' and r['channel'] in ('u','v')}
found={}
for line in (DEST/'REPORT.md').read_text().splitlines():
    if not line.startswith('| '):continue
    parts=[p.strip() for p in line.strip('|').split('|')]
    if len(parts)!=6 or parts[0] not in ('X','XP','XPF') or parts[1] not in ('42','123','2026'):continue
    found[(parts[0],parts[1],parts[2])]=parts[3:]
ok=set(found)==set(rows)
for key,values in found.items():
    r=rows[key]
    for actual,name in zip(values,('crps','coverage90','interval_score90')):
        ok &= math.isclose(float(actual),float(r[name]),rel_tol=2e-6,abs_tol=5.1e-7 if name=='coverage90' else 5e-9)
result={'checks':{'report_matches_csv':ok,'expected_18_raw_channel_rows':len(found)==18},'passed':ok and len(found)==18}
(DEST/'report_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert result['passed']
