"""Bind the completed bounded exploration to its reviewed artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import json
import re
import pandas as pd
from audit import W, OUT, sha

record = json.loads((OUT / 'audit.json').read_text())
assert record['passed']
assert record['source_sha256'] == sha(Path(__file__).with_name('audit.py'))
assert record['requirements_sha256'] == sha(Path(__file__).with_name('requirements.tsv'))
for name in ['verified_hashes.csv', 'evidence_inventory.csv']:
    for row in pd.read_csv(OUT / name).to_dict('records'):
        path = row.get('file', row.get('path'))
        assert sha(W / path) == row['sha256'], path
linked = 0
for p in [OUT / 'REPORT.md', OUT / 'REQUIREMENTS.md', Path(__file__).with_name('README.md')]:
    for link in re.findall(r'\]\(([^)]+)\)', p.read_text()):
        if not link.startswith(('http:', 'https:', '#')):
            assert (p.parent / link).is_file(), (p, link)
            linked += 1
for name in ['overview.png', 'overview.pdf']:
    assert (OUT / name).stat().st_size > 10000
attachment = W.parent.parent / '.app/.codex-runtime/home/attachments/e8488a22-5727-49ed-b165-d54aaa9e579c/pasted-text-1.txt'
artifact_files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in ['completion_manifest.json', 'PRE_COMPLETION_LEDGER.md'])
source_files = sorted(p for p in Path(__file__).resolve().parent.iterdir() if p.suffix in ('.py', '.md', '.tsv'))
manifest = dict(
    exploration_complete=True,
    scope='Original RealPDE small data; primarily single training seed42; simple models; all six routes explored.',
    completed_utc=datetime.now(timezone.utc).isoformat(),
    attachment=str(attachment), attachment_sha256=sha(attachment),
    requirements=43, recomputed_comparisons=record['recomputed_comparisons'],
    checked_existing_hashes=record['verified_existing_hashes'], checks=len(record['checks']),
    valid_local_document_links=linked,
    route_outcomes={
        'structured_correction': 'Small observed-history benefit; free-rollout negative.',
        'lowcost_joint': 'Measured cost-quality tradeoff; extreme-speed degradation.',
        'online_state': 'Geometric error tracking benefit; no complex-transition advantage.',
        'active_observation': 'Three bounded mechanisms fail to establish stable VOI superiority or sensor savings.',
        'compute_allocation': 'Equal-budget mean benefit; some trajectory CIs cross zero.',
        'physical_task_error': 'Empirical task-estimation benefit; strict PDE-certificate assumptions unavailable.',
    },
    not_established=['Fresh independent confirmation', 'Unique diffusion-F advantage',
                     'Long-term autonomous dynamics improvement', 'Reliable VOI sensor-efficiency superiority',
                     'Strict physical or conformal error guarantee', 'Universal failure of any research route'],
    artifacts={str(p.relative_to(W)): sha(p) for p in artifact_files},
    sources={str(p.relative_to(W)): sha(p) for p in source_files},
    primary_weights_replaced=False, external_publish=False,
)
(OUT / 'completion_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
print('FINAL_ARTIFACTS_VERIFIED', linked, 'document links;', len(artifact_files), 'artifacts;', len(source_files), 'sources')
