"""Freeze hashes and provenance for the completed valid training comparison."""
import hashlib,json
from pathlib import Path
from .common import OUT,ROOT,dump,digest

def main():
    protected=json.loads((ROOT/'results/trust_causal_audit_v2/PROTECTED_BEFORE.json').read_text())
    current={name:digest(ROOT/name) for name in protected}
    assert current==protected
    dump(OUT/'V2_PROTECTION.json',dict(file_count=len(protected),all_hashes_unchanged=True,files=protected))
    manifest=json.loads((ROOT/'results/trust_causal_audit_v2/WINDOW_MANIFEST.json').read_text())
    feedback=[]
    for x in manifest:
        if x['split'] in ('train','development'):
            feedback.append(dict(split=x['split'],trajectory_id=x['trajectory_id'],start=x['start'],
               previous_cache=x['previous_cache'],previous_cache_row=x['previous_cache_row'],
               previous_prediction_interval=x['previous_prediction_interval'],history_interval=x['history_interval'],
               previous_path_sha256=x['previous_path_sha256'],history_sha256=x['history_sha256'],
               raw_risk_sha256=x['raw_risk_sha256'],risk_sha256=x['risk_sha256']))
    dump(OUT/'FEEDBACK_IDENTITIES.json',feedback)
    weights=[]
    for arm in ('G_context','G_trust'):
        for x in json.loads((OUT/f'checkpoints_{arm}.json').read_text()):weights.append(x)
    dump(OUT/'WEIGHT_HASHES.json',weights)
    hashes={}
    for directory in (ROOT/'research/trust_routing_retrain_v2',OUT):
        for p in sorted(directory.rglob('*')):
            if p.is_file() and p.name not in ('ARTIFACT_HASHES.json','FINAL_PACKAGE.json') and '__pycache__' not in p.parts:
                hashes[str(p.relative_to(ROOT))]=digest(p)
    dump(OUT/'ARTIFACT_HASHES.json',hashes)
    dump(OUT/'FINAL_PACKAGE.json',dict(passed=True,ready_for_retraining_input_consumed=True,
        valid_training_updates=2000,training_seed=42,arms=['G_context','G_trust'],
        candidate_steps=[0,250,500,1000],selected_native_for_both=True,
        formal_test_started=False,feedback_records=len(feedback),weight_records=len(weights),
        artifact_hash_records=len(hashes),old_v2_results_untouched=True))
    print('PACKAGE FROZEN:',len(feedback),'feedback records;',len(weights),'weights;',len(hashes),'hashes',flush=True)

if __name__=='__main__':main()
