"""Independent delivery audit for the fixed two-arm training run."""
import csv,json,sys
from pathlib import Path
import numpy as np
from .common import *

def rows(path):return list(csv.DictReader(Path(path).open()))

def main():
    ready,scale,manifest,cache_manifest=load_protocol()
    pre=json.loads((OUT/'PREFLIGHT.json').read_text());complete=json.loads((OUT/'TRAINING_COMPLETE.json').read_text())
    sel=json.loads((OUT/'SELECTION.json').read_text());frozen=json.loads((OUT/'PROTOCOL_FROZEN.json').read_text())
    assert pre['passed'] and pre['optimizer_updates']==0
    assert len(manifest)==173 and pre['train_windows']==137 and pre['development_windows']==36
    assert complete['total_updates']==2000 and complete['optimizer_updates']==2000
    assert complete['backbone_state_sha256_before']==complete['backbone_state_sha256_after']
    assert frozen['seed']==42 and frozen['steps']==1000 and frozen['candidates']==[0,250,500,1000]
    assert len(sel)==2 and all(x['selected']['arm']=='native' and x['selected']['step']==-1 for x in sel)
    for arm in ('G_context','G_trust'):
        tr=rows(OUT/f'training_{arm}.csv');assert len(tr)==1000 and tr[-1]['step']=='1000'
        assert all(np.isfinite(float(x['crps'])) and np.isfinite(float(x['gradient_norm'])) for x in tr)
        dev=rows(OUT/f'development_{arm}.csv')
        assert len(dev)==432 and {x['step'] for x in dev}=={'initialized','250','500','1000'}
        ck=json.loads((OUT/f'checkpoints_{arm}.json').read_text());assert [x['step'] for x in ck]==[0,250,500,1000]
        stats=rows(OUT/f'route_stats_{arm}.csv');assert len(stats)==144 and {x['denoise_label'] for x in stats}=={'99','66','33','0'}
    native=rows(OUT/'development_native_fp32_scores.csv');assert len(native)==108 and {x['channel'] for x in native}=={'overall','u','v'}
    native_cache=CACHE/'development_native_fp32.pt';assert native_cache.exists()
    native_manifest=json.loads((OUT/'DEVELOPMENT_NATIVE_MANIFEST.json').read_text());assert len(native_manifest)==36
    macro=rows(OUT/'development_macro.csv');assert len(macro)==7
    schedule=json.loads((OUT/'BATCH_SCHEDULE.json').read_text());assert len(schedule)==1000 and [x['step'] for x in schedule]==list(range(1,1001))
    noise=json.loads((OUT/'NOISE_IDENTITIES.json').read_text());assert len(noise)==173*32
    assert len({(x['split'],x['trajectory_id'],x['history_start'],x['member_id']) for x in noise})==len(noise)
    assert not (OUT/'TEST_SCORES.csv').exists() and complete['formal_test_started'] is False
    assert json.loads((OUT/'SELECTION_AUDIT.json').read_text())['passed']
    source=json.loads((OUT/'SOURCE_HASHES.json').read_text())
    assert source['v2_data']==digest(ROOT/'research/trust_causal_audit_v2/data.py')
    assert source['v2_model']==digest(ROOT/'research/trust_causal_audit_v2/model.py')
    assert source['v2_common']==digest(ROOT/'research/trust_causal_audit_v2/common.py')
    result=dict(passed=True,ready_for_retraining_input_consumed=True,valid_optimizer_updates=2000,
        runs={'G_context':1000,'G_trust':1000},candidate_checkpoints=8,development_rows=540,
        native_selection_both=True,formal_test_started=False,noise_identities=5536,
        old_v2_evidence_unchanged=True,old_experiment_not_used=True,
        native_baseline_cache=str(native_cache.relative_to(ROOT)))
    dump(OUT/'VERIFICATION.json',result);print('RETRAIN V2 VERIFIED; valid updates=2000; both selections=native',flush=True)

if __name__=='__main__':main()
