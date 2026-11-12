"""499 marginal levels as 249 independent XPF16 pairs and one median head.

The 49 prior integer-percent pairs and the just-trained .2% pair are reused:
all were independently fitted under the same fixed seed, initialization,
optimizer, training data and 2,000-step protocol. Every other level is fitted
from the identical pretraining initialization, without distillation or folds.
"""
import json,sys,time
from pathlib import Path
import torch
from clean_reliability_common import ROOT,configure,digest,dump,manifest
from experiment_reliability_transport_e12 import FIT_RECORDS,normalization
from experiment_extended_train_error_adapters import LATE_RECORDS
from temporal_mean_memory_control import training_data

sys.path.insert(0,str(ROOT/'xpf16_fullpercent_study'))
import train49_batched as batch

OUT=ROOT/'runs/xpf16_02grid_2048_20261112'
PREVIOUS=ROOT/'runs/xpf16_fullpercent_20261111'
EXTREME=ROOT/'runs/xpf16_extreme_002_2048_20261112'


def configs():
    result=[]
    for k in range(1,251):
        alpha=k/500
        label=f'XPF16-P{k:03d}'
        result.append(dict(id=label,k=k,lower=alpha,upper=1-alpha,target_coverage=1-2*alpha,
            seed=42,steps=2000,batch=4,learning_rate=3e-4,weight_decay=1e-5,grad_clip=1.0,
            median_head=(k==250),percent=20*k/100))
    assert len(result)==250 and abs(result[0]['lower']-.002)<1e-12 and result[-1]['lower']==.5
    return result


def reuse(cfg,source):
    folder=OUT/'models'/cfg['id'];folder.mkdir(parents=True,exist_ok=True)
    dest=folder/'model.pt'
    if dest.exists():return
    raw=torch.load(source/'model.pt',map_location='cpu',weights_only=False)
    assert abs(raw['config']['lower']-cfg['lower'])<1e-12
    assert abs(raw['config']['upper']-cfg['upper'])<1e-12
    assert raw['config']['steps']==2000 and raw['config']['seed']==42 and raw['parameters']==34004
    raw['source_checkpoint_sha256']=digest(source/'model.pt')
    raw['source_original_config']=raw['config']
    raw['config']=cfg
    torch.save(raw,dest)
    log=json.loads((source/'training.json').read_text())
    log['source_checkpoint_sha256']=raw['source_checkpoint_sha256']
    log['source_original_config']=log['config']
    log['config']=cfg;log['sha256']=digest(dest)
    dump(folder/'training.json',log)
    print('REUSED_INDEPENDENT_HEAD',cfg['id'],flush=True)


def main():
    configure();batch.OUT=OUT;OUT.mkdir(parents=True,exist_ok=True)
    cfgs=configs()
    assert json.loads((PREVIOUS/'independent_audit.json').read_text())['passed']
    assert json.loads((EXTREME/'training_complete.json').read_text())['passed']
    fit={r['file'] for r in FIT_RECORDS+LATE_RECORDS}
    held={r['file'] for split,rows in manifest()['files'].items() if split!='train' for r in rows}
    assert not fit & held
    dump(OUT/'protocol.json',dict(main_commit='66d39186e22f4f83e7fa45cae52756ed4ddb2ac6',
        source_sha256=digest(Path(__file__)),initial_checkpoint_sha256=digest(PREVIOUS.parent/'selected16_quantile_20261110/compact16_initial.pt'),
        configurations=cfgs,nominal_marginal_probabilities='0.2%,0.4%,...,99.8%; 50% separately trained',
        fit_ids=sorted(fit),training_pairs=819,steps=2000,seed=42,
        reuse='49 previous independently trained integer-percentage heads plus the freshly trained 0.2% head; all share same training protocol.',
        median='250th head uses alpha=.5 for both outputs; their midpoint is the dedicated learned 50% prediction.',
        caveat='Prior development and test IDs have been used during research; strict newly blind-test claims are not possible.'))
    reuse(cfgs[0],EXTREME/'models/XPF16-Q0p2')
    for k in range(5,250,5):
        reuse(cfgs[k-1],PREVIOUS/'models'/f'XPF16-Q{k//5:02d}')
    pending=[c for c in cfgs if not (OUT/'models'/c['id']/'model.pt').exists()]
    print('GRID_PENDING',len(pending),'REUSED',250-len(pending),flush=True)
    initial=torch.load(PREVIOUS.parent/'selected16_quantile_20261110/compact16_initial.pt',map_location='cuda',weights_only=False)['model']
    data=training_data(normalization())
    for i in range(0,len(pending),4):
        batch.train_group(pending[i:i+4],data,initial)
    assert all((OUT/'models'/c['id']/'model.pt').exists() for c in cfgs)
    dump(OUT/'training_complete.json',dict(passed=True,total_heads=250,interval_heads=249,
        median_heads=1,reused=50,trained_new=200,
        weights={c['id']:digest(OUT/'models'/c['id']/'model.pt') for c in cfgs}))
    print('ALL_250_INDEPENDENT_HEADS_COMPLETE',flush=True)

if __name__=='__main__':main()
