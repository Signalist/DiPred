"""Train an X-only IQN adaptation on the same 819 train trajectory windows."""
import json,math,os,sys,time
from pathlib import Path
import torch
from model import XOnlyIQN,quantile_huber
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from clean_reliability_common import configure,digest,manifest
from experiment_reliability_transport_e12 import FIT_RECORDS,FIT_STARTS,normalization
from experiment_extended_train_error_adapters import LATE_RECORDS,LATE_STARTS
from export_aligned_test_distribution import from_hdf
OUT=ROOT/'runs/xonly_iqn_20261112'
STEPS=2000


def data(norm):
    xs=[];ys=[]
    for starts,records in ((FIT_STARTS,FIT_RECORDS),(LATE_STARTS,LATE_RECORDS)):
        for start in starts:
            xs.append(((from_hdf(records,start)-norm['mean'])/norm['std']).cpu())
            ys.append(((from_hdf(records,start+20)-norm['mean'])/norm['std']).cpu())
    x=torch.cat(xs).cuda();y=torch.cat(ys).cuda()
    assert len(x)==819 and x.shape==y.shape
    return x,y


def atomic(value,path):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)


def main():
    configure();OUT.mkdir(parents=True,exist_ok=True)
    splits=manifest()['files']
    assert {r['group'] for r in splits['train']}.isdisjoint({r['group'] for k in ('calibration','development','test') for r in splits[k]})
    norm=normalization();x,y=data(norm)
    torch.manual_seed(42);model=XOnlyIQN().cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    sampler=torch.Generator(device='cuda').manual_seed(42)
    checkpoint=OUT/'progress.pt';step0=0;elapsed=0.;logs=[]
    if checkpoint.exists():
        old=torch.load(checkpoint,map_location='cuda',weights_only=False)
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
        sampler.set_state(old['sampler'].cpu());step0=old['step'];elapsed=old['elapsed'];logs=old['logs']
    begin=time.monotonic()
    for step in range(step0+1,STEPS+1):
        ids=torch.randint(len(x),(4,),device='cuda',generator=sampler)
        tau=torch.rand(4,device='cuda',generator=sampler).clamp(.002,.998)
        optimizer.param_groups[0]['lr']=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/STEPS))
        optimizer.zero_grad(set_to_none=True)
        loss=quantile_huber(model(x[ids],tau),y[ids],tau)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        if not torch.isfinite(gradient):raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        if step==1 or step%100==0:
            logs.append(dict(step=step,loss=float(loss),elapsed=elapsed+time.monotonic()-begin))
            print('XONLY_IQN_TRAIN',step,float(loss),flush=True)
        if step%250==0:
            atomic(dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.get_state(),logs=logs,elapsed=elapsed+time.monotonic()-begin),checkpoint)
    elapsed+=time.monotonic()-begin
    final=OUT/'model.pt';atomic(dict(model=model.state_dict(),seed=42,steps=STEPS,examples=819,loss=logs,elapsed=elapsed,parameters=sum(p.numel() for p in model.parameters())),final)
    (OUT/'training.json').write_text(json.dumps(dict(seed=42,steps=STEPS,examples=819,parameters=sum(p.numel() for p in model.parameters()),seconds=elapsed,model_sha256=digest(final),loss=logs,quantile_loss='quantile Huber kappa=0.1',inputs='history X only; no DiPred prediction or features',tau='Uniform(.002,.998) via clipping',caveat='PDE adaptation of IQN, not reproduction of RL experiments'),indent=2)+'\n')
    print('XONLY_IQN_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
