"""Fit frozen, seeded corrective actions and gain/risk routers on train/cal only."""
from pathlib import Path
import sys,json,time,math,hashlib
import numpy as np
import pandas as pd
import torch
W=Path(__file__).resolve().parent.parent
for n in ['direct_correction_controls_20261117','error_structure_20261117','compute_allocation_20261117']:sys.path.insert(0,str(W/n))
import controls as ctl
import local_learn as local
import allocation as alloc
OUT=W/'runs/recoverability_20261118';OUT.mkdir(exist_ok=True)
SEEDS=[42,123,2025]

def meanerr(e,meta):return float(pd.DataFrame({'file':[m['file'] for m in meta],'e':e}).groupby('file').e.mean().mean())

def main():
 ctl.setup();torch.set_num_threads(4)
 data={s:ctl.data(s) for s in ['train','calibration']}
 # Use actual single-window F/P captures, same shared local projection for all seeds.
 pro=ctl.common_projection(); units=ctl.st.norm()['std'].square().mean()
 desc={}
 for s in data:
  x=data[s]['x'];p=data[s]['p'];feat=ctl.load(ctl.FEATURES/f'{s}.pt')['features']['F_last'].cuda()
  desc[s]=torch.cat([local.pooled(x),local.pooled(p),feat],1).cpu().double()
 b=alloc.basis_fit(desc['train'],16);z={s:alloc.project(desc[s],b) for s in data}
 results={};preds={}
 for seed in SEEDS:
  results[str(seed)]={};preds[str(seed)]={}
  for method in ['warp','rescrct_original']:
   torch.manual_seed(seed);head=(local.Head('warp') if method=='warp' else ctl.factory(method)).cuda();opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-5);rng=torch.Generator(device='cuda').manual_seed(seed);t=time.perf_counter();d=data['train']
   for step in range(2000):
    ix=torch.randint(len(d['meta']),(8,),device='cuda',generator=rng);lr=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*step/2000));opt.param_groups[0]['lr']=lr;opt.zero_grad()
    q=head(d['z'][ix],d['p'][ix]) if method=='warp' else head(d['x'][ix],d['p'][ix],d['z'][ix]);loss=ctl.mse(q,d['y'][ix]).mean()/units;loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
    if (step+1)%500==0:print('FIT',seed,method,step+1,float(loss),flush=True)
   torch.cuda.synchronize();elapsed=time.perf_counter()-t;head.eval();errors={};deltas={}
   with torch.no_grad():
    for split,d in data.items():
     p=d['p'];y=d['y'];qq=[]
     for i in range(0,len(p),8):
      q=head(d['z'][i:i+8],p[i:i+8]) if method=='warp' else head(d['x'][i:i+8],p[i:i+8],d['z'][i:i+8]);qq.append((q-p[i:i+8]).cpu())
     delta=torch.cat(qq).cuda();base=ctl.mse(p,y).cpu().numpy();errors[split]={str(g):ctl.mse(p+g*delta,y).cpu().numpy() for g in (0.,.25,.5,1.)};deltas[split]=base
   gain=min([0.,.25,.5,1.],key=lambda g:meanerr(errors['calibration'][str(g)],data['calibration']['meta']))
   ck={'model':head.state_dict(),'gain':gain,'seed':seed,'method':method,'projection':pro if method=='warp' else None,'train_seconds':elapsed,'protocol_sha256':ctl.digest(W/'recoverability_20261118/PROTOCOL.md')};ctl.save(ck,OUT/f'{method}_{seed}.pt')
   preds[str(seed)][method]={'gain':gain,'errors':errors,'base':deltas};results[str(seed)][method]={'gain':gain,'train_seconds':elapsed,'calibration':{g:meanerr(v,data['calibration']['meta']) for g,v in errors['calibration'].items()}}
   del head,opt,delta;torch.cuda.empty_cache()
  # Train risk and gain predictors using actual difference of P and the seeded warp.
  item=preds[str(seed)]['warp'];gain=item['gain'];train=item['base']['train']-item['errors']['train'][str(gain)];calbase=item['base']['calibration'];calerr=item['errors']['calibration'][str(gain)];models={}
  for role,target in [('benefit',train),('risk',item['base']['train'])]:
   candidates=[]
   for alpha in [.01,.1,1.,10.]:
    coef=alloc.ridge(z['train'],torch.tensor(target),alpha);priority=(torch.cat([torch.ones_like(z['calibration'][:,:1]),z['calibration']],1)@coef).numpy();mask=alloc.choose(data['calibration']['meta'],priority,.5);loss=meanerr(np.where(mask,calerr,calbase),data['calibration']['meta']);candidates.append((loss,alpha,coef))
   best=min(candidates,key=lambda t:t[0]);models[role]={'alpha':best[1],'coef':best[2],'calibration_loss':best[0],'candidates':[(a,b) for a,b,_ in candidates]}
  ctl.save({'projection':b,'models':models,'seed':seed},OUT/f'router_{seed}.pt');results[str(seed)]['routers']={k:{a:v for a,v in m.items() if a!='coef'} for k,m in models.items()}
  ctl.dump(OUT/'fit_progress.json',results);print('FROZEN_SEED',seed,results[str(seed)],flush=True)
 ctl.save(preds,OUT/'train_cal_errors.pt');ctl.dump(OUT/'selection.json',{'seeds':SEEDS,'results':results,'protocol_sha256':ctl.digest(W/'recoverability_20261118/PROTOCOL.md'),'test_access_before_freeze':False,'sources':{f.name:ctl.digest(f) for f in (W/'recoverability_20261118').glob('*.py')},'weights':{f.name:ctl.digest(f) for f in OUT.glob('*.pt')}});print('TRAIN_CAL_FROZEN',flush=True)
if __name__=='__main__':main()
