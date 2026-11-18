"""Profiler operator FLOPs are partial estimates, not total GPU operations."""
from pathlib import Path
import sys,time,json
import numpy as np
import pandas as pd
import torch
W=Path(__file__).resolve().parent.parent
for n in ['direct_correction_controls_20261117','error_structure_20261117']:sys.path.insert(0,str(W/n))
import controls as ctl
import local_learn as loc
O=W/'runs/recoverability_20261118'

def main():
 ctl.setup();d=ctl.data('calibration');x,p,z=d['x'][:1],d['p'][:1],d['z'][:1];ck=ctl.load(O/'warp_42.pt','cuda');models={'warp':loc.Head('warp').cuda().eval(),'rescrct_original':ctl.factory('rescrct_original').cuda().eval()};models['warp'].load_state_dict(ck['model']);models['rescrct_original'].load_state_dict(ctl.load(O/'rescrct_original_42.pt','cuda')['model']);rows=[]
 for name,model in models.items():
  def call():return model(z,p) if name=='warp' else model(x,p,None)
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True) as prof:
   with torch.no_grad():call();torch.cuda.synchronize()
  forward=sum(e.flops for e in prof.key_averages() if e.flops)
  opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True) as prof:
   opt.zero_grad();call().square().mean().backward();opt.step();torch.cuda.synchronize()
  train=sum(e.flops for e in prof.key_averages() if e.flops)
  rows.append(dict(method=name,parameters=sum(v.numel() for v in model.parameters()),batch=1,forward_profiler_reported_flops=forward,training_step_profiler_reported_flops=train,updates=2000,training_batch_size=8,note='torch profiler reports supported operators only; backward grid/interpolation/conv ops may be uncounted; not total FLOPs; profiler training step here is batch1'))
 pd.DataFrame(rows).to_csv(O/'operator_flops_partial.csv',index=False);print('PARTIAL_FLOPS',rows,flush=True)
if __name__=='__main__':main()
