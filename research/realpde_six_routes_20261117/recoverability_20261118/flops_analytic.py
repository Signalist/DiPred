"""Analytic MAC accounting using actual Conv/Linear hook shapes, not wall-clock counters."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
W=Path(__file__).resolve().parent.parent
for n in ['direct_correction_controls_20261117','error_structure_20261117']:sys.path.insert(0,str(W/n))
import controls as ctl
import local_learn as loc
O=W/'runs/recoverability_20261118'

def calc(model,fn):
 records=[];hooks=[]
 def hook(m,inp,out):
  if not isinstance(out,torch.Tensor):return
  if isinstance(m,(nn.Conv1d,nn.Conv2d,nn.Conv3d)):
   kernel=int(np.prod(m.kernel_size));flops=2*out.numel()*(m.in_channels//m.groups)*kernel
  elif isinstance(m,nn.Linear):flops=2*out.numel()*m.in_features
  else:return
  records.append(dict(layer=type(m).__name__,input_shape=str(tuple(inp[0].shape)),output_shape=str(tuple(out.shape)),forward_flops=int(flops),input_requires_grad=bool(inp[0].requires_grad),parameter_requires_grad=bool(m.weight.requires_grad)))
 for m in model.modules():
  if isinstance(m,(nn.Conv1d,nn.Conv2d,nn.Conv3d,nn.Linear)):hooks.append(m.register_forward_hook(hook))
 fn()
 for h in hooks:h.remove()
 return records

def main():
 ctl.setup();d=ctl.data('calibration');x,p,z=d['x'][:8],d['p'][:8],d['z'][:8];net=ctl.st.model();ck=ctl.load(O/'warp_42.pt','cuda');warp=loc.Head('warp').cuda().eval();warp.load_state_dict(ck['model']);res=ctl.factory('rescrct_original').cuda().eval();res.load_state_dict(ctl.load(O/'rescrct_original_42.pt','cuda')['model']);rows=[];layers=[]
 for name,model,fn in [('DiPred',net,lambda:ctl.st.capture(net,x[:1])),('warp',warp,lambda:warp(z[:1],p[:1])),('rescrct',res,lambda:res(x[:1],p[:1],None)),('warp_train_batch8',warp,lambda:warp(z,p)),('rescrct_train_batch8',res,lambda:res(x,p,None))]:
  for param in model.parameters():param.requires_grad_(True)
  with torch.enable_grad():r=calc(model,fn)
  total=sum(v['forward_flops'] for v in r);back=sum(v['forward_flops']*(int(v['input_requires_grad'])+int(v['parameter_requires_grad'])) for v in r);rows.append(dict(method=name,forward_conv_linear_flops=total,estimated_forward_backward_conv_linear_flops=total+back,invocations=len(r),scope='2 FLOPs per multiply-add; forward hooks; excludes normalization, activations, interpolation/grid_sample, optimizer, PCA, loss, H2D and all other ops; backward estimated from conv/linear gradient multiplications, not a hardware measurement'));layers.extend([dict(method=name,**v) for v in r])
 pd.DataFrame(rows).to_csv(O/'analytic_conv_linear_flops.csv',index=False);pd.DataFrame(layers).to_csv(O/'analytic_flops_layers.csv',index=False);print('ANALYTIC_FLOPS',rows,flush=True)
if __name__=='__main__':main()
