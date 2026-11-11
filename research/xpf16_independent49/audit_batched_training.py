"""Check parallel-model optimizer matches independent serial optimization."""
import json
import torch
from functorch import vmap
from clean_reliability_common import configure
from train49 import OUT, pinball
from train49_batched import Wrapper, stack

configure()
initial=torch.load(OUT.parent/'selected16_quantile_20261110/compact16_initial.pt',map_location='cuda',weights_only=False)['model']
alphas=[.01,.05,.25,.49]
serial=[Wrapper(initial).cuda().train() for _ in alphas]
par=[Wrapper(initial).cuda().train() for _ in alphas]
f,params,buffers=stack(par)
opts=[torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-5) for m in serial]
opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-5)
torch.manual_seed(42)
base=torch.randn(4,14,20,32,64,device='cuda');last=torch.randn(4,48,32,64,device='cuda');y=torch.randn(4,2,20,32,64,device='cuda')
for m,optimizer,a in zip(serial,opts,alphas):
    optimizer.zero_grad(set_to_none=True)
    lo,hi=m(base,last)
    loss=.5*(pinball(y-lo,a)+pinball(y-hi,1-a)).mean()
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);optimizer.step()
a=torch.tensor(alphas,device='cuda').reshape(4,1,1,1,1,1)
opt.zero_grad(set_to_none=True)
lo,hi=vmap(lambda p,b:f(p,b,base,last))(params,buffers)
loss=.5*(pinball(y.unsqueeze(0)-lo,a)+pinball(y.unsqueeze(0)-hi,1-a)).flatten(1).mean(1)
loss.sum().backward()
norm=torch.zeros(4,device='cuda')
for p in params:norm+=p.grad.flatten(1).square().sum(1)
factor=(1/(norm.sqrt()+1e-6)).clamp(max=1)
for p in params:p.grad.mul_(factor.reshape(4,*([1]*(p.ndim-1))))
opt.step()
maxerr=0.
for i,m in enumerate(serial):
    for p,q in zip(m.parameters(),params):
        maxerr=max(maxerr,float((p-q[i]).abs().max()))
assert maxerr<2e-6,maxerr
report={'passed':True,'maximum_one_step_parameter_difference':maxerr,'levels':alphas,
        'meaning':'separate weights and optimizer moments, per-model gradient norm, identical initial weights and input'}
(OUT/'batch_training_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
