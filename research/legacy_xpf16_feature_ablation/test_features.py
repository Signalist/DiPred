"""Check legacy equivalence and prohibited-source invariance before training."""
import sys,torch
from pathlib import Path
from functorch import combine_state_for_ensemble,vmap
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'xpf16_fullpercent_study'))
from experiment_selected16_quantile import new_head
from features import ARMS,AblatedHead,mask_base
from clean_reliability_common import configure

def main():
    configure();torch.manual_seed(42)
    head=new_head('compact16',None).cuda().eval()
    assert sum(p.numel() for p in head.parameters())==34004
    base=torch.randn(2,14,20,8,8,device='cuda');last=torch.randn(2,48,8,8,device='cuda')
    a=head.bounds(base,last,None,None,False);b=AblatedHead(head,'XPF')(base,last)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    for arm in ARMS:
        wrapper=AblatedHead(head,arm);before=wrapper(base,last)
        changed=base.clone();other=last.clone()
        if 'X' not in arm:
            changed[:,4:6]+=23;changed[:,10:12,0]+=17
        if 'P' not in arm:changed[:,8:12]+=31
        if 'F' not in arm:other+=47
        after=wrapper(changed,other)
        assert all(torch.equal(x,y) for x,y in zip(before,after)),arm
        assert sum(p.numel() for p in wrapper.parameters())==34004
    # Batched parameters and losses retain independent per-model gradients.
    models=[AblatedHead(new_head('compact16',None).cuda(),arm) for arm in ('X','X','X','X')]
    fun,params,buffers=combine_state_for_ensemble(models)
    def call(p,b):return fun(p,b,base,last)
    batched=vmap(call)(params,buffers)
    serial=[m(base,last) for m in models]
    for i in range(4):
        for j in range(2):
            d=(batched[j][i]-serial[i][j]).abs().max().item()
            print('vmap_diff',i,j,d,flush=True)
            assert torch.allclose(batched[j][i],serial[i][j],rtol=1e-5,atol=2e-6)
    # Compare independent AdamW updates including separate gradient clipping.
    from train49 import pinball
    seedhead=new_head('compact16',None).cuda();initial=seedhead.state_dict()
    for arm in ('X','P','F','XP','XF','PF'):
        batch_models=[];serial_models=[]
        for _ in range(4):
            a=new_head('compact16',None).cuda();a.load_state_dict(initial)
            b=new_head('compact16',None).cuda();b.load_state_dict(initial)
            batch_models.append(AblatedHead(a,arm));serial_models.append(AblatedHead(b,arm))
        f,ps,bs=combine_state_for_ensemble(batch_models);ps=tuple(q.requires_grad_() for q in ps)
        opt=torch.optim.AdamW(ps,lr=3e-4,weight_decay=1e-5)
        target=torch.randn(2,2,20,8,8,device='cuda');alpha=torch.tensor([.01,.05,.25,.49],device='cuda').reshape(4,1,1,1,1,1)
        def apply(p,b):return f(p,b,base,last)
        lo,hi=vmap(apply)(ps,bs);loss=.5*(pinball(target[None]-lo,alpha)+pinball(target[None]-hi,1-alpha)).flatten(1).mean(1);loss.sum().backward()
        norm=torch.zeros(4,device='cuda')
        for q in ps:
            if q.grad is not None:norm+=q.grad.flatten(1).square().sum(1)
        factor=(1/(norm.sqrt()+1e-6)).clamp(max=1)
        for q in ps:
            if q.grad is not None:q.grad.mul_(factor.reshape(4,*([1]*(q.ndim-1))))
        opt.step()
        for i,m in enumerate(serial_models):
            so=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-5)
            l,h=m(base,last);a=float(alpha[i,0,0,0,0,0]);loss=.5*(pinball(target-l,a)+pinball(target-h,1-a)).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);so.step()
            for q,serial in zip(ps,m.parameters()):assert torch.allclose(q[i],serial,atol=3e-6,rtol=1e-5),arm
    print('PASS XPF exact, masked-source invariance, 34004 params, batched/serial optimizer updates')
if __name__=='__main__':main()
