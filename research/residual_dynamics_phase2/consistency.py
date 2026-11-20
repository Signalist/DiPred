"""Feasibility probe on observed physical fields: direct vs composed maps and JVPs."""
import torch
import torch.nn.functional as F
from common import aux, bootstrap, csvwrite, OUT, protocol, dump


def stencil(x):
    return F.unfold(F.pad(x,(1,1,1,1),mode='replicate'),3).transpose(1,2)


def apply_map(x,coef):
    inc=(stencil(x)@coef).transpose(1,2).reshape_as(x)
    return x+inc


def run():
    torch.manual_seed(137)
    f=aux('train')
    # Continuous physical span 400..459, with overlapping history included once.
    def sequence(fields):
        return torch.cat((fields['history'][:,0],fields['truth'][:,0],fields['truth'][:,1]),2).transpose(1,2)
    train=sequence(f).double()
    coeff=[]
    for lag in (1,2):
        gram=torch.zeros(18,18,device=train.device,dtype=train.dtype)
        xy=torch.zeros(18,2,device=train.device,dtype=train.dtype)
        for traj in train:
            x=traj[:-lag]; inc=traj[lag:]-x
            design=stencil(x).reshape(-1,18)
            y=inc.permute(0,2,3,1).reshape(-1,2)
            gram+=design.T@design
            xy+=design.T@y
        coeff.append(torch.linalg.solve(gram+1e-8*gram.diag().mean()*torch.eye(18,device=train.device),xy))
    rows=[]
    for split in ('development','test'):
        for r,traj in zip(protocol()['records'][split],sequence(aux(split)).double()):
            x=traj[:-2]; direct=lambda z:apply_map(z,coeff[1])
            composed=lambda z:apply_map(apply_map(z,coeff[0]),coeff[0])
            def mse(x,y):return float((x-y).square().mean())
            # Same randomized unit-RMS directions in both routes; these are model sensitivities.
            jv=[]
            for _ in range(4):
                v=torch.randn_like(x[:1])
                jd=torch.autograd.functional.jvp(direct,x[:1],v)[1]
                jc=torch.autograd.functional.jvp(composed,x[:1],v)[1]
                jv.append(mse(jd,jc)/mse(jd,0))
            rows.append(dict(split=split,trajectory=r['file'],Re=r['group'],
                persistence1=mse(x,traj[1:-1]),aux1=mse(apply_map(x,coeff[0]),traj[1:-1]),
                persistence2=mse(x,traj[2:]),direct2=mse(direct(x),traj[2:]),composed2=mse(composed(x),traj[2:]),
                path_gap=mse(direct(x),composed(x)),jvp_relative_squared_gap=sum(jv)/len(jv)))
    csvwrite(OUT/'auxiliary_physical_operator.csv',rows)
    dump(OUT/'auxiliary_fit.json',dict(coefficients=[c.tolist() for c in coeff],seed=137,
         role='physical velocity proxy; train-only closed form, no consistency loss, no diffusion changes',
         direct_vs_persistence=bootstrap([r['direct2']-r['persistence2'] for r in rows if r['split']=='test'])))
    print('Auxiliary direct/composed physical map and JVP evaluated on',len(rows),'trajectories')


if __name__=='__main__':
    run()
