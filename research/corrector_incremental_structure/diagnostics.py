"""Train-only calibration, coupled timing diagnostics and physical interventions."""
from statistics import mean

import torch

from research.corrector_residual_feedback.model import predict as previous_predict
from research.corrector_spacetime_validation.fit import load_net
from research.corrector_spacetime_validation.gradients import cosine,vectors
from research.corrector_spacetime_validation.objectives import terms


def calibrate(head,z,y,batches,keys,rho):
    params=tuple(head.parameters())
    measurements={n:[] for n in ("mse",*keys)}
    for ix in batches:
        values=terms(head(z[ix]),y[ix],keys)
        for i,(name,value) in enumerate(values.items()):
            grad=torch.autograd.grad(value,params,retain_graph=i+1<len(values))
            norm=torch.stack([v.square().sum() for v in grad]).sum().sqrt()
            assert torch.isfinite(norm) and norm>1e-12
            measurements[name].append(float(norm))
    return {k:rho*mean(measurements["mse"])/mean(measurements[k]) for k in keys},measurements


def matched_e_rows(root,records,train,dev,z,dz,calibration):
    from research.corrector_residual_optimization.fit import track_weights
    grad,lead=[],[]
    train_target=(train.y-train.p)/calibration["scale"]
    datasets=[(name,data,design,track_weights(data))
              for name,data,design in (("train",train,z),("development",dev,dz))]
    for name,record in records.items():
        for step in (1000,2000,4000,6000):
            row=next(i for i in record["training"] if i["step"]==step)
            net=load_net(root/row["path"],row["sha256"],"E",train.p.device)
            term,slope=next(iter(record["config"]["weights"].items()))
            packets=vectors(net,z,train_target,calibration["batches"],(term,))
            for i,p in enumerate(packets):
                a,b=p["mse"][1],p[term][1]
                mse_norm,aux_norm=float(a.norm()),float(b.norm())
                total=(a+slope*b).norm()
                grad.append(dict(model=name,step=step,batch=i,mse_norm=mse_norm,auxiliary_norm=aux_norm,
                                 aux_to_mse=slope*aux_norm/mse_norm,
                                 cosine=cosine(a,b),total_norm=float(total),clipped=bool(total>1),
                                 coefficient=min(1.,1./(float(total)+1e-6)),
                                 train_mse=row["train_mse"],dev_mse=row["dev_mse"]))
            for split,data,design,weight in datasets:
                output=previous_predict(net,design,calibration["scale"])
                each=(data.p.double()+output.double()-data.y.double()).square().mean((1,3,4))
                lead.extend(dict(model=name,step=step,split=split,lead=i+1,physical_mse=v)
                            for i,v in enumerate((each*weight[:,None]).sum(0).tolist()))
    return grad,lead


@torch.inference_mode()
def relationships(view):
    """Uncentered association and centered Pearson on a fixed train pixel subsample."""
    from research.corrector_residual_optimization.fit import track_weights
    weight=track_weights(view.data).sqrt().float()
    past=view.true.float()[:,:,:,::4,::4]
    a=(past*weight[:,None,None,None,None]).movedim(2,0).flatten(1)
    m=(past*weight.square()[:,None,None,None,None]).mean((0,1,3,4))*len(past)
    result=[]
    for label,target in (("native_residual",view.target),("s3_remaining",view.target-view.current.double())):
        pix=target.float()[:,:,:,::4,::4]
        b=(pix*weight[:,None,None,None,None]).movedim(2,0).flatten(1)
        n=(pix*weight.square()[:,None,None,None,None]).mean((0,1,3,4))*len(pix)
        denom=past.shape[1]*past.shape[3]*past.shape[4]
        c=(b@a.T)/denom
        u=a.square().sum(1)/denom
        v=b.square().sum(1)/denom
        cos=(c/(v[:,None]*u).clamp_min(1e-30).sqrt()).tolist()
        corr=((c-n[:,None]*m)/((v-n.square())[:,None]*(u-m.square())).clamp_min(1e-30).sqrt()).tolist()
        result.extend(dict(target=label,source=i+1,lead=j+1,lag=20+j-i,C=float(c[j,i]),cosine=cos[j][i],
                           pearson=corr[j][i]) for j in range(20) for i in range(20))
    return result


@torch.inference_mode()
def intervention(net,view,cfg,name):
    from .model import inputs,residual
    from research.corrector_residual_optimization.fit import track_weights
    j=torch.arange(len(view.data.cases),device=view.current.device)
    fixed=residual(view,j,cfg,net(inputs(view,j,cfg)))
    w=track_weights(view.data)
    r=[]
    for i in range(20):
        raw=view.true.clone()
        raw[:,:,i]=0
        p=residual(view,j,cfg,net(inputs(view,j,cfg,intervened=raw)))
        delta=(((p-fixed).square().mean((1,3,4)))*w[:,None]).sum(0).sqrt().tolist()
        r.extend(dict(model=name,source=i+1,lead=k+1,physical_lag=20+k-i,response_rms=v)
                 for k,v in enumerate(delta))
    mapping=net.mapping().tolist()
    return r,[dict(model=name,source=i+1,lead=j+1,lag=20+j-i,value=mapping[j][i])
              for j in range(20) for i in range(20)]


def oracle(view,basis):
    from research.corrector_residual_optimization.fit import track_weights
    w=track_weights(view.data)
    rows=[]
    for label,p in (("direct_target",view.target),("increment_target",view.target-view.current.double())):
        c=(p.movedim(2,-1)@basis.to(p)).square().mean((1,2,3))
        levels=(c*w[:,None]).sum(0)
        rows.extend(dict(target=label,mode=i+1,energy=float(e),
                         residual_mse_rank4=float(levels[4:].sum()/20),
                         residual_mse_rank8=float(levels[8:].sum()/20)) for i,e in enumerate(levels))
    return rows
