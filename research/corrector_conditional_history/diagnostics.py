"""Trajectory-isolated conditional regression and inference-time state summaries."""
from collections import defaultdict
from types import SimpleNamespace

import torch

from research.corrector_residual_optimization.fit import track_weights
from research.corrector_simple_baseline.fit import table


def observable(view):
    """Per-window physical inputs known at issue time."""
    r = view.true
    p = view.historical.double()
    e = r-p
    observation = view.data.x.double()
    native = view.data.p.double()
    axes = (1,2,3,4)
    mse = lambda x: x.square().mean(axes)
    total, missing = mse(r), mse(e)
    u, v = observation.square().mean((2,3,4)).unbind(1)
    return dict(history_R_rms=total.sqrt(),history_E_rms=missing.sqrt(),history_S3_energy=mse(p),
                history_S3_gain=total-missing,history_remaining_ratio=missing/total.clamp_min(1e-30),
                history_R_E_cosine=(r*e).mean(axes)/(total*missing).sqrt().clamp_min(1e-30),
                history_diff1_energy=mse(e.diff(dim=2)),history_diff2_energy=mse(e.diff(n=2,dim=2)),
                current_native_change=mse(native-observation[:,:,-1:]),
                current_S3_energy=mse(view.current.double()),observed_uv_ratio=(u/v.clamp_min(1e-30)).sqrt(),
                observed_speed_rms=mse(observation).sqrt(),
                observed_spatial_gradient=mse(observation.diff(dim=3))+mse(observation.diff(dim=4)))


@torch.inference_mode()
def observable_thresholds(view):
    fields=observable(view)
    assert all(torch.isfinite(x).all() for x in fields.values())
    return {name:torch.quantile(x,x.new_tensor([1/3,2/3])).tolist() for name,x in fields.items()}


def spatial_design(view, ix):
    current=view.base[ix].double().unflatten(1,(3,2))
    common=current.unbind(1)
    s3=view.current[ix].double()
    history=view.true[ix]
    remaining=history-view.historical[ix].double()
    target=view.target[ix]-s3
    scale=view.scale
    values=(torch.ones_like(s3),*common,s3/scale,history/scale,remaining/scale,target/scale)
    return torch.stack(values,dim=-1).flatten(3,4)


@torch.inference_mode()
def ridge(out, view, penalty):
    cases=view.data.cases
    by_track=defaultdict(list)
    for i,case in enumerate(cases):
        by_track[case["file"]].append(i)
    refs=view.identity["refs"]
    channels=("u","v")
    designs={"current":[0,1,2,3,4],"current_R":[0,1,2,3,4,5],"current_E":[0,1,2,3,4,6]}
    output=[]
    for fold in refs:
        tracks=sorted(fold["predicted"])
        assert len(tracks)>=2 and set(tracks).isdisjoint(fold["train"])
        for side in (0,1):
            active, held=tracks[side::2],tracks[1-side::2]
            statistics=view.target.new_zeros((2,20,8,8))
            for file in active:
                ids=by_track[file]
                assert set(ids)<=set(fold["ids"])
                per_track=torch.zeros_like(statistics)
                for first in range(0,len(ids),4):
                    samples=spatial_design(view,ids[first:first+4])
                    matrix=samples.permute(1,2,0,3,4).flatten(2,3)
                    per_track+=matrix.transpose(-1,-2)@matrix
                statistics+=per_track/(len(ids)*samples.shape[3]*len(active))
            for name, cols in designs.items():
                gram=statistics[...,cols,:][...,:,cols]
                rhs=statistics[...,cols,7]
                regularization=torch.eye(len(cols),device=gram.device,dtype=gram.dtype)*penalty
                regularization[0,0]=0.
                beta=torch.linalg.solve(gram+regularization,rhs.unsqueeze(-1)).squeeze(-1)
                for file in held:
                    ids=by_track[file]
                    assert set(ids)<=set(fold["ids"])
                    results=[]
                    for first in range(0,len(ids),4):
                        val=spatial_design(view,ids[first:first+4])
                        estimate=(val[...,cols]*beta[None,:,:,None,:]).sum(-1)
                        results.append((estimate-val[...,7]).square().mean(-1)*view.scale**2)
                    mse=torch.cat(results).mean(0).tolist()
                    for c, values in zip(channels,mse):
                        output.extend(dict(fold=fold["fold"],split=side,model=name,trajectory=file,
                            re=cases[ids[0]]["re"],channel=c,lead=k,mse=value,windows=len(ids))
                            for k,value in enumerate(values,1))
    assert len(output)==len(by_track)*3*2*20
    table(out/"conditional_history_scores.csv",output)


def gradient_ratios(engine, records, donors):
    """Probe the next eight registered batches at the auxiliary branch point."""
    from research.corrector_incremental_factorial.core import load_head, step_row
    from research.corrector_incremental_structure.model import inputs, labels

    rows=[]
    for name, record in records.items():
        config=record["config"]["model"]
        if not config["weights"]:
            continue
        key,weight=next(iter(config["weights"].items()))
        start=record["config"]["first"]
        model=load_head(record,step_row(record,start),engine.basis,engine.device)
        for update in range(start,start+8):
            ix=engine.schedule[update].to(engine.device)
            history=donors[update].to(engine.device) if "_shuffled_" in name else None
            target=labels(engine.train,ix,config)
            estimates=[model(inputs(engine.train,ix,config,history_ix=history))]
            if config.get("paired_history"):
                estimates.append(model(inputs(engine.train,ix,config,history_available=False)))
            mse=sum((e-target).square().mean() for e in estimates)/len(estimates)
            aux=sum((e.square().mean() if key=="energy" else (e-target).diff(dim=2).square().mean())
                    for e in estimates)/len(estimates)
            gm=torch.autograd.grad(mse,tuple(model.parameters()),retain_graph=True)
            ga=torch.autograd.grad(aux,tuple(model.parameters()))
            a,b=(float(torch.cat([v.flatten() for v in group]).norm()) for group in (gm,ga))
            assert a>0 and all(torch.isfinite(v).all() for v in (*gm,*ga))
            rows.append(dict(arm=record["config"]["name"],branch_step=start,update=update,term=key,
                             coefficient=weight,mse_grad=a,aux_grad=b,ratio=weight*b/a))
    return rows
