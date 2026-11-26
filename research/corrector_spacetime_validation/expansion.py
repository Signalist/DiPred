"""Plan same-trajectory validation from metadata and release predictions in observation order."""
import h5py
import numpy as np
import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.data import LABELS,guard,write_tensor
from research.corrector_residual_feedback.model import ARMS,base,encode,predict
from research.corrector_window_study.data import read_json,tensor_sha


def plan_periods(study):
    source = study.triplets.source
    files = sorted(p.name for p in source.raw.glob("*.h5"))
    manifest = {r["file"] for rows in source.records.values() for r in rows}
    candidates = sorted(set(files)-manifest)
    inventory = source.out.parent/"corrector_distribution_study/CONFIRMATION_INVENTORY.json"
    documented = {r["file"]:r for r in read_json(inventory)["full_local_real_inventory"]}
    assert set(files)==set(documented)
    assert not read_json(inventory)["confirmation"]
    excluded = {}
    protocol = read_json(source.out.parent/"residual_dynamics/protocol.json")
    for group in protocol["exclusions"].values() if isinstance(protocol["exclusions"],dict) else [protocol["exclusions"]]:
        for entry in group:
            if isinstance(entry,dict) and "file" in entry:
                excluded[entry["file"]] = entry
    existing = {(c["file"],c["start"]) for c in study.triplets.plan["test"]["cases"]}
    length = study.plan["verification"]["feedback_chain"]
    plans = []
    for item in sorted(source.records["test"],key=lambda r:r["file"]):
        start = 400
        seq = [s for s in range(start,start+20*length,20) if s+40<=item["frames"]]
        assert len(seq)>=2
        plans.append(dict(file=item["file"],re=item["re"],frames=item["frames"],starts=seq,
                          new_vs_main=[s for s in seq if (item["file"],s) not in existing],
                          new_vs_prior_window_study=[s for s in seq if s>460]))
    return dict(kind="same_trajectory_temporal_expansion",split="historical_test",
                goals=dict(releases=length,data_access="after main candidate freeze"),
                independent=dict(verified=[],listed_outside_original_manifest=candidates,
                                 known_exclusions={k:excluded[k] for k in candidates if k in excluded},
                                 documented={k:documented[k] for k in candidates},
                                 provenance_sha256=sha(inventory),
                                 reason="No independently identified, eligible RealPDE trajectory with documented provenance"),
                chains=plans)


@torch.inference_mode()
def period_data(study,test):
    from types import SimpleNamespace
    frozen = guard(study.out)
    info = read_json(study.out/"period_plan.json")
    source = study.triplets.source
    reused = {}
    for i,c in enumerate(test.cases):
        key = c["file"],c["start"]
        reused[key] = (test.x[i],test.p[i])
        reused[key[0],key[1]-20] = (test.previous_x[i],test.previous_p[i])
    evidence = dict(freeze_sha256=sha(study.out/"FROZEN.json"),backbone=source.identity,releases=[],
                    planned_cache_binding=frozen["bindings"]["period_plan.json"])
    x,p,y,prior_x,prior_p,cases = [],[],[],[],[],[]
    calls = []
    device = test.p.device

    def observe(_,args):
        assert len(args[0])==len(args[1])==1
        if not calls:
            assert not torch.count_nonzero(args[0])
        calls.append(int(args[1].item()))

    hook = source.backbone.net.register_forward_pre_hook(observe)
    try:
        for item in info["chains"]:
            file = item["file"]
            start,stop = item["starts"][0]-20,item["starts"][-1]+40
            with h5py.File(source.raw/file,"r") as f:
                assert f["u"].shape[0]==f["v"].shape[0]==item["frames"]
                values = np.stack([f[k][start:stop,::2,::2] for k in ("u","v")]).astype("float32")
            field = torch.as_tensor(values,device=device)
            assert torch.isfinite(field).all()
            for s in [start,*item["starts"]]:
                u = field[:,s-start:s-start+20]
                key = file,s
                if key in reused:
                    previous,released = reused[key]
                    assert torch.equal(u,previous)
                    path = "previously certified"
                else:
                    cache = study.cache/"period_native"/f"{file}_{s}.pt"
                    cache.parent.mkdir(exist_ok=True)
                    signature = dict(x=tensor_sha(u),backbone=source.identity,zero=True,labels=LABELS)
                    meta = cache.with_suffix(".json")
                    if cache.exists() and meta.exists():
                        from .fit import atomic_json
                        old = read_json(meta)
                        assert old["signature"]==signature and old["sha256"]==sha(cache)
                        released = torch.load(cache,map_location=device,weights_only=True)
                    else:
                        from .fit import atomic_json
                        mean,std = study.norm["mean"],study.norm["std"]
                        calls.clear()
                        value = source.backbone.rollout((u[None]-mean)/std,seed=500000,sign=0)
                        assert calls==LABELS
                        released = (value*std+mean)[0]
                        write_tensor(cache,released.cpu())
                        atomic_json(meta,dict(signature=signature,sha256=sha(cache)))
                    path = "audited new single path"
                reused[key] = u,released
                evidence["releases"].append(dict(file=file,start=s,kind=path,x=tensor_sha(u),p=tensor_sha(released)))
                if s==start:
                    continue
                x.append(u)
                p.append(released)
                y.append(field[:,s-start+20:s-start+40])
                a,b = reused[file,s-20]
                prior_x.append(a)
                prior_p.append(b)
                cases.append(dict(file=file,re=item["re"],start=s,earlier=[s-20,s],
                                  issue_frame=s+20,score_frame=s+40,expanded=s in item["new_vs_main"],
                                  beyond_prior_window=s in item["new_vs_prior_window_study"]))
    finally:
        hook.remove()
    for i in range(1,len(cases)):
        if cases[i]["file"]==cases[i-1]["file"]:
            assert cases[i]["start"]==cases[i-1]["start"]+20
            assert torch.equal(x[i],y[i-1]) and torch.equal(prior_p[i],p[i-1])
    return SimpleNamespace(x=torch.stack(x),p=torch.stack(p),y=torch.stack(y),previous_x=torch.stack(prior_x),
                           previous_p=torch.stack(prior_p),cases=cases),evidence


def feedback_inputs(current,true,predicted,branch,scale):
    return encode(current,(true,predicted,true-predicted.double()),ARMS[branch],scale)


@torch.inference_mode()
def rollout(data,norm,net,fixed,scale):
    """Issue one correction from observed input, then score against its newly arrived truth."""
    states,estimates = [],[]
    previous = None
    last_out = None
    for i,c in enumerate(data.cases):
        linked = previous is not None and c["file"]==previous["file"] and c["start"]==previous["start"]+20
        if linked:
            assert previous["score_frame"]==c["issue_frame"]
        observed,point = data.x[i:i+1],data.p[i:i+1]
        true = observed.double()-data.previous_p[i:i+1].double()
        prediction = last_out if linked else fixed[i:i+1]
        design = feedback_inputs(base(observed,point,norm),true,prediction,"E",scale)
        current = predict(net,design,scale)
        assert torch.isfinite(current).all()
        states.append(dict(file=c["file"],start=c["start"],issue_frame=c["issue_frame"],score_frame=c["score_frame"],
                           prior_source="published" if linked else "fixed_reset",state_sha256=tensor_sha(prediction),
                           observed_residual_sha256=tensor_sha(true),issued_sha256=tensor_sha(current)))
        estimates.append(current)
        last_out,previous = current,c
    return torch.cat(estimates),states
