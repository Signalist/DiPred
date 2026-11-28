"""Matched history perturbations with the present prediction held fixed."""
from collections import defaultdict

import torch

from research.corrector_incremental_factorial.core import load_head
from research.corrector_incremental_structure.model import inputs,residual
from research.corrector_residual_optimization.fit import track_weights


def donors(cases):
    groups=defaultdict(list)
    for i,case in enumerate(cases):
        groups[case["re"]].append(i)
    indices=list(range(len(cases)))
    for members in groups.values():
        tracks=defaultdict(list)
        for i in sorted(members,key=lambda j:(cases[j]["file"],cases[j].get("start",0),j)):
            tracks[cases[i]["file"]].append(i)
        ordered=[i for track in tracks.values() for i in track]
        dominant=max(tracks,key=lambda key:len(tracks[key]))
        biggest=tracks[dominant]
        if 2*len(biggest)<=len(ordered):
            shifted=ordered[len(biggest):]+ordered[:len(biggest)]
            for i,j in zip(ordered,shifted):
                indices[i]=j
        else:
            others=[i for key,track in tracks.items() if key!=dominant for i in track]
            for i,j in zip(biggest,others):
                indices[i],indices[j]=j,i
    assert sorted(indices)==list(range(len(cases)))
    assert all(i==j or (cases[i]["re"]==cases[j]["re"] and cases[i]["file"]!=cases[j]["file"])
               for i,j in enumerate(indices))
    return indices


@torch.inference_mode()
def examine(view, models, plan, basis, split):
    actual=view.true
    predicted=view.historical.double()
    remaining=actual-predicted
    fields=dict(R=actual,P=predicted,E=remaining)
    cases=view.data.cases
    source=donors(cases)
    assignments=[dict(split=split,re=case["re"],trajectory=case["file"],start=case["start"],
                      donor_trajectory=cases[j]["file"],donor_start=cases[j]["start"],moved=i!=j)
                 for i,(case,j) in enumerate(zip(cases,source))]
    moved=sum(row["moved"] for row in assignments)
    location=torch.randperm(actual.shape[-2]*actual.shape[-1],generator=torch.Generator().manual_seed(
        plan["interventions"]["spatial_seed"])).to(actual.device)
    transforms={
        "original":lambda v:v, "zero":torch.zeros_like, "same_re":lambda v:v[source],
        "reverse":lambda v:v.flip(2),
        "spatial":lambda v:v.flatten(-2)[:,:,:,location].reshape_as(v),
    }
    for count in plan["interventions"]["last"]:
        def mask(value, count=count, keep=False):
            result=value.clone()
            if keep:
                result[:,:,:-count]=0
            else:
                result[:,:,-count:]=0
            return result
        transforms[f"keep_last_{count}"]=lambda v,c=count:mask(v,c,True)
        transforms[f"mask_last_{count}"]=lambda v,c=count:mask(v,c,False)
    for lead in range(20):
        def remove(value, lead=lead):
            result=value.clone()
            result[:,:,lead]=0
            return result
        transforms[f"zero_history_lead_{lead+1}"]=remove
    sample_weight=track_weights(view.data)
    ids=torch.arange(len(view.data.cases),device=actual.device)
    rows=[]
    for arm in ("IB","IC","ID","IE","IF"):
        active=plan["inputs"][arm]
        for kind in ("MSE","TM1"):
            name=f"{arm}_{kind}"
            record=models[name]
            net=load_head(record,record["final"],basis,actual.device).eval()
            cfg=record["config"]["model"]
            def predict(overrides):
                batches=[]
                for batch in ids.split(8):
                    batch_history={key:value[batch] for key,value in overrides.items()}
                    if "history_fields" in cfg:
                        features=inputs(view,batch,cfg,intervened=batch_history)
                    else:
                        feedback=batch_history.get("P")
                        if feedback is not None:
                            feedback=feedback.to(view.historical.dtype)
                        features=inputs(view,batch,cfg,feedback=feedback,intervened=batch_history.get("R"))
                    batches.append(residual(view,batch,cfg,net(features)))
                return torch.cat(batches)
            original=predict({})
            for intervention, transform in transforms.items():
                scope=[("joint",active)]
                if len(active)>1 and intervention!="original":
                    scope += [(f"single_{k}",[k]) for k in active]
                for label, keys in scope:
                    values={k:transform(fields[k]) for k in keys}
                    if label=="joint":
                        rx,px=transform(actual),transform(predicted)
                        assert torch.equal(rx-px,transform(remaining)), intervention
                    changed=original if intervention=="original" else predict(values)
                    assert torch.isfinite(changed).all()
                    delta=(changed-original).square().mean((1,3,4))
                    before=(original-view.target).square().mean((1,3,4))
                    after=(changed-view.target).square().mean((1,3,4))
                    c=((changed-view.current.double())*(view.target-view.current.double())).mean((1,3,4))
                    u=(changed-view.current.double()).square().mean((1,3,4))
                    metrics=[(x*sample_weight[:,None]).sum(0).tolist() for x in (before,after,delta,c,u)]
                    for lead,(b,a,d,inner,energy) in enumerate(zip(*metrics),1):
                        rows.append(dict(split=split,arm=name,intervention=intervention,scope=label,lead=lead,
                                         original_mse=b,mse=a,delta_mse=a-b,response_rms=d**.5,
                                         C_delta=inner,U_delta=energy,eligible_swaps=moved,
                                         relational_ood=label!="joint"))
    return rows,assignments
