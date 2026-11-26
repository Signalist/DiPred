"""Evaluate frozen configurations on matched historical and holdout cohorts."""
import csv
import os
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.data import ROOT,guard
from research.corrector_residual_feedback.model import base,predict as predict_teacher
from research.corrector_simple_baseline.fit import table
from research.corrector_spacetime_validation.expansion import period_data
from research.corrector_spacetime_validation.fit import atomic_json,load_net
from research.corrector_spacetime_validation.outer import subset
from research.corrector_window_study.data import read_json,tensor_sha

from .fit import Fits,open_head,predict
from .model import inputs,residual
from .run import OPTIMIZATION,VALIDATION
from .score import IncrementalScores
from .teachers import bundle,full_teacher,main_bundle,outer_bundles


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def own_controls(freeze):
    models=freeze["models"]
    controls={"DB":"D0","DE":"DB"}
    for name,item in models.items():
        if item["kind"]=="structured":
            cfg=item["config"]["model"]
            root=cfg["family"]+("0","B","E")[cfg["history"]]
            if name!=root:
                controls[name]=root
            elif cfg["family"]=="I":
                controls[name]="D"+root[1:]
        else:
            root="old_"+item["config"]["branch"]+"_MSE"
            if name!=root:
                controls[name]=root
    return controls


def comparison_rules(freeze,controls,q_parent):
    selection=freeze["selection"]

    def comparisons(split,arms):
        pairs=[(label,a,b) for label,entries in selection["primary"].items() for a,b in entries]
        if split.startswith("holdout_"):
            pairs.extend(("P1_outer",name,"S3") for name in selection["outer"])
        for name in arms:
            pairs.append(("S0_Native",name,"Native"))
            pairs.append(("S0_S3",name,"S3"))
            if name in controls:
                pairs.append(("S1_own",name,controls[name]))
        pairs.extend(("S2_rank",name,name.split("_")[0]) for name in selection["representation"])
        pairs.extend(("S3_mix","IB_"+name,"IB") for name in ("last","reverse","lag","dual","free"))
        pairs.extend(("S3_dual","IB_dual","IB_"+name) for name in ("lag","last","reverse","free"))
        for tag,parent in (("MSE","IE"),("aux",q_parent)):
            a,b=f"Q_{tag}_self",f"Q_{tag}_ref"
            pairs.extend((("S4_ref_run",a,b),("S4_sequence",b,parent),
                          ("S4_self_run",a+"_self_run",b+"_self_run"),
                          ("S4_switch",a+"_self_run",a),("S4_switch",b+"_self_run",b)))
        return sorted({(label,a,b) for label,a,b in pairs if a!=b and a in arms and b in arms})
    return comparisons


def load_candidate(item,row,out,basis,device):
    if item["kind"]=="structured":
        return open_head(item,row,out,basis,device)
    cfg=item["config"]
    return load_net(ROOT/row["path"],row["sha256"],cfg["branch"],device,cfg.get("layout"))


def estimate(item,net,view,z=None):
    if item["kind"]=="structured":
        return predict(net,view,item["config"]["model"])
    cfg=item["config"]
    return predict_teacher(net,z[cfg["branch"]],cfg["scale"])


def target_invariance(view,build,model,net,before):
    data=view.data
    old=data.y
    old_target=view.target
    data.y=old.roll(1,0)+.03125
    view.target=data.y.double()-data.p.double()
    try:
        assert torch.equal(estimate(model,net,view,build()),before)
    finally:
        data.y=old
        view.target=old_target


def recorded_training(out,freeze,outer_records):
    rows=[]
    for arm,item in {**freeze["models"],**outer_records}.items():
        rows.extend(dict(row,arm=arm,kind=item["kind"]) for row in item["training"])
    keys=list(dict.fromkeys(k for row in rows for k in row))
    table(out/"training.csv",[{k:r.get(k) for k in keys} for r in rows])


def verify_historical(out,book):
    previous=read_json(VALIDATION/"artifact_audit.json")
    assert sha(VALIDATION/"scores.csv")==previous["files"]["scores.csv"]
    replay=[]
    sources={root:{(r["split"],r["policy"],r["arm"]):float(r["mse"])
                  for r in read_rows(root/"scores.csv")} for root in (VALIDATION,OPTIMIZATION)}
    for row in book.recomputed:
        split,arm,policy=(row[k] for k in ("split","arm","policy"))
        if split not in ("development","historical_test"):
            continue
        name=arm[4:] if arm.startswith("old_") else arm
        if name in ("Native","S3") or name.startswith(("E_EN_","E_DIAG","E_PERM")):
            source=VALIDATION
        elif arm.startswith("old_"):
            source=OPTIMIZATION
            split="test" if split=="historical_test" else split
        else:
            continue
        error=abs(row["mse"]-sources[source][split,policy,name])
        assert error<1e-12,(row,source,error)
        replay.append(dict(row,source=str((source/"scores.csv").relative_to(ROOT)),absolute_error=error))
    atomic_json(out/"historical_reproduction.json",replay)


def evaluate_outer(study,train,freeze,basis,book):
    assert read_json(study.out/"outer_plan.json")==read_json(VALIDATION/"outer_split_manifest.json")
    lineage={}
    records={}
    for label in ("interpolation","upper","lower"):
        active,held,plan=outer_bundles(study,train,label)
        engine=Fits(study.out,study.cache,active,None,plan,basis)
        split="holdout_"+label
        book.add_view(held,split,"Native","final",torch.zeros_like(held.current),study.norm["std"])
        book.add_view(held,split,"S3","final",held.current,study.norm["std"])
        candidates={}
        for arm in freeze["selection"]["outer"]:
            run=f"outer_{label}_{arm}"
            cfg=freeze["models"][arm]["config"]["model"]
            shared=study.out/f"shared_outer_{label}_{('I0','IB','IE')[cfg['history']]}.pt"
            prefix=(study.out/f"shared_outer_{label}_{arm}.pt") if arm in ("IB","IE") else None
            item=engine.fit(run,cfg,shared=shared if cfg["weights"] else None,save_shared=prefix)
            row=item["final"]
            net=open_head(item,row,study.out,basis,held.current.device)
            value=predict(net,held,cfg)
            item=dict(item,kind="structured")
            target_invariance(held,lambda:None,item,net,value)
            book.add_view(held,split,arm,"final",value,study.norm["std"])
            candidates[arm]=dict(config=item["config"],step=row["step"],checkpoint=row["checkpoint"],
                                 sha256=row["sha256"],target_invariance=True)
            records[run]=item
        lineage[label]=dict(held.identity,candidates=candidates)
        atomic_json(study.out/"outer_lineage.json",dict(freeze_sha256=sha(study.out/"FROZEN.json"),regimes=lineage))
        print("OUTER COMPLETE",label,len(active.data.cases),len(held.data.cases),flush=True)
    return records


def timed(fn,device,repeat=7):
    def sync():
        if device.type=="cuda":
            torch.cuda.synchronize(device)
    fn()
    sync()
    samples=[]
    for _ in range(repeat):
        then=perf_counter()
        fn()
        sync()
        samples.append(1000*(perf_counter()-then))
    samples.sort()
    return samples[len(samples)//2]


def inference_cost(study,reference,view,name,item,net):
    source=study.triplets.source
    x=view.data.x[:1]
    prev_result=view.historical[:1]
    prev_native=view.data.previous_p[:1]
    device=x.device
    scale=study.cfg["external_rms"]
    mean,std=study.norm["mean"],study.norm["std"]
    native_count=sum(p.numel() for p in source.backbone.net.parameters())
    s3_count=sum(p.numel() for p in reference.parameters())
    head_count=sum(p.numel() for p in net.parameters()) if net is not None else 0
    structured=item is not None and item["kind"]=="structured"
    cfg=None if item is None else item["config"]
    require_current=name=="S3" or structured
    cached_history=bool(cfg and (cfg["model"]["history"]==2 if structured else cfg["branch"]=="E"))

    def pipeline():
        p=source.backbone.rollout((x-mean)/std,seed=500000,sign=0)*std+mean
        if name=="Native":
            return p
        f=base(x,p,study.norm)
        s3=predict_teacher(reference,f,scale) if require_current else None
        if name=="S3":
            return p.double()+s3.double()
        temp=SimpleNamespace(base=f,current=s3,scale=view.scale,true=x.double()-prev_native.double(),
                             historical=prev_result)
        if structured:
            index=slice(None)
            return p.double()+residual(temp,index,cfg["model"],net(inputs(temp,index,cfg["model"])))
        from research.corrector_spacetime_validation.expansion import feedback_inputs
        features=feedback_inputs(f,temp.true,prev_result,cfg["branch"],cfg["scale"])
        return p.double()+predict_teacher(net,features,cfg["scale"]).double()
    s3_used=require_current or cached_history
    s3_ms=timed(lambda:predict_teacher(reference,view.base[:1],scale),device) if s3_used else 0.
    return dict(arm=name,trainable_parameters=head_count,backbone_parameters=native_count,
                frozen_s3_parameters=s3_count if s3_used else 0,
                total_parameters=native_count+(s3_count if s3_used else 0)+head_count,
                current_s3_calls=int(require_current),historical_s3_cached=cached_history,
                ms_s3_only=s3_ms,windows_per_call=1,
                ms_end_to_end=timed(pipeline,device))


def feedback_rollout(view,item,net,mode):
    """Issue each correction and propagate only already-issued residuals."""
    cfg=item["config"]["model"]
    assert mode in ("ref","self") and cfg["history"]==2
    last_case=None
    last_prediction=None
    outputs=[]
    states=[]
    net.eval()
    for i,case in enumerate(view.data.cases):
        linked=last_case is not None and case["file"]==last_case["file"]
        if linked:
            assert case["start"]==last_case["start"]+20 and case["issue_frame"]==last_case["score_frame"]
        history=last_prediction if linked and mode=="self" else view.historical[i:i+1]
        j=slice(i,i+1)
        output=residual(view,j,cfg,net(inputs(view,j,cfg,feedback=history)))
        assert torch.isfinite(output).all()
        state_error=(history.double()-view.historical[j].double()).square().mean().sqrt()
        states.append(dict(case,prior_source="published" if linked and mode=="self" else "fixed_s3",
                           previous_sha256=tensor_sha(history),published_sha256=tensor_sha(output),
                           state_rms_from_fixed_s3=float(state_error),
                           observation_sha256=tensor_sha(view.data.x[j]),native_sha256=tensor_sha(view.data.p[j])))
        outputs.append(output)
        last_case,last_prediction=case,output
    return torch.cat(outputs),states


def reuse_period_cache(study):
    source=study.cache.parent/VALIDATION.name/"period_native"
    destination=study.cache/"period_native"
    destination.mkdir(exist_ok=True)
    for path in source.iterdir() if source.is_dir() else ():
        target=destination/path.name
        if target.exists():
            assert sha(target)==sha(path)
        elif path.suffix in (".pt",".json"):
            os.link(path,target)


def altered_arrival(study,reference,view):
    data=view.data
    first,second=data.cases[:2]
    assert first["file"]==second["file"] and first["score_frame"]==second["issue_frame"]
    assert torch.equal(data.y[:1],data.x[1:2])
    observed=data.y[:1]+.03125
    mean,std=study.norm["mean"],study.norm["std"]
    backbone=study.triplets.source.backbone
    prediction=backbone.rollout((observed-mean)/std,seed=500000,sign=0)*std+mean
    features=base(observed,prediction,study.norm)
    current=predict_teacher(reference,features,study.cfg["external_rms"])
    changed=SimpleNamespace(base=features,current=current,true=observed.double()-data.p[:1].double(),
                            scale=view.scale,historical=view.historical[1:2])
    return changed,prediction


def confirmation_inventory(study,period_plan):
    inventory=ROOT/"results/corrector_distribution_study/CONFIRMATION_INVENTORY.json"
    digest=sha(inventory)
    assert digest==period_plan["independent"]["provenance_sha256"]
    recorded=read_json(inventory)
    known={r["file"] for r in recorded["full_local_real_inventory"]}
    present={p.name for p in study.triplets.source.raw.glob("*.h5")}
    assert not recorded["confirmation"] and not period_plan["independent"]["verified"]
    assert present==known,dict(new=sorted(present-known),missing=sorted(known-present))
    return dict(inventory_sha256=digest,files=len(known),verified=[])


def feedback_interactions(out,bootstrap):
    tracks=read_rows(out/"trajectories.csv")
    estimates={(r["split"],r["arm"],r["trajectory"]):float(r["mse"]) for r in tracks if r["policy"]=="final"}
    rows=[]
    for split in ("period","period_new"):
        files=sorted(file for cohort,arm,file in estimates if cohort==split and arm=="Native")
        rng=np.random.default_rng(bootstrap["seed"])
        draws=rng.integers(len(files),size=(bootstrap["draws"],len(files)))
        for objective in ("MSE","aux"):
            ref=f"Q_{objective}_ref"
            own=f"Q_{objective}_self"
            values=np.array([[estimates[split,a,f] for a in (ref,ref+"_self_run",own,own+"_self_run")] for f in files])
            fixed_shift=values[:,1]-values[:,0]
            self_shift=values[:,3]-values[:,2]
            difference=fixed_shift-self_shift
            low,high=np.percentile(difference[draws].mean(axis=1),(2.5,97.5))
            rows.append(dict(split=split,objective=objective,trajectories=len(files),
                             ref_training_shift=float(fixed_shift.mean()),self_training_shift=float(self_shift.mean()),
                             interaction=float(difference.mean()),lower=float(low),upper=float(high),
                             reduced_shift_tracks=int((difference>0).sum())))
    table(out/"feedback_interactions.csv",rows)


def period_scores(study,test,reference,basis,freeze,book):
    reuse_period_cache(study)
    data,evidence=period_data(study,test)
    plan=read_json(study.out/"period_plan.json")
    assert len(data.cases)==96 and all(len(c["starts"])==8 for c in plan["chains"])
    previous=base(data.previous_x,data.previous_p,study.norm)
    features=base(data.x,data.p,study.norm)
    scale=study.cfg["external_rms"]
    current=predict_teacher(reference,features,scale)
    history=predict_teacher(reference,previous,scale)
    view=bundle(data,features,current,history,study.scale,{"period":True})
    data.historical_prediction=history
    _,design=study.inputs(data)
    ids=[i for i,c in enumerate(data.cases) if c["beyond_prior_window"]]
    extended=bundle(subset(data,ids),features[ids],current[ids],history[ids],study.scale,{"period_new":True})
    def record(arm,result):
        book.add_view(view,"period",arm,"final",result,study.norm["std"])
        book.add_view(extended,"period_new",arm,"final",result[ids],study.norm["std"])
    record("Native",torch.zeros_like(current))
    record("S3",current)
    feedback=[]
    arrival,arrival_native=altered_arrival(study,reference,view)
    interventions=[]
    policies=freeze["selection"]
    for name,item in freeze["models"].items():
        net=load_candidate(item,item["final"],study.out,basis,current.device)
        if name in policies["feedback"]:
            state={}
            for mode in ("ref","self"):
                value,rows=feedback_rollout(view,item,net,mode)
                original=data.y
                original_target=view.target
                data.y=original.roll(1,0)+.03125
                view.target=data.y.double()-data.p.double()
                try:
                    perturbed,_=feedback_rollout(view,item,net,mode)
                    assert torch.equal(perturbed,value)
                finally:
                    data.y=original
                    view.target=original_target
                state[mode]=value
                record(name if mode=="ref" else name+"_self_run",value)
                cfg=item["config"]["model"]
                ix=slice(None)
                previous=value[:1] if mode=="self" else arrival.historical
                after=arrival_native.double()+residual(arrival,ix,cfg,net(inputs(arrival,ix,cfg,feedback=previous)))
                old=data.p[1:2].double()+value[1:2]
                interventions.append(dict(arm=name,run_mode=mode,current_unchanged=True,
                                          future_issue=data.cases[1]["issue_frame"],
                                          changed=(not torch.equal(old,after)),
                                          next_forecast_rms=float((after-old).square().mean().sqrt())))
                mse=(data.p.double()+value-original.double()).square().mean((1,2,3,4)).tolist()
                for row,score in zip(rows,mse):
                    feedback.append(dict(row,arm=name,run_mode=mode,mse=score,future_invariance=True))
            starts=[i for i,c in enumerate(data.cases) if c["start"]==400]
            assert torch.equal(state["self"][starts],state["ref"][starts])
        else:
            record(name,estimate(item,net,view,design))
    atomic_json(study.out/"period_provenance.json",evidence)
    table(study.out/"feedback_rollout.csv",feedback)
    table(study.out/"feedback_arrival_intervention.csv",interventions)
    return dict(windows=len(data.cases),later_windows=len(ids),feedback_rows=len(feedback),
                native_sha256=tensor_sha(data.p),independent=confirmation_inventory(study,plan))


@torch.inference_mode()
def score_frozen(study,reference,basis,freeze,book):
    out=study.out
    costs=[]
    future=[]
    feature_invariance={}
    fixed={}
    for tag,split in (("development","development"),("test","historical_test")):
        data,features,design=study.get(tag)
        view=main_bundle(study,data,features,tag)
        case=read_json(out/"experiment_plan.json")["fixed_case"]
        dev=tag=="development"
        if dev:
            i,=[i for i,c in enumerate(data.cases) if (c["file"],c["start"])==(case["file"],case["start"])]
            t=case["lead"]-1
            fixed["truth_residual"]=view.target[i,:,t].cpu()
        else:
            original=data.y
            data.y=original.flip(0)+.03125
            try:
                recreated,recreated_design=study.inputs(data)
                recreated_s3=predict_teacher(reference,recreated,study.cfg["external_rms"])
                feature_invariance=dict(base=torch.equal(recreated,features),
                                        s3=torch.equal(recreated_s3,view.current),
                                        **{arm:torch.equal(z,recreated_design[arm]) for arm,z in design.items()})
                assert all(feature_invariance.values())
            finally:
                data.y=original
        for name in ("Native","S3",*freeze["models"]):
            item=freeze["models"].get(name)
            for policy in ("selected","final"):
                if item is None:
                    value=view.current if name=="S3" else torch.zeros_like(view.current)
                    net=None
                else:
                    net=load_candidate(item,item[policy],out,basis,data.p.device)
                    value=estimate(item,net,view,design)
                    if not dev:
                        target_invariance(view,lambda:study.inputs(data)[1],item,net,value)
                        future.append(dict(arm=name,policy=policy,bitwise=True))
                book.add_view(view,split,name,policy,value,study.norm["std"])
                if dev and policy=="selected":
                    costs.append(inference_cost(study,reference,view,name,item,net))
                    fixed[name]=value[i,:,t].double().cpu()
        print("SCORED",split,flush=True)
    table(out/"inference_cost.csv",costs)
    fields=[]
    for name,correction in fixed.items():
        actual=fixed["truth_residual"]
        for c,channel in enumerate(("u","v")):
            for y in range(actual.shape[1]):
                for x in range(actual.shape[2]):
                    fields.append(dict(arm=name,channel=channel,y=y,x=x,value=float(correction[c,y,x]),
                                       remaining=float(actual[c,y,x]-correction[c,y,x])))
    table(out/"fixed_fields.csv",fields)
    return dict(period=period_scores(study,data,reference,basis,freeze,book),future_invariance=future,
                feature_invariance=feature_invariance)


def validate(study,figures):
    out=study.out
    freeze=guard(out)
    policy=read_json(out/"validation_config.json")
    assert policy["outer_step"]==policy["period_step"]==6000 and policy["alpha"]==1.
    assert policy["sequence_models"]==freeze["selection"]["feedback"]
    assert policy["period_arms"]==sorted(freeze["models"])
    assert all(sha(ROOT/name)==digest for name,digest in freeze["checkpoints"].items())
    basis=torch.load(out/"temporal_basis.pt",map_location="cpu",weights_only=True)
    train,features,design=study.get("train")
    del features,design
    threshold_field=(train.y.double()-train.p.double()).abs()[:,:,::2,::2,::2]
    thresholds=[float(torch.quantile(threshold_field[:,c].contiguous().flatten(),.95)) for c in (0,1)]
    atomic_json(out/"region_thresholds.json",dict(source="training residual absolute 95th percentile",
                                                 stride=[2,2,2],physical_thresholds=thresholds))
    book=IncrementalScores(basis,thresholds)
    records=evaluate_outer(study,train,freeze,basis,book)
    reference,_=full_teacher(study,train.p.device)
    del train,threshold_field
    evidence=score_frozen(study,reference,basis,freeze,book)
    verify_historical(out,book)
    controls=own_controls(freeze)
    q_parent=read_json(out/"feedback_training_selection.json")["auxiliary"]
    for arm in freeze["selection"]["feedback"]:
        controls[arm]=q_parent if "_aux_" in arm else "IE"
        controls[arm+"_self_run"]=arm
    recorded_training(out,freeze,records)
    metrics=book.write(out,controls,comparison_rules(freeze,controls,q_parent))
    feedback_interactions(out,policy["bootstrap"])
    atomic_json(out/"verification.json",dict(metrics,**evidence,freeze_sha256=sha(out/"FROZEN.json")))
    from .plot import plot
    from .report import report
    from .audit import audit
    plot(out,figures)
    report(out,figures)
    audit(study,figures)
