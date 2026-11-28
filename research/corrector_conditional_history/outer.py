"""Train prespecified Corrector holdouts without using held truth for selection."""
from collections import defaultdict

import torch

from research.corrector.run import sha
from research.corrector_incremental_factorial.core import prefix, stamp
from research.corrector_incremental_structure.fit import Fits
from research.corrector_incremental_structure.teachers import bundle, outer_bundles
from research.corrector_remaining_residual_transfer.core import verify_archive
from research.corrector_spacetime_validation.data import OUT as REFERENCE_STUDY
from research.corrector_residual_feedback.model import base, predict
from research.corrector_residual_feedback.run import references, rms, save_plan
from research.corrector_residual_optimization.fit import shared_plan, track_weights
from research.corrector_simple_baseline.fit import table
from research.corrector_spacetime_validation.fit import Fits as ReferenceFits, atomic_json, load_net
from research.corrector_spacetime_validation.outer import subset
from research.corrector_window_study.data import read_json, tensor_sha

from .core import shared_archive, training_queue
from .donors import prepare
from .diagnostics import gradient_ratios
from .run import PREVIOUS


def fit_early(engine, record, base, region):
    start=record["config"]["first"]
    assert start==250
    source=prefix(base,engine.out,through=start)
    return stamp(engine.fit(f"outer_{region}_ID_TM1_early",record["config"]["model"],shared=source,
                            shared_step=start),engine.out)


def new_split(data, re):
    held=[i for i,c in enumerate(data.cases) if c["re"]==re]
    active=[i for i,c in enumerate(data.cases) if c["re"]!=re]
    by_re=defaultdict(set)
    for i in active:
        c=data.cases[i]
        by_re[c["re"]].add(c["file"])
    folds=[[] for _ in range(4)]
    for value in sorted(by_re):
        order=sorted(range(4),key=lambda j:(len(folds[j]),j))
        for i,file in enumerate(sorted(by_re[value])):
            folds[order[i%4]].append(file)
    train_files={file for group in folds for file in group}
    held_files={data.cases[i]["file"] for i in held}
    assert len(held_files)>=4 and train_files.isdisjoint(held_files)
    return active,held,folds,train_files,held_files


def observed_normalization(data):
    """Fit physical input moments on this teacher or Corrector's own trajectories."""
    weights=track_weights(data).to(dtype=torch.float64).view(-1,1)
    observed=data.x.double()
    average=(observed.mean((2,3,4))*weights).sum(0).view(1,2,1,1,1)
    variance=(((observed-average).square().mean((2,3,4)))*weights).sum(0)
    std=variance.sqrt().view_as(average)
    assert torch.isfinite(average).all() and torch.isfinite(std).all() and (std>0).all()
    return dict(mean=average.float(),std=std.float())


def normalization_record(norm, files):
    return dict(source="trajectory_equal_observed_windows", training_files=sorted(files),
                **{name:values.flatten().tolist() for name,values in norm.items()})


@torch.inference_mode()
def reference_output(out,record,data,norm):
    row=record["final"]
    net=load_net(out/row["checkpoint"],row["sha256"],"S3",data.p.device,"NCDHW").eval().requires_grad_(False)
    scale=record["config"]["scale"]
    return (predict(net,base(data.x,data.p,norm),scale),
            predict(net,base(data.previous_x,data.previous_p,norm),scale))


def new_teacher_bundles(study, data, re, label):
    active,held,folds,files,held_files=new_split(data,re)
    train,test=subset(data,active),subset(data,held)
    norm=observed_normalization(train)
    z,zt=base(train.x,train.p,norm),base(test.x,test.p,norm)
    cur,history=torch.empty_like(train.p),torch.empty_like(train.p)
    _,_,initial=references()
    teachers,metadata={},[]
    for fold,excluded in enumerate([*folds,sorted(held_files)]):
        target=subset(train,[i for i,c in enumerate(train.cases) if c["file"] in excluded]) if fold<4 else test
        ref_ids=[i for i,c in enumerate(train.cases) if c["file"] not in excluded]
        source=subset(train,ref_ids)
        trained={c["file"] for c in source.cases}
        assert trained==files-set(excluded) and trained.isdisjoint(held_files)
        ref_norm=observed_normalization(source)
        ref_norm_record=normalization_record(ref_norm,trained)
        schedule=shared_plan(source)
        name=f"outer_{label}_reference_{fold}"
        sample=save_plan(study.out,name,schedule)
        engine=ReferenceFits(study.out,study.cache,source,base(source.x,source.p,ref_norm),
                             rms(source),schedule,initial,reference_layout=True)
        result=engine.fit(name,"S3",{},origin=dict(outer=label,reference=fold,files=sorted(trained),
                                                 normalization=ref_norm_record),
                          archive_states=True)
        teachers[name]=result
        current,previous=reference_output(study.out,result,target,ref_norm)
        if fold<4:
            indices=[i for i,c in enumerate(train.cases) if c["file"] in excluded]
            cur[indices],history[indices]=current,previous
        metadata.append(dict(fold=fold,run=name,train_files=sorted(trained),held_files=sorted(excluded),
                             normalization=ref_norm_record,
                             target_windows=[[c["file"],c["start"]] for c in target.cases],
                             final_sha256=result["final"]["sha256"],schedule=sample,
                             current_sha256=tensor_sha(current),history_sha256=tensor_sha(previous)))
    schedule=shared_plan(train)
    scale=rms(train)
    backbone_files=set(read_json(REFERENCE_STUDY/"reference_lineage.json")["backbone"]["training_files"])
    identity=dict(regime=label,phys_re=re,refs=metadata,train_removed=sorted(held_files),scale=scale,
                  normalization=normalization_record(norm,files),
                  train_current=tensor_sha(cur),train_history=tensor_sha(history),held_current=tensor_sha(current),
                  held_history=tensor_sha(previous),backbone_seen=sorted(held_files & backbone_files),
                  schedule=save_plan(study.out,f"outer_{label}",schedule))
    atomic_json(study.out/f"teacher_outer_{label}.json",identity)
    return bundle(train,z,cur,history,scale,identity),bundle(test,zt,current,previous,scale,identity),schedule,teachers


def fit_outer(study, data, main, plan, audit, basis):
    prior=read_json(PREVIOUS/"outer_lineage.json")["regimes"]
    registrations=read_json(study.out/"donor_plan.json")
    models=main["models"]
    pair_selected=list(main["selection"]["pair_outer"].values())
    all_records, views, donors, coverage, teachers, gradients = {}, {}, [], {}, {}, []
    for label, re in plan["outer"].items():
        print("OUTER START",label,re,flush=True)
        active, held, schedule=outer_bundles(study,data,label)
        archived=prior[label]
        assert active.identity=={k:v for k,v in archived.items() if k!="candidates"}
        baselines=archived["candidates"]
        engine=Fits(study.out,study.cache,active,None,schedule,basis)
        for name,record in baselines.items():
            verify_archive(record,engine,audit,models[name]["config"]["model"],
                           shared_archive(record,baselines[name.split("_")[0]+"_MSE"],label))
        slots, rows, summary=prepare(study.out,label,active,schedule,registrations["seed"],
                                    registrations["time_slack_frames"])
        donors.extend(rows)
        coverage[label]=summary
        wanted=[n for n in plan["new"] if "_shuffled_" in n or n in pair_selected]
        fresh=training_queue(engine,models,wanted,slots,scope=f"outer_{label}_")
        # Prefix-only MSE phases are recorded separately from scored configurations.
        counts={n:r for n,r in fresh.items() if n in pair_selected or "_shuffled_" in n}
        counts["ID_TM1_early"]=fit_early(engine,models["ID_TM1_early"],baselines["ID_MSE"],label)
        all_records[label]=dict(teacher=held.identity,models={**baselines,**counts},
                                training_only={n:r for n,r in fresh.items() if n not in counts})
        views[label]=held
        gradients.extend(gradient_ratios(engine,counts,slots))
        print("OUTER FIT COMPLETE",label,len(all_records[label]["models"]),flush=True)
    extra=plan["additional_outer"]
    eligible_id=main["selection"]["pair_outer"]["ID"]
    total_existing=sum(sum(r["directory"].endswith(study.out.name) for r in (*v["models"].values(),
                       *v["training_only"].values())) for v in all_records.values())
    allow_extra=len(plan["new"])+total_existing+14+int(eligible_id.endswith("_EN"))<=plan["budget"]
    status=dict(re=extra["re"],executed=allow_extra,selected_pair=eligible_id)
    if allow_extra:
        label="additional"
        active, held, schedule, teachers=new_teacher_bundles(study,data,extra["re"],label)
        engine=Fits(study.out,study.cache,active,None,schedule,basis)
        slots, rows, summary=prepare(study.out,label,active,schedule,registrations["seed"],
                                    registrations["time_slack_frames"])
        donors.extend(rows)
        coverage[label]=summary
        ordinary={}
        for family in plan["focal"]:
            for loss in ("MSE","EN"):
                name=f"{family}_{loss}"
                shared=prefix(ordinary[f"{family}_MSE"],study.out) if loss=="EN" else None
                ordinary[name]=stamp(engine.fit(f"outer_{label}_{name}",models[name]["config"]["model"],
                                                shared=shared),study.out)
        fresh=training_queue(engine,models,["IB_shuffled_MSE","ID_shuffled_MSE",eligible_id],slots,
                             scope=f"outer_{label}_")
        selected={name:r for name,r in fresh.items() if "_shuffled_" in name or name==eligible_id}
        all_records[label]=dict(teacher=held.identity,models={**ordinary,**selected},
                                training_only={name:r for name,r in fresh.items() if name not in selected})
        views[label]=held
        gradients.extend(gradient_ratios(engine,all_records[label]["models"],slots))
    else:
        status["reason"]="保守计入所有独立归档的 MSE 前缀后，剩余预算不足。"
    table(study.out/"history_donors_outer.csv",donors)
    atomic_json(study.out/"donor_outer.json",coverage)
    return all_records,views,teachers,status,gradients
