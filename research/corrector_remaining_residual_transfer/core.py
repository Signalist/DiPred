"""Complete the fixed-rho, common-weight and early-start development queues."""
from copy import copy
from shutil import copyfile
from statistics import mean

import torch

from research.corrector.run import sha
from research.corrector_history_free_refinement.core import replay_development
from research.corrector_incremental_factorial.core import calibrate, details, prefix, probes, stamp, verify_reuse
from research.corrector_incremental_structure.fit import Fits
from research.corrector_incremental_structure.model import spec
from research.corrector_incremental_structure.teachers import main_bundle
from research.corrector_residual_feedback.data import guard
from research.corrector_simple_baseline.fit import table
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_spacetime_validation.gradients import monitor_indices
from research.corrector_window_study.data import ROOT,read_json

from .diagnose import stage_zero
from .run import FACTORIAL,REFINEMENT,STRUCTURE

REGISTRATION=("PROTOCOL.md","hypotheses.json","experiment_plan.json","data_roles.json")
DIAGNOSTICS=("residual_transfer_correlations","temporal_correlations","spatial_correlations","transfer_modes","amplitude_transfer")
LEGACY={"IB":dict(TM1="IB_match",TM2="IB_match2",EN="IB_energy"),
        "IE":dict(TM1="IE_match",TM2="IE_match2",EN="IE_energy")}


def prior():
    folders=(STRUCTURE,FACTORIAL,REFINEMENT)
    frozen={folder:guard(folder) for folder in folders}
    hashes={}
    for folder in folders:
        audit=read_json(folder/"artifact_audit.json")
        assert audit["passed"] and audit["freeze_sha256"]==sha(folder/"FROZEN.json")
        for path,digest in audit["checkpoints"].items():
            assert hashes.get(path,digest)==digest,path
            hashes[path]=digest
    basis_file=STRUCTURE/"temporal_basis.pt"
    assert sha(basis_file)==frozen[FACTORIAL]["basis_sha256"]==frozen[REFINEMENT]["basis_sha256"]
    return frozen,dict(checkpoints=hashes),torch.load(basis_file,map_location="cpu",weights_only=True)


def model_config(fields,weights=None,mix="original"):
    cfg=spec("I",len(fields),weights,mix=mix)
    if fields not in ([],["R"],["R","P"]):
        cfg["history_fields"]=fields
    return cfg


def archive_record(folder,name):
    return stamp(read_json(folder/"structured_fits"/f"{name}.json"),folder)


def verify_archive(record, engine, audit, model, shared=None):
    other=copy(engine)
    old_dev=record["config"]["inputs"]["development"]
    now=engine.identity["development"]
    if old_dev is not None:
        skip={"distribution","current"}
        assert {k:v for k,v in old_dev.items() if k not in skip}=={k:v for k,v in now.items() if k not in skip}
    else:
        assert now is None
    other.identity=dict(engine.identity,development=old_dev)
    verify_reuse(record,other,audit,model,shared)


def fit_core(study):
    out=study.out
    for f in REGISTRATION:
        assert (out/f).is_file()
    plan=read_json(out/"experiment_plan.json")
    frozen,audit,basis=prior()
    train,z,_=study.get("train")
    dev,dz,_=study.get("development")
    x=main_bundle(study,train,z,"train")
    y=main_bundle(study,dev,dz,"development")
    for split in ("train","development"):
        file=f"teacher_{split}.json"
        old=read_json(REFINEMENT/file)
        current=read_json(out/file)
        assert sha(REFINEMENT/file)==frozen[REFINEMENT]["bindings"][file]
        omit={"distribution"} if split=="train" else {"distribution","current"}
        assert {k:v for k,v in old.items() if k not in omit}=={k:v for k,v in current.items() if k not in omit}
    atomic_json(out/"reference_lineage.json",dict(train=x.identity,development=y.identity,
                backbone=study.triplets.source.identity,previous={p.name:sha(p/"FROZEN.json") for p in frozen},
                current_and_history_same_fold=True,schedule=frozen[REFINEMENT]["models"]["M0"]["config"]["schedule"]))
    if not all((out/f"{f}.csv").is_file() for f in DIAGNOSTICS):
        stage_zero(out,x,basis)
        print("TRANSFER DIAGNOSTICS COMPLETE",flush=True)
    engine=Fits(out,study.cache,x,y,study.schedule,basis)
    models={}
    replays={}
    origin={}
    configs={k:model_config(v) for k,v in plan["inputs"].items()}
    batches=monitor_indices(train.cases)
    assert batches==read_json(REFINEMENT/"calibration.json")["batches"]

    def use_old(name,record,wanted,shared=None):
        verify_archive(record,engine,audit,wanted,shared)
        origin[name]=dict(directory=record["directory"],run=record["config"]["name"],config_hash=record["config_hash"])
        if record["config"]["inputs"]["development"]!=y.identity:
            replays[name]=replay_development(record,y,basis)
        models[name]=record

    def fit(name,cfg,shared=None,start=1000,replay=None):
        models[name]=stamp(engine.fit(name,cfg,shared=shared,shared_step=start,replayed_prefix=replay),out)

    for arm in ("I0","IB","IE","IC","ID","IF"):
        name=f"{arm}_MSE"
        if arm in ("I0","IB","IE"):
            use_old(name,archive_record(STRUCTURE,arm),configs[arm])
        else:
            fit(name,configs[arm])
    scales=[]
    fields={"R":x.true,"P":x.historical,"E":x.true-x.historical.double()}
    for key,field in fields.items():
        values=(field.double()/x.scale).square().mean((1,2,3,4)).sqrt()
        scales.append(dict(field=key,physical_rms=float(field.double().square().mean().sqrt()),scale=x.scale,
                           normalized_window_rms_quantiles=values.quantile(values.new_tensor([.05,.5,.95])).tolist()))
    atomic_json(out/"input_distribution.json",scales)
    calibrations={}
    for arm in plan["inputs"]:
        plain=f"{arm}_MSE"
        base=models[plain]
        shared=prefix(base,out)
        replay=dict(original_development=base["config"]["inputs"]["development"],metrics=replays[plain]) if plain in replays else None
        terms=tuple(v for v in plan["terms"].values() if v)
        packets=probes(engine,base,1000,terms,batches)
        for family,term in plan["terms"].items():
            if term is None:
                continue
            name=f"{arm}_{family}"
            weight=calibrate(packets,term,plan["rho"])
            candidate, candidate_shared=None,None
            if arm=="I0":
                candidate=frozen[REFINEMENT]["models"][f"I0_{family}_rho03"]
                candidate_shared=shared
                assert read_json(REFINEMENT/"calibration.json")["coefficients"][family]["0.3"]==candidate["config"]["model"]["weights"][term]
            elif arm in LEGACY:
                candidate=archive_record(STRUCTURE,LEGACY[arm][family])
                candidate_shared=STRUCTURE/f"shared_{arm}.pt"
                source_cal=read_json(STRUCTURE/f"{arm}_calibration.json")
                assert source_cal["probe_indices"]==batches
                assert source_cal["coefficients"][term]==candidate["config"]["model"]["weights"][term]
            old_weight=candidate["config"]["model"]["weights"][term] if candidate else None
            old_rho=plan["rho"]*old_weight/weight if old_weight else None
            if old_rho is not None and abs(old_rho-plan["rho"])<plan["rho_tolerance"]:
                cfg=dict(configs[arm],weights={term:old_weight})
                use_old(name,candidate,cfg,candidate_shared)
            else:
                cfg=dict(configs[arm],weights={term:weight})
                fit(name,cfg,shared,replay=replay)
            calibrations[name]=dict(step=1000,term=term,recomputed=weight,archived=old_weight,
                                    actual=models[name]["config"]["model"]["weights"][term],
                                    archived_recomputed_rho=old_rho)
        atomic_json(out/"calibration.json",dict(batches=batches,arms=calibrations))

    anchor_tm1=frozen[REFINEMENT]["models"]["I0_TM1_rho03"]["config"]["model"]["weights"]["match"]
    anchor_tm2=frozen[FACTORIAL]["anchor"]
    use_old("I0_TM2_fixed",frozen[FACTORIAL]["models"]["I0_match2"],
            dict(configs["I0"],weights={"match2":anchor_tm2}),prefix(models["I0_MSE"],out))
    old=frozen[FACTORIAL]["models"]["IB_match2"]
    use_old("IB_TM2_fixed",old,dict(configs["IB"],weights={"match2":anchor_tm2}),STRUCTURE/"shared_IB.pt")
    for arm in plan["focal"]:
        plain=f"{arm}_MSE"
        base=models[plain]
        replay=dict(original_development=base["config"]["inputs"]["development"],metrics=replays[plain]) if plain in replays else None
        if arm!="I0":
            shared=prefix(base,out)
            fit(f"{arm}_TM1_fixed",dict(configs[arm],weights={"match":anchor_tm1}),shared,replay=replay)
        if arm=="ID":
            fit("ID_TM2_fixed",dict(configs[arm],weights={"match2":anchor_tm2}),shared,replay=replay)
        first=plan["early_start"]
        shared=prefix(base,out,through=first)
        types=tuple(plan["terms"][kind] for kind in plan["early"])
        packets=probes(engine,base,first,types,batches)
        for family,term in zip(plan["early"],types):
            weight=calibrate(packets,term,plan["rho"])
            name=f"{arm}_{family}_early"
            calibrations[name]=dict(step=first,term=term,recomputed=weight,actual=weight)
            fit(name,dict(configs[arm],weights={term:weight}),shared,first,replay)

    def mse(name,policy):
        row=models[name][policy]
        return replays.get(name,{}).get(str(row["step"]),row)["dev_mse"]

    extensions=[]
    for mix in plan["maps"]:
        name=f"ID_{mix}_MSE"
        fit(name,model_config(["E"],mix=mix))
        extensions.append(name)
    selected_map=min(extensions,key=lambda name:mse(name,"final"))
    if 1-mse(selected_map,"final")/mse("ID_MSE","final")>=plan["map_gate"]:
        data=models[selected_map]
        shared=prefix(data,out)
        term="match"
        weight=calibrate(probes(engine,data,1000,(term,),batches),term,plan["rho"])
        name=selected_map.replace("_MSE","_TM1")
        fit(name,dict(data["config"]["model"],weights={term:weight}),shared)
        calibrations[name]=dict(step=1000,term=term,recomputed=weight,actual=weight)
        extensions.append(name)

    core=[f"{arm}_{loss}" for arm in plan["inputs"] for loss in plan["terms"]]
    early=[f"{arm}_{loss}_early" for arm in plan["focal"] for loss in plan["early"]]
    candidates=core+early+extensions
    selection={policy:min(candidates,key=lambda n:(mse(n,policy),models[n][policy]["step"],n))
               for policy in ("final","selected")}
    best_early=min(early,key=lambda n:mse(n,"final"))
    selection.update(outer_extra=best_early if mse(best_early,"final")<=min(mse(n,"final") for n in core) else None,
                     core=core,early=early,extensions=extensions,
                     fixed=["I0_TM1","IB_TM1_fixed","ID_TM1_fixed","I0_TM2_fixed","IB_TM2_fixed","ID_TM2_fixed"])
    (out/"exploration_log.md").write_text("# 有限探索\n\n"
        +"末帧广播与恒等初始化 Lag 已按协议训练。额外映射 TM1 配置为 "
        + (extensions[-1] if len(extensions)==3 else "未达到开发改善门槛")
        +"。本轮未启用其他可选扩展。\n",encoding="utf-8")
    atomic_json(out/"reuse_audit.json",dict(reused=origin,development_replay={n:max(abs(
        row["dev_mse"]-r[str(row["step"])]["dev_mse"]) for row in models[n]["training"]) for n,r in replays.items()}))
    atomic_json(out/"calibration.json",dict(batches=batches,arms=calibrations))
    rows=[]
    for name,item in models.items():
        weights=item["config"]["model"]["weights"]
        for step in sorted(set(plan["monitor"]+[calibrations[name]["step"]] if name in calibrations else plan["monitor"])):
            if step<item["config"]["first"]:
                continue
            packets=probes(engine,item,step,tuple(weights),batches)
            for row in details(name,step,packets,weights):
                layers=packets[row["batch"]][row["term"]][2]
                rows.append(dict(row,conv1=layers[0],conv2=layers[1],conv3=layers[2]))
    table(out/"gradient_diagnostics.csv",rows)
    atomic_json(out/"selections.json",selection)
    for file in ("outer_plan.json","period_plan.json"):
        copyfile(REFINEMENT/file,out/file)
    # The expanded 168-window cohort and its prior identity are fixed before validation.
    registered=read_json(REFINEMENT/"experiment_plan.json")["later_same_trajectory"]
    atomic_json(out/"later_plan.json",registered)
    bindings=(*REGISTRATION,"outer_plan.json","period_plan.json","later_plan.json","reference_lineage.json",
              "teacher_train.json","teacher_development.json","native_train.json","native_development.json","triplet_manifest.json",
              "input_distribution.json","reuse_audit.json","calibration.json","selections.json","exploration_log.md",
              "gradient_diagnostics.csv",*(f"{name}.csv" for name in DIAGNOSTICS))
    new=sum(item["directory"].endswith(out.name) for item in models.values())
    assert new+36+3*bool(selection["outer_extra"])<=plan["budget"]
    atomic_json(out/"FROZEN.json",dict(prior_test_manifest_blob=frozen[REFINEMENT]["prior_test_manifest_blob"],
        bindings={name:sha(out/name) for name in bindings},models=models,selection=selection,
        main_new=new,prior_freeze=sha(REFINEMENT/"FROZEN.json"),anchors=dict(TM1=anchor_tm1,TM2=anchor_tm2),
        basis_sha256=frozen[REFINEMENT]["basis_sha256"]))
    print("FROZEN",selection,flush=True)
