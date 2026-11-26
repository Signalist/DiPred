"""Verify frozen inputs, fitted checkpoints, mathematical identities and released scores."""
from math import isfinite
from pathlib import Path
from statistics import mean

from PIL import Image
import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.data import OUT as FEEDBACK,ROOT,guard
from research.corrector_residual_feedback.fit import STEPS,state_digest
from research.corrector_simple_baseline.measure import groups
from research.corrector_spacetime_optimization.fit import config_hash
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_spacetime_validation.objectives import ORDER,operator,terms,verify_math
from research.corrector_window_study.data import read_json,tensor_sha

from .model import Head,spec,verify_basis
from .run import OPTIMIZATION,VALIDATION
from .validation import read_rows


def math_audit(basis):
    rng=torch.Generator().manual_seed(42)
    prediction,target=[torch.randn((2,2,20,3,4),dtype=torch.float64,generator=rng) for _ in range(2)]
    features=(prediction-target).movedim(2,-1).flatten(0,-2)
    d=operator()
    h=d.T@d
    p=torch.eye(20,dtype=torch.float64)[list(ORDER),:]
    loss=terms(prediction,target,("match2","diag","perm"))
    checks={}
    for name,gram in (("match2",h),("diag",torch.diag(h.diag())),("perm",p.T@h@p)):
        expected=(features@gram*features).sum(1).mean()/18
        checks[name]=float((loss[name]-expected).abs())
    assert max(checks.values())<1e-12
    checks.update(verify_basis(basis))
    for name in ("lag","dual","free"):
        net=Head(spec("I",1,mix=name),basis,torch.device("cpu"))
        assert torch.equal(net.mapping(),torch.eye(20))
    return checks


def audit_teachers(out,frozen):
    for split,expected in (("train",822),("development",24),("test",24)):
        item=read_json(out/f"teacher_{split}.json")
        assert len(item["keys"])==expected
        assert all(k in item for k in ("native","observed","current","historical","distribution","scale"))
        if split=="train":
            refs=item["refs"]
            assert len(refs)==4
            assert sorted(v for r in refs for v in r["ids"])==list(range(expected))
            for r in refs:
                assert set(r["train"]).isdisjoint(r["predicted"])
                assert {item["keys"][i][0] for i in r["ids"]}==set(r["predicted"])
                original=read_json(FEEDBACK/f"fit_oof_{r['fold']}.json")
                assert original["final"]["sha256"]==r["sha256"]
                assert original["config"]["schedule_sha256"]==r["schedule_sha256"]
                assert r["training_inputs"]=={k:original["config"][k] for k in ("train_P","train_Y","train_input")}
        else:
            full,=item["refs"]
            assert full["scope"]=="full"
        if split in ("train","development"):
            name=f"teacher_{split}.json"
            assert sha(out/name)==frozen["bindings"][name]
    lineage=read_json(out/"outer_lineage.json")
    assert lineage["freeze_sha256"]==sha(out/"FROZEN.json")
    assert set(lineage["regimes"])=={"interpolation","upper","lower"}
    outer_checkpoints={}
    for label,v in lineage["regimes"].items():
        manifest=read_json(out/"outer_plan.json")[label]
        active,held=(set(manifest[k]) for k in ("active_files","held_files"))
        assert set(v["train_removed"])==held==set(v["backbone_seen"])
        assert len(v["refs"])==5 and set(v["candidates"])==set(frozen["selection"]["outer"])
        for i,r in enumerate(v["refs"]):
            a,b=(set(r[k]) for k in ("train_files","held_files"))
            assert a.isdisjoint(b|held)
            assert a==active-b
            assert {f for f,s in r["target_windows"]}==b
            assert sha(ROOT/r["verified_checkpoint"])==r["final_sha256"]
            original=read_json(VALIDATION/"fits"/f"{r['run']}.json")
            assert original["config"]["inputs"]==r["training_inputs"]
            if i==4:
                assert a==active and b==held
            else:
                assert b<=active
        for arm,item in v["candidates"].items():
            expected=frozen["models"][arm]["config"]["model"]
            assert item["config"]["model"]==expected and item["config"]["scale"]==v["scale"]
            assert item["step"]==6000 and item["target_invariance"]
            assert sha(out/item["checkpoint"])==item["sha256"]
            outer_checkpoints[item["checkpoint"]]=item["sha256"]
    return outer_checkpoints


def verify_state(out,conf,state_row,checkpoint):
    path=out/state_row["path"]
    assert sha(path)==state_row["sha256"]
    state=torch.load(path,map_location="cpu",weights_only=False)
    assert state["config_hash"]==config_hash(conf)
    assert state["step"]==state["scheduler"]["last_epoch"]==state_row["step"]
    assert {int(v["step"]) for v in state["optimizer"]["state"].values()}==({state["step"]} if state["step"] else set())
    assert state["schedule"]==conf["schedule"] and state["identity"]==conf["inputs"]
    assert set(state["rng"])=={"python","numpy","torch","cuda"}
    assert state_digest(state["model"])==state_digest(torch.load(checkpoint,map_location="cpu",weights_only=True))


def audit_incremental_pairs(out,models):
    for suffix in ("0","B","E"):
        left,right=(models[f"{family}{suffix}"] for family in ("I","D"))
        left_cfg,right_cfg=(dict(r["config"]) for r in (left,right))
        assert left_cfg["model"]["family"]=="I" and right_cfg["model"]["family"]=="D"
        for cfg in (left_cfg,right_cfg):
            cfg.pop("name")
            cfg["model"]={k:v for k,v in cfg["model"].items() if k!="family"}
        assert left_cfg==right_cfg
        weight_states=[torch.load(out/r["training"][0]["checkpoint"],map_location="cpu",weights_only=True)
                       for r in (left,right)]
        assert state_digest(weight_states[0])==state_digest(weight_states[1])
        for split in ("train","dev"):
            inc, direct=(r["training"][0] for r in (left,right))
            assert inc[f"{split}_C_delta"]==inc[f"{split}_U_delta"]==inc[f"{split}_G_delta"]==0
            assert direct[f"{split}_mse"]==direct[f"{split}_native"]
            assert abs(inc[f"{split}_mse"]-direct[f"{split}_mse"]-direct[f"{split}_G_delta"])<1e-15


def audit_feedback_pairs(models):
    for loss in ("MSE","aux"):
        ref,own=(models[f"Q_{loss}_{mode}"] for mode in ("ref","self"))
        cfgs=[dict(r["config"],model=dict(r["config"]["model"])) for r in (ref,own)]
        assert [cfg["model"].pop("sequence") for cfg in cfgs]==["ref","self"]
        for cfg in cfgs:
            cfg.pop("name")
        assert cfgs[0]==cfgs[1] and ref["cost"]==own["cost"]
        a,b=([r["sha256"] for r in fit["training"] if r["step"]<=1000] for fit in (ref,own))
        assert a==b


def audit_auxiliary_designs(out,models,branches=("IB","IE")):
    for branch in branches:
        reference=models[branch]["config"]
        calibration=read_json(out/f"{branch}_calibration.json")
        coefficients=calibration["coefficients"]
        assert set(coefficients)=={"energy","match","match2"}
        probes=calibration["probe_indices"]
        norms=calibration["norms"]
        assert len(probes)==8 and set(norms)=={"mse",*coefficients}
        assert all(len(values)==len(probes) and all(isfinite(v) and v>0 for v in values)
                   for values in norms.values())
        prefix=sha(out/f"shared_{branch}.pt")
        for objective,weight in coefficients.items():
            assert isfinite(weight) and weight>0
            assert abs(weight*mean(norms[objective])/mean(norms["mse"])-.3)<1e-12
            name=f"{branch}_{objective}"
            expected=dict(reference,name=name,first=1000,shared=prefix,
                          model=dict(reference["model"],weights={objective:weight}))
            assert models[name]["config"]==expected


def audit_projection_designs(out,models,selection):
    chosen=min(("I0","IB","IE"),key=lambda name:(models[name]["selected"]["dev_mse"],
                                                models[name]["selected"]["step"],name))
    assert chosen==selection["best_MSE_input"]
    expected={f"{base}_K{rank}" for base in (chosen,"D"+chosen[1:]) for rank in (4,8)}
    assert set(selection["representation"])==expected
    for name in expected:
        base,suffix=name.split("_K")
        anchor,projected=models[base],models[name]
        original=anchor["config"]
        assert original["model"]["rank"]==20
        assert projected["config"]==dict(original,name=name,model=dict(original["model"],rank=int(suffix)))
        initial=[torch.load(out/r["training"][0]["checkpoint"],map_location="cpu",weights_only=True)
                 for r in (anchor,projected)]
        assert state_digest(initial[0])==state_digest(initial[1])


def audit_time_mix_designs(out,models):
    anchor=models["IB"]
    original=anchor["config"]
    weights=torch.load(out/anchor["training"][0]["checkpoint"],map_location="cpu",weights_only=True)
    digest=state_digest(weights)
    basis=torch.load(out/"temporal_basis.pt",map_location="cpu",weights_only=True)
    eye=torch.eye(20)
    for mix,expected in (("last",eye[-1:].expand_as(eye)),("reverse",eye.flip(1)),
                         ("lag",eye),("dual",eye),("free",eye)):
        name="IB_"+mix
        fit=models[name]
        cfg=fit["config"]
        assert cfg==dict(original,name=name,model=dict(original["model"],mix=mix))
        initial=torch.load(out/fit["training"][0]["checkpoint"],map_location="cpu",weights_only=True)
        assert state_digest({k:v for k,v in initial.items() if k.startswith("cnn.")})==digest
        head=Head(cfg["model"],basis,torch.device("cpu"))
        head.load_state_dict(initial)
        assert torch.equal(head.mapping(),expected)


def audit_training(out,frozen):
    structured=[read_json(path) for path in sorted((out/"structured_fits").glob("*.json"))]
    ecal=[read_json(path) for path in sorted((out/"fits").glob("Ecal_*.json"))]
    assert len(ecal)==6
    n_outer=sum(r["config"]["name"].startswith("outer_") for r in structured)
    assert n_outer==frozen["counts"]["outer"] and len(structured)-n_outer==25
    assert len(ecal)+len(structured)<=80
    by_name={r["config"]["name"]:r for r in structured}
    audit_incremental_pairs(out,by_name)
    audit_feedback_pairs(by_name)
    audit_auxiliary_designs(out,by_name)
    audit_projection_designs(out,by_name,frozen["selection"])
    audit_time_mix_designs(out,by_name)
    checkpoints=dict(frozen["checkpoints"])
    prefix_files={sha(path):path for path in [*out.glob("shared_*.pt")]}
    checkpoints.update({str(path.relative_to(ROOT)):digest for digest,path in prefix_files.items()})
    verified_states={}
    seq=read_json(out/"feedback_training_selection.json")
    assert seq["trajectories"]==46 and seq["sampled_chains"]==seq["eligible_chains"]
    assert seq["schedule"]==tensor_sha(torch.load(out/"sequence_schedule.pt",map_location="cpu",weights_only=True))
    for r in structured:
        cfg=r["config"]
        assert cfg["seed"]==42 and cfg["updates"]==r["final"]["step"]==6000
        assert cfg["lr"]==.001 and cfg["lr_end"]==1e-6 and cfg["clip"]==1 and cfg["weight_decay"]==1e-5
        assert r["config_hash"]==config_hash(cfg)
        assert cfg["batch"]==8 and cfg["first"]==(1000 if cfg["shared"] else 0)
        assert [row["step"] for row in r["training"]]==cfg["steps"]==list(STEPS)
        for row in r["training"]:
            assert row["train_identity_error"]<1e-15
            assert row["step"]==0 or all(isfinite(row[k]) for k in ("loss","grad","postclip","clip_frequency"))
            path=out/row["checkpoint"]
            assert sha(path)==row["sha256"]
            checkpoints[str(path.relative_to(ROOT))]=row["sha256"]
        assert [s["step"] for s in r["states"]]==cfg["steps"]
        for row,state in zip(r["training"],r["states"]):
            path=out/state["path"]
            assert sha(path)==state["sha256"]
            assert state["borrowed"]==(row["phase"]=="shared_MSE")
            if state["borrowed"]:
                assert cfg["shared"]
            else:
                verify_state(out,cfg,state,out/row["checkpoint"])
                verified_states[state["path"]]=state["sha256"]
            checkpoints[str(path.relative_to(ROOT))]=state["sha256"]
        selected=min(r["training"],key=lambda v:(v["dev_mse"],v["step"])) if "dev_mse" in r["final"] else r["final"]
        assert r["selected"]==selected
        if cfg["shared"]:
            prefix=torch.load(prefix_files[cfg["shared"]],map_location="cpu",weights_only=False)
            assert prefix["step"]==prefix["scheduler"]["last_epoch"]==1000
            assert {int(v["step"]) for v in prefix["optimizer"]["state"].values()}=={1000}
            assert cfg["prefix_schedule"]==prefix["schedule"] and prefix["identity"]==cfg["inputs"]
            expected=[dict(row,run=cfg["name"],phase="shared_MSE") for row in prefix["rows"]]
            assert r["training"][:len(expected)]==expected
        if cfg["model"]["sequence"]:
            assert cfg["schedule"]==seq["schedule"]
            assert r["cost"]==dict(own_updates=5000,forward_training_calls=15000,backward_calls=5000,
                                   supervised_windows=120000,borrowed_updates=1000)
    legacy_archive=read_json(out/"calibrated_state_archive.json")
    assert set(legacy_archive)=={r["config"]["name"] for r in ecal}
    legacy_count=0
    shared_e=OPTIMIZATION/"shared_E.pt"
    shared_state=torch.load(shared_e,map_location="cpu",weights_only=False)
    assert shared_state["step"]==shared_state["scheduler"]["last_epoch"]==1000
    assert {int(v["step"]) for v in shared_state["optimizer"]["state"].values()}=={1000}
    assert set(shared_state["rng"])=={"python","numpy","torch","cuda"}
    for r in ecal:
        cfg=r["config"]
        assert cfg["seed"]==42 and cfg["updates"]==r["final"]["step"]==6000 and cfg["first"]==1000
        assert r["config_hash"]==config_hash(cfg) and cfg["batch"]==8
        assert all(row["train_identity_error"]<1e-15 for row in r["training"])
        archive=legacy_archive[cfg["name"]]
        assert archive["shared_step"]==1000 and sha(shared_e)==archive["shared_sha256"]==cfg["shared"]
        steps={s["step"] for s in archive["retained"]}
        assert not steps.intersection(archive["unavailable"])
        assert steps.union(archive["unavailable"])=={step for step in STEPS if step>1000}
        for state in archive["retained"]:
            path=out/state["path"]
            assert sha(path)==state["sha256"]==frozen["bindings"][state["path"]]
            checkpoints[str(path.relative_to(ROOT))]=state["sha256"]
        legacy_count+=len(steps)
    assert all(verified_states[s["path"]]==s["sha256"] for r in structured for s in r["states"])
    assert all(sha(ROOT/p)==digest for p,digest in checkpoints.items())
    return checkpoints,dict(core=31,outer=n_outer,total=31+n_outer,
                            seconds=sum(r["elapsed_seconds"] for r in [*structured,*ecal]),
                            reused_teacher_weights=20,historical_references=9,
                            retained_legacy_calibration_states=legacy_count,
                            full_archived_checkpoint_states=len(verified_states))


def audit_scores(out,freeze):
    proof=read_json(out/"verification.json")
    digest=sha(out/"FROZEN.json")
    assert proof["freeze_sha256"]==digest
    assert read_json(out/"native_test.json")["test_freeze_sha256"]==digest
    assert proof["identity_max_error"]<1e-15 and proof["delta_identity_max_error"]<1e-15
    assert proof["feature_invariance"]==dict(base=True,s3=True,B=True,E=True)
    assert max(p["absolute_error"] for p in proof["independent_recomputation"])<1e-15
    assert max(p["absolute_error"] for p in read_json(out/"historical_reproduction.json"))<1e-12
    assert {(r["arm"],r["policy"]) for r in proof["future_invariance"] if r["bitwise"]}=={
        (arm,policy) for arm in freeze["models"] for policy in ("selected","final")}
    period=read_json(out/"period_provenance.json")
    assert period["freeze_sha256"]==digest and period["planned_cache_binding"]==freeze["bindings"]["period_plan.json"]
    assert len(period["releases"])==108 and proof["period"]["windows"]==96
    rows=read_rows(out/"feedback_rollout.csv")
    assert len(rows)==4*2*96==proof["period"]["feedback_rows"]
    arrival=read_rows(out/"feedback_arrival_intervention.csv")
    assert len(arrival)==8 and all(r["current_unchanged"]=="True" and r["changed"]=="True"
                                   and float(r["next_forecast_rms"])>0 for r in arrival)
    for (arm,mode,file),items in groups(rows,("arm","run_mode","file")).items():
        ordered=sorted(items,key=lambda r:int(r["start"]))
        assert len(ordered)==8 and ordered[0]["prior_source"]=="fixed_s3"
        assert all(r["future_invariance"]=="True" and int(r["issue_frame"])<int(r["score_frame"]) for r in ordered)
        assert all(float(r["state_rms_from_fixed_s3"])==0 for r in ordered if r["prior_source"]=="fixed_s3")
        for a,b in zip(ordered,ordered[1:]):
            assert int(b["start"])-int(a["start"])==20 and a["score_frame"]==b["issue_frame"]
            if mode=="self":
                assert a["published_sha256"]==b["previous_sha256"] and b["prior_source"]=="published"
            else:
                assert b["prior_source"]=="fixed_s3"
    interactions=read_rows(out/"feedback_interactions.csv")
    assert {(r["split"],r["objective"]) for r in interactions}=={
        (s,o) for s in ("period","period_new") for o in ("MSE","aux")}
    for r in interactions:
        value,a,b,low,high=(float(r[k]) for k in ("interaction","ref_training_shift","self_training_shift","lower","upper"))
        assert abs(value-(a-b))<1e-15 and low<=high and int(r["trajectories"])==12
    summary=read_rows(out/"scores.csv")
    leads=groups(read_rows(out/"leads.csv"),("split","policy","arm"))
    for r in summary:
        cell=tuple(r[k] for k in ("split","policy","arm"))
        group=leads[cell]
        assert sorted(int(x["lead"]) for x in group)==list(range(1,21))
        m,b,c,u,g,s0,cd,ud,gd=(float(r[k]) for k in ("mse","native_mse","C","correction_energy","G",
                                                     "s3_mse","C_delta","U_delta","G_delta"))
        assert max(abs(g-2*c+u),abs(g-b+m),abs(gd-2*cd+ud),abs(gd-s0+m))<1e-15
        for metric in ("mse","native_mse","C","correction_energy","G","s3_mse","C_delta","U_delta","G_delta"):
            assert abs(mean(float(v[metric]) for v in group)-float(r[metric]))<1e-15
    projected=[("I",read_rows(out/"incremental_mode_scores.csv"),"delta_energy"),
               ("D",read_rows(out/"temporal_mode_scores.csv"),"predicted")]
    for family,mode_rows,energy in projected:
        for arm in freeze["selection"]["representation"]:
            if not arm.startswith(family):
                continue
            rank=int(arm.rsplit("_K",1)[1])
            partitions=groups([r for r in mode_rows if r["arm"]==arm],("split","policy","channel"))
            assert partitions
            for modes in partitions.values():
                assert sorted(int(r["mode"]) for r in modes)==list(range(20))
                retained=sum(float(r[energy]) for r in modes if int(r["mode"])<rank)
                outside=sum(float(r[energy]) for r in modes if int(r["mode"])>=rank)
                assert outside<=1e-18+retained*1e-8
    return dict(score_rows=len(summary),feedback_rows=len(rows),recomputed=len(proof["independent_recomputation"]))


def audit_calibration(out):
    calibration=read_json(out/"E_calibration.json")
    rows=groups(read_rows(out/"matched_gradients.csv"),("model","step"))
    leads=groups(read_rows(out/"matched_gradient_leads.csv"),("model","step","split"))
    errors=[]
    for name,weights in calibration["weights"].items():
        coefficient=weights["lambda_"]
        for step in ("1000","2000","4000","6000"):
            probes=rows[name,step]
            assert sorted(int(r["batch"]) for r in probes)==list(range(8))
            for p in probes:
                a,b,t=(float(p[k]) for k in ("mse_norm","auxiliary_norm","total_norm"))
                assert all(isfinite(x) and x>0 for x in (a,b,t))
                assert abs(float(p["aux_to_mse"])-coefficient*b/a)<1e-12
                assert (p["clipped"]=="True")== (t>1)
                assert abs(float(p["coefficient"])-min(1.,1./(t+1e-6)))<1e-12
            if step=="1000":
                achieved=coefficient*sum(float(p["auxiliary_norm"]) for p in probes)/sum(float(p["mse_norm"]) for p in probes)
                errors.append(abs(achieved-weights["rho"]))
            for split in ("train","development"):
                records=leads[name,step,split]
                assert sorted(int(r["lead"]) for r in records)==list(range(1,21))
                assert all(isfinite(float(r["physical_mse"])) for r in records)
    assert max(errors)<1e-6
    return max(errors)


def audit_time_maps(out):
    axes=("model","policy")
    maps=groups(read_rows(out/"time_maps.csv"),axes)
    effects=groups(read_rows(out/"history_interventions.csv"),axes)
    expected={(arm,policy) for arm in ("IB","IB_last","IB_reverse","IB_lag","IB_dual","IB_free")
              for policy in ("selected","final")}
    cells={(i,j) for i in range(1,21) for j in range(1,21)}
    assert set(maps)==set(effects)==expected
    for key in expected:
        arm,_=key
        for records in (maps[key],effects[key]):
            assert len(records)==len(cells)
            assert {(int(r["source"]),int(r["lead"])) for r in records}==cells
        by_lag={}
        for row in maps[key]:
            source,lead=int(row["source"]),int(row["lead"])
            weight=float(row["value"])
            lag=int(row["lag"])
            assert isfinite(weight) and lag==20+lead-source
            if arm=="IB":
                assert weight==float(source==lead)
            elif arm=="IB_last":
                assert weight==float(source==20)
            elif arm=="IB_reverse":
                assert weight==float(source+lead==21)
            elif arm=="IB_lag":
                previous=by_lag.setdefault(lag,weight)
                assert abs(weight-previous)<1e-7
        for row in effects[key]:
            source,lead=int(row["source"]),int(row["lead"])
            response=float(row["response_rms"])
            assert isfinite(response) and response>=0 and int(row["physical_lag"])==20+lead-source
            unavailable=((arm=="IB" and abs(lead-source)>3) or (arm=="IB_last" and source!=20)
                         or (arm=="IB_reverse" and abs(lead-(21-source))>3))
            if unavailable:
                assert response==0
    return len(expected)*len(cells)


def audit(study,figures):
    out=study.out
    freeze=guard(out)
    basis=torch.load(out/"temporal_basis.pt",map_location="cpu",weights_only=True)
    assert read_json(out/"E_calibration.json")["math"]==verify_math()
    checks=math_audit(basis)
    checks["maximum_initial_gradient_ratio_error"]=audit_calibration(out)
    checks["time_map_and_intervention_cells"]=audit_time_maps(out)
    assert read_json(out/"selections.json")==freeze["selection"]
    assert all(sha(ROOT/path)==digest for path,digest in read_json(out/"reuse_manifest.json")["bound_sources"].items())
    teacher_checkpoints=audit_teachers(out,freeze)
    weights,fits=audit_training(out,freeze)
    results=audit_scores(out,freeze)
    images={}
    for path in sorted(figures.glob("*.png")):
        with Image.open(path) as image:
            assert image.format=="PNG" and min(image.size)>=300
            image.verify()
        images[path.name]=sha(path)
    assert len(images)==13
    manifests={str(p.relative_to(out)):sha(p) for p in out.rglob("*")
               if p.is_file() and p.suffix in (".md",".json",".csv",".gz") and p.name!="artifact_audit.json"}
    source_files={str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    atomic_json(out/"artifact_audit.json",dict(passed=True,freeze_sha256=sha(out/"FROZEN.json"),
                                               files=manifests,figures=images,checkpoints=weights,
                                               source_files=source_files,
                                               outer_checkpoints=teacher_checkpoints,math=checks,
                                               training=fits,evaluation=results))
    print("AUDIT COMPLETE",fits,results,flush=True)
