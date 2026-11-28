"""Derive signed physical contrasts and audit the evaluated release."""
from collections import Counter
from itertools import product
from math import isclose,isfinite
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
import torch

from research.corrector.run import sha
from research.corrector_incremental_factorial.publishables import equal_regimes
from research.corrector_incremental_structure.audit import verify_state
from research.corrector_incremental_structure.validation import read_rows
from research.corrector_residual_feedback.data import guard
from research.corrector_simple_baseline.fit import table
from research.corrector_simple_baseline.measure import groups
from research.corrector_spacetime_optimization.fit import config_hash
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_spacetime_validation.objectives import verify_math
from research.corrector_window_study.data import ROOT,read_json

from .run import STRUCTURE


def interval(values, draws):
    order=np.random.default_rng(42).integers(len(values),size=(draws,len(values)))
    bounds=np.quantile(values[order].mean(1),[.025,.975])
    return float(values.mean()),float(bounds[0]),float(bounds[1])


def contrast_tables(out, freeze, plan):
    tr=read_rows(out/"trajectories.csv")
    cohorts=groups(tr,("split","policy"))
    interactions=[]
    early=[]
    indexed={}
    for identity,cells in cohorts.items():
        values={}
        for row in cells:
            values.setdefault(row["arm"],{})[row["trajectory"]]=float(row["mse"])
        indexed[identity]=values
    for (split,policy), values in indexed.items():
        if policy!="final":
            continue
        tracks=sorted(values["S3"])
        assert all(set(x)==set(tracks) for x in values.values())
        arm={n:np.array([v[t] for t in tracks]) for n,v in values.items()}
        for panel in ("fixed_rho","fixed_lambda"):
            for family in ("TM1","TM2","EN"):
                if panel=="fixed_lambda" and family=="EN":
                    continue
                for left,right in [("ID","IB")]+[(x,"I0") for x in ("IB","IC","ID","IE","IF")]:
                    tag=lambda n:f"{n}_{family}" if panel=="fixed_rho" or (n=="I0" and family=="TM1") else f"{n}_{family}_fixed"
                    names=(f"{right}_MSE",f"{left}_MSE",tag(right),tag(left))
                    if not all(n in arm for n in names):
                        continue
                    baseline,method,base_aux,method_aux=(arm[n] for n in names)
                    difference=(base_aux-method_aux)-(baseline-method)
                    effect,low,high=interval(difference,plan["bootstrap"])
                    interactions.append(dict(split=split,policy=policy,comparison=panel,loss=family,arm=left,control=right,
                        samples=len(tracks),mse_interaction=effect,ci_low=low,ci_high=high,
                        mse_gain_plain=float((baseline-method).mean()),mse_gain_aux=float((base_aux-method_aux).mean())))
    for (split,policy), values in indexed.items():
        if policy not in ("final","selected","step500","step1000"):
            continue
        early_arms=freeze["selection"]["early"]
        late_arms=[name for name in freeze["selection"]["core"] if not name.endswith("_MSE")]
        for name in (*early_arms,*late_arms):
            if name not in values:
                continue
            plain=name.split("_")[0]+"_MSE"
            comparison="P5" if name in early_arms else "late_vs_mse_early"
            for control_policy in ("selected","step500"):
                controls=indexed.get((split,control_policy),{})
                if plain not in controls:
                    continue
                v,w=values[name],controls[plain]
                tracks=sorted(v)
                assert set(v)==set(w)
                a,b=(np.array([part[t] for t in tracks]) for part in (v,w))
                rng=np.random.default_rng(42).integers(len(tracks),size=(plan["bootstrap"],len(tracks)))
                effects=100*(1-a[rng].mean(1)/b[rng].mean(1))
                lo,hi=np.quantile(effects,[.025,.975])
                early.append(dict(split=split,policy=policy,comparison=comparison,arm=name,control=plain,
                                  control_policy=control_policy,effect_pct=float(100*(1-a.mean()/b.mean())),
                                  lower_pct=float(lo),upper_pct=float(hi),improved=int((a<b).sum()),
                                  trajectories=len(tracks)))
    table(out/"input_loss_interactions.csv",interactions)
    table(out/"early_stopping_results.csv",early)
    rows=[]
    core=freeze["selection"]["core"]
    for row in read_rows(out/"scores.csv"):
        if row["arm"] in core:
            rows.append(dict(row,structure=row["arm"].split("_")[0],loss=row["arm"].split("_")[1]))
    table(out/"input_loss_matrix.csv",rows)


def training_ledgers(out, models, freeze, plan):
    paths, rows, costs = {}, [], []
    counts = {"new_main":0,"reused_main":0,"new_outer":0,"reused_outer":0}
    for name,item in models.items():
        directory=ROOT/item["directory"]
        new=directory==out
        outer=name.startswith("outer_")
        counts[("new_" if new else "reused_")+("outer" if outer else "main")]+=1
        cfg=item["config"]
        assert cfg["seed"]==42 and cfg["layout"]=="NCDHW" and item["config_hash"]==config_hash(cfg)
        assert (cfg["batch"],cfg["lr"],cfg["lr_end"],cfg["weight_decay"],cfg["clip"])==(8,.001,1e-6,1e-5,1.)
        assert cfg["updates"]==item["final"]["step"]==6000
        costs.append(dict(arm=name,reused=not new,seconds=item["elapsed_seconds"],
                          own_updates=item["cost"]["own_updates"],borrowed_updates=item["cost"]["borrowed_updates"],
                          oof_teachers_per_training_cohort=4,checkpoint_directory=item["directory"]))
        assert [r["step"] for r in item["training"]]==[r["step"] for r in item["states"]]==cfg["steps"]
        for checkpoint,state in zip(item["training"],item["states"]):
            for row,key in ((checkpoint,"checkpoint"),(state,"path")):
                path=directory/row[key]
                assert sha(path)==row["sha256"]
                paths[str(path.relative_to(ROOT))]=row["sha256"]
            if new and not state["borrowed"]:
                verify_state(out,cfg,state,directory/checkpoint["checkpoint"])
            assert isfinite(checkpoint["train_mse"])
            if checkpoint["step"]:
                assert all(isfinite(checkpoint[key]) for key in ("loss","grad","postclip","clip_frequency"))
            rows.append(dict(checkpoint,arm=name,first_aux=cfg["first"],borrowed=state["borrowed"],
                             selected=checkpoint["step"]==item["selected"]["step"],
                             selected_shared_prefix=bool(cfg["first"] and item["selected"]["step"]<=cfg["first"]),
                             family=cfg["model"]["family"],seconds=item["elapsed_seconds"],directory=item["directory"]))
        if cfg["first"]:
            arm=name.split("_")
            control=f"outer_{arm[1]}_{arm[2]}_MSE" if outer else arm[0]+"_MSE"
            if not outer and arm[1] in plan["maps"]:
                control=f"{arm[0]}_{arm[1]}_MSE"
            base=models[control]
            for table_key in ("training","states"):
                prefix=lambda obj:[(r["step"],r["sha256"]) for r in obj[table_key] if r["step"]<=cfg["first"]]
                assert prefix(item)==prefix(base),(name,table_key)
            assert cfg["schedule"]==base["config"]["schedule"]
            past=next(r for r in base["states"] if r["step"]==cfg["first"])
            prefix_path=ROOT/base["directory"]/past["path"]
            if sha(prefix_path)!=cfg["shared"]:
                scope=f"outer_{arm[1]}_IB" if outer else arm[0]
                alternate=STRUCTURE/f"shared_{scope}.pt"
                assert sha(alternate)==cfg["shared"],name
    assert counts["new_main"]==freeze["main_new"]
    assert counts["new_main"]+counts["new_outer"]<=plan["budget"]
    columns=sorted({k for row in rows for k in row})
    table(out/"training.csv",[{k:row.get(k) for k in columns} for row in rows])
    table(out/"training_cost.csv",costs)
    return paths,counts


def audit_lineage(out, publication):
    triplets=read_json(out/"triplet_manifest.json")
    stamp=read_json(out/"reference_lineage.json")
    distinct={}
    for split,nw,nt in (("train",822,46),("development",24,12),("test",24,12)):
        cases=triplets[split]["cases"]
        files={c["file"] for c in cases}
        assert len(cases)==nw and len(files)==nt
        assert all(files.isdisjoint(old) for old in distinct.values())
        distinct[split]=files
        signatures=[]
        for c in cases:
            first=c["start"]
            assert c["earlier"]==[first-20,first]
            assert c["history"]==[first,first+20]
            assert c["target"]==[first+20,first+40]
            assert c["issue_frame"]==first+20 and c["score_frame"]==first+40
            signatures.append([c["file"],first])
        assert len({tuple(p) for p in signatures})==nw
        teacher=read_json(out/f"teacher_{split}.json")
        assert teacher["keys"]==signatures
        native=read_json(out/f"native_{split}.json")["added_recipe"]
        assert native["backbone"]==stamp["backbone"] and native["denoising_labels"]==[99,66,33,0]
        assert native["zero_initial_noise"]
        if split=="train":
            assigned=[]
            for fold in teacher["refs"]:
                scope=set(fold["predicted"])
                assert set(fold["train"])==files-scope
                assert {cases[i]["file"] for i in fold["ids"]}==scope
                assigned.extend(fold["ids"])
            assert sorted(assigned)==list(range(nw)) and len(teacher["refs"])==4
            assert stamp["current_and_history_same_fold"] and stamp["train"]==teacher
        else:
            selected,=teacher["refs"]
            assert selected["sha256"]==publication["s3"] and selected["scope"]=="full"
    return dict(windows={part:len(data["cases"]) for part,data in triplets.items()},
                trajectories={part:len(files) for part,files in distinct.items()},
                same_fold_current_and_history=True,observed_before_release=True)




def audit_transfer_diagnostics(out):
    cases=read_json(out/"triplet_manifest.json")["train"]["cases"]
    regimes={str(row["re"]) for row in cases}|{"all"}
    pairs=list(product(("R","P","E"),("Rn","En")))
    pairs.extend(("E",field) for field in ("S3","last","native","observed"))
    lead_axes=("source_index","target_index")
    lead_values=tuple(str(k) for k in range(1,21))
    specs=(
        ("temporal_correlations",regimes,lead_axes,(lead_values,lead_values),("inner","pearson","cosine")),
        ("transfer_modes",{"all"},lead_axes,(lead_values,lead_values),("inner","pearson","cosine")),
        ("spatial_correlations",{"all"},("y","x"),(tuple(map(str,range(32))),tuple(map(str,range(64)))),
         ("inner","pearson","cosine")),
        ("residual_transfer_correlations",regimes,(),(),("inner","pearson","cosine","source_rms","target_rms")),
        ("amplitude_transfer",regimes,("quantity",),(("energy","rms"),),("pearson","source_mean","target_mean")),
    )
    counts={}
    for file,scope,indices,coordinates,metrics in specs:
        rows=read_rows(out/f"{file}.csv")
        keys=("source","target","re","channel",*indices)
        expected={(source,target,re,channel,*pos) for (source,target),re,channel in product(pairs,scope,("u","v"))
                  for pos in product(*coordinates)}
        assert len(rows)==len(expected) and {tuple(r[key] for key in keys) for r in rows}==expected
        for row in rows:
            assert all(isfinite(float(row[key])) for key in metrics)
            assert abs(float(row["pearson"]))<=1.+1e-9
            if "cosine" in row:
                assert abs(float(row["cosine"]))<=1.+1e-9
            if file=="temporal_correlations":
                i,j=(int(row[key]) for key in lead_axes)
                assert int(row["lag"])==20+j-i
                assert row["same_lead"]==str(i==j) and row["boundary"]==str(i==20 and j==1)
        counts[file]=len(rows)
    return counts


def audit_donors(out):
    triplets=read_json(out/"triplet_manifest.json")
    assignments=groups(read_rows(out/"history_donors.csv"),("split","re"))
    intervention_counts=groups(read_rows(out/"history_interventions.csv"),("split",))
    coverage={}
    for flag,split in (("development","development"),("test","historical_test")):
        cases=triplets[flag]["cases"]
        regimes=groups(cases,("re",))
        assert {re for label,re in assignments if label==split}=={str(re) for (re,) in regimes}
        moved_total=0
        for (re,),windows in regimes.items():
            expected={(r["file"],r["start"]) for r in windows}
            rows=assignments[split,str(re)]
            source=[(r["trajectory"],int(r["start"])) for r in rows]
            donor=[(r["donor_trajectory"],int(r["donor_start"])) for r in rows]
            assert len(source)==len(expected)==len(set(donor)) and set(source)==set(donor)==expected
            assert all(a==b or a[0]!=b[0] for a,b in zip(source,donor))
            assert all(r["moved"]==str(a!=b) for r,a,b in zip(rows,source,donor))
            moved=sum(a!=b for a,b in zip(source,donor))
            largest=max(Counter(file for file,_ in expected).values())
            assert moved==min(len(expected),2*(len(expected)-largest))
            moved_total+=moved
        assert {int(r["eligible_swaps"]) for r in intervention_counts[(split,)]}=={moved_total}
        coverage[split]=dict(windows=len(cases),moved=moved_total)
    return coverage



def audit_gradients(out,freeze,plan):
    calibration=read_json(out/"calibration.json")
    arms=calibration["arms"]
    models=freeze["models"]
    selection=freeze["selection"]
    calibrated={n for n in (*selection["core"],*selection["early"],*selection["extensions"])
                if models[n]["config"]["first"]}
    assert set(arms)==calibrated
    assert len(calibration["batches"])==8 and all(len(batch)==8 for batch in calibration["batches"])
    records={}
    for row in read_rows(out/"gradient_diagnostics.csv"):
        name,step,term,batch=row["arm"],int(row["step"]),row["term"],int(row["batch"])
        identity=(name,step,term,batch)
        assert identity not in records
        records[identity]=row
        numeric=("loss","weight","norm","mse_norm","weighted_ratio","cosine_to_mse",
                 "total_norm","clip_factor","conv1","conv2","conv3")
        values={key:float(row[key]) for key in numeric}
        assert all(isfinite(value) for value in values.values())
        expected_weight=models[name]["config"]["model"]["weights"].get(term,1.)
        assert values["weight"]==expected_weight and values["mse_norm"]>0
        ratio=expected_weight*values["norm"]/values["mse_norm"]
        assert isclose(values["weighted_ratio"],ratio,rel_tol=1e-12,abs_tol=1e-15)
        assert abs(values["cosine_to_mse"])<=1.+1e-6
        assert values["clip_factor"]==min(1.,1./(values["total_norm"]+1e-6))
    expected=set()
    for name,record in models.items():
        cfg=record["config"]
        steps=set(plan["monitor"])
        if name in arms:
            steps.add(arms[name]["step"])
        terms=("mse",*cfg["model"]["weights"])
        expected.update((name,step,term,batch) for step in steps if step>=cfg["first"]
                        for term in terms for batch in range(len(calibration["batches"])))
    assert set(records)==expected
    ratios={}
    for name,row in arms.items():
        cfg=models[name]["config"]
        assert row["step"]==cfg["first"] and row["actual"]==cfg["model"]["weights"][row["term"]]
        probes=[records[name,row["step"],row["term"],i] for i in range(len(calibration["batches"]))]
        ratio=sum(float(r["norm"])*float(r["weight"]) for r in probes)/sum(float(r["mse_norm"]) for r in probes)
        assert abs(ratio-plan["rho"])<plan["rho_tolerance"]
        ratios[name]=ratio
    return dict(rows=len(records),calibration_ratios=ratios)


def complete(out,figures,freeze,outer,verification,plan):
    assert guard(out)==freeze
    digest=sha(out/"FROZEN.json")
    assert read_json(out/"native_test.json")["test_freeze_sha256"]==digest
    assert read_json(out/"period_provenance.json")["freeze_sha256"]==digest
    publication=read_json(out/"publication_verification.json")
    assert publication["freeze_sha256"]==digest and publication["backbone_frozen"] and publication["s3_frozen"]
    assert publication["algebra"] and publication["restoration_max_error"]<1e-15
    portable=publication["portable_releases"]
    assert set(portable)=={"final","selected"}
    for policy,result in portable.items():
        assert result["arm"]==freeze["selection"][policy] and result["bitwise"]
    assert portable["final"]["native_sha256"]==portable["selected"]["native_sha256"]
    assert portable["final"]["s3_sha256"]==portable["selected"]["s3_sha256"]
    verification["data_lineage"]=audit_lineage(out,publication)
    verification["fold_out_transfer_diagnostics"]=audit_transfer_diagnostics(out)
    verification["history_donor_coverage"]=audit_donors(out)
    assert all(x["hot_calls"]==[99,66,33,0] and x["zero_increment_bitwise"] and x["altered_future_bitwise"]
               for x in publication["models"].values())
    assert verification["identity_max_error"]<1e-15 and verification["delta_identity_max_error"]<1e-15
    assert all(row["bitwise"] for row in verification["target_invariance"])
    assert max(r["absolute_error"] for r in verification["independent_recomputation"])<1e-15
    lineage=read_json(out/"outer_lineage.json")
    assert lineage["freeze_sha256"]==digest and set(lineage["regimes"])==set(plan["outer"])
    split=read_json(out/"outer_plan.json")
    outer_arms={f"{arm}_{loss}" for arm in plan["focal"] for loss in plan["terms"]}
    if freeze["selection"]["outer_extra"]:
        outer_arms.add(freeze["selection"]["outer_extra"])
    for region,item in lineage["regimes"].items():
        assert set(item["candidates"])==outer_arms
        held=set(item["train_removed"])
        assert held==set(split[region]["held_files"])==set(item["backbone_seen"])
        assert item["phys_re"]==plan["outer"][region] and len(item["refs"])==5
        active=set(split[region]["active_files"])
        assert active.isdisjoint(held)
        for entry in item["refs"]:
            predicted=set(entry["held_files"])
            assert set(entry["train_files"])==active-predicted
            assert predicted=={f for f,_ in entry["target_windows"]}
            assert (predicted==held) if entry["fold"]==4 else (predicted<=active)
            assert held.isdisjoint(entry["train_files"])
            assert sha(ROOT/entry["verified_checkpoint"])==entry["final_sha256"]
        for arm in outer_arms:
            cfg=item["candidates"][arm]["config"]
            reference=freeze["models"][arm]["config"]
            assert cfg["model"]==reference["model"] and cfg["first"]==reference["first"]
            assert cfg["scale"]==item["scale"] and cfg["schedule"]==split[region]["sampling_sha256"]
            assert cfg["inputs"]["development"] is None
            assert cfg["inputs"]["current"]==item["train_current"]
            assert cfg["inputs"]["previous"]==item["train_history"]
    assert len(freeze["selection"]["core"])==24 and len(freeze["selection"]["early"])==6
    for name in freeze["selection"]["early"]:
        assert freeze["models"][name]["config"]["first"]==250
    for name in freeze["selection"]["fixed"]:
        cfg=freeze["models"][name]["config"]["model"]
        branch="TM2" if "TM2" in name else "TM1"
        assert list(cfg["weights"].values())==[freeze["anchors"][branch]]
    verification["gradient_audit"]=audit_gradients(out,freeze,plan)
    scores={(r["split"],r["policy"],r["arm"]):float(r["mse"]) for r in read_rows(out/"scores.csv")}
    scored_models={key for key in scores if key[2] not in ("Native","S3")}
    checked_steps={}
    fixed_steps={"step500":500,"step1000":1000,"final":6000}
    for row in verification["target_invariance"]:
        split,policy,arm=row["split"],row["policy"],row["arm"]
        key=(split,policy,arm)
        assert key not in checked_steps
        expected_step=freeze["models"][arm]["selected"]["step"] if policy=="selected" else fixed_steps[policy]
        assert row["step"]==expected_step and row["freeze_sha256"]==digest
        record=outer[f"outer_{split[len('holdout_'):]}_{arm}"] if split.startswith("holdout_") else freeze["models"][arm]
        expected=next(r for r in record["training"] if r["step"]==expected_step)
        assert row["config_hash"]==record["config_hash"]
        assert row["checkpoint"]==expected["checkpoint"] and row["checkpoint_sha256"]==expected["sha256"]
        checked_steps[key]=expected_step
    assert set(checked_steps)==scored_models
    candidates=(*freeze["selection"]["core"],*freeze["selection"]["early"],*freeze["selection"]["extensions"])
    development_selection={}
    for policy in ("final","selected"):
        arm=freeze["selection"][policy]
        recorded=scores["development",policy,arm]
        best=min(scores["development",policy,name] for name in candidates)
        assert abs(recorded-best)<1e-15
        development_selection[policy]=dict(arm=arm,step=checked_steps["development",policy,arm],
                                           mse=recorded,minimum_mse=best)
    verification["development_selection"]=development_selection
    for key,rows in groups(read_rows(out/"leads.csv"),("split","policy","arm")).items():
        assert sorted(int(r["lead"]) for r in rows)==list(range(1,21))
        assert abs(mean(float(r["mse"]) for r in rows)-scores[key])<1e-15
    assert len(scores)==verification["scores"]
    checkpoint_hashes,counts=training_ledgers(out,{**freeze["models"],**outer},freeze,plan)
    contrast_tables(out,freeze,plan)
    equal_regimes(out)
    direction=[]
    for row in read_rows(out/"scores.csv"):
        direction.append(dict(split=row["split"],policy=row["policy"],arm=row["arm"],
            global_inner=row["C_delta"],global_cosine=row["global_delta_cosine"],window_mean_cosine=row["delta_cosine"],
            delta_energy=row["U_delta"],remaining_energy=row["s3_mse"],gain=row["G_delta"],
            energy_ratio=float(row["U_delta"])/float(row["s3_mse"]),
            window_mean_rms_ratio=row["window_delta_rms_ratio"]))
    table(out/"direction_energy_summary.csv",direction)
    verification.update(freeze_sha256=digest,publication=publication,training=counts,loss_math=verify_math(),
                        runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name()))
    atomic_json(out/"verification.json",verification)
    from .report import plot,render

    render(out,freeze,plan)
    plot(out,figures,freeze,plan)
    figure_hashes={}
    for path in figures.glob("*.png"):
        with Image.open(path) as image:
            assert image.format=="PNG" and min(image.size)>300
            image.verify()
        figure_hashes[path.name]=sha(path)
    assert len(figure_hashes)==14
    roots=list(Path(__file__).parent.glob("*.py"))+[ROOT/"research"/module for module in
          ("corrector_incremental_structure/model.py","corrector_incremental_structure/fit.py",
           "corrector_incremental_structure/score.py","corrector_incremental_factorial/core.py")]
    files={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file() and
           p.suffix in (".csv",".json",".md",".pt",".gz") and p.name!="artifact_audit.json"}
    atomic_json(out/"artifact_audit.json",dict(passed=True,freeze_sha256=digest,files=files,figures=figure_hashes,
                checkpoints=checkpoint_hashes,source_files={str(p.relative_to(ROOT)):sha(p) for p in roots},training=counts))
    print("AUDIT COMPLETE",counts,flush=True)
