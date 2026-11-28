"""Audit the frozen study, then bind human and machine readable deliverables."""
from collections import Counter
from math import isfinite
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch
from PIL import Image

from research.corrector.run import sha
from research.corrector_incremental_factorial.publishables import equal_regimes
from research.corrector_incremental_structure.audit import verify_state
from research.corrector_incremental_structure.validation import read_rows
from research.corrector_residual_feedback.data import guard
from research.corrector_residual_feedback.fit import state_digest
from research.corrector_residual_optimization.fit import shared_plan
from research.corrector_simple_baseline.fit import table
from research.corrector_simple_baseline.measure import groups
from research.corrector_spacetime_optimization.fit import config_hash
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_window_study.data import ROOT, read_json, tensor_sha

from .metrics import eligible_pairs, paired_metrics, stratify, summarize_gain
from .publication import release_checks
from .run import PREVIOUS


def ledger(out,freeze,outer,teachers):
    tagged={**freeze["models"]}
    for scope,v in outer.items():
        tagged.update({f"outer_{scope}_{arm}":r for arm,r in v["models"].items()})
        tagged.update({f"outer_{scope}_{arm}_prefix":r for arm,r in v["training_only"].items()})
    sources={**tagged,**{k:dict(v,directory=str(out.relative_to(ROOT))) for k,v in teachers.items()}}
    hashes,training,costs={},{},[]
    counts=dict(main_new=0,main_reused=0,outer_new=0,outer_reused=0,prefix_phases=0,teachers=len(teachers))
    for name,record in sources.items():
        directory=ROOT/record["directory"]
        new=directory==out
        cfg=record["config"]
        assert cfg["seed"]==42 and cfg["layout"]=="NCDHW" and config_hash(cfg)==record["config_hash"]
        assert cfg["updates"]==6000 and (cfg["batch"],cfg["clip"],cfg["weight_decay"])==(8,1.,1e-5)
        expected=[step for step in cfg["steps"] if step<=cfg.get("end_step",6000)]
        assert expected==[r["step"] for r in record["training"]]==[r["step"] for r in record["states"]]
        if name not in teachers:
            clock="outer" if name.startswith("outer_") else "main"
            if name.endswith("_prefix"):
                counts["prefix_phases"]+=1
            else:
                counts[f"{clock}_{'new' if new else 'reused'}"]+=1
        costs.append(dict(arm=name,reused=not new,seconds=record["elapsed_seconds"],**record.get("cost",dict(
                          own_updates=6000,borrowed_updates=0,forward_training_calls=6000,backward_calls=6000,
                          supervised_windows=48000))))
        for row,state in zip(record["training"],record["states"]):
            for item,key in ((row,"checkpoint"),(state,"path")):
                path=directory/item[key]
                assert sha(path)==item["sha256"]
                key_path=str(path.relative_to(ROOT))
                assert hashes.get(key_path,item["sha256"])==item["sha256"]
                hashes[key_path]=item["sha256"]
            assert isfinite(row["train_mse"]) and (row["step"]==0 or isfinite(row["loss"]))
            if name in teachers:
                saved=torch.load(directory/state["path"],map_location="cpu",weights_only=False)
                assert saved["config_hash"]==record["config_hash"]
                assert saved["step"]==state["step"]==saved["scheduler"]["last_epoch"]
                assert saved["schedule_sha256"]==cfg["schedule"]
                assert state_digest(saved["model"])==state_digest(torch.load(directory/row["checkpoint"],weights_only=True))
            elif new and not state.get("borrowed",False):
                verify_state(out,cfg,state,directory/row["checkpoint"])
            training[name,row["step"]]=dict(arm=name,checkpoint_directory=record["directory"],**row)
        if cfg["first"] and new:
            family=name.removeprefix("outer_") if hasattr(name,"removeprefix") else name[6:] if name.startswith("outer_") else name
            scope=family.split("_",1)[0] if name.startswith("outer_") else None
            arm=family.split("_",1)[1] if scope else name
            plain=arm.rsplit("_",1)[0]+"_MSE"
            if arm=="ID_TM1_early":
                plain="ID_MSE"
            base=f"outer_{scope}_{plain}" if scope else plain
            baseline=sources.get(base) or sources[base+"_prefix"]
            for key in ("training","states"):
                def common(record):
                    return [(r["step"],r["sha256"]) for r in record[key] if r["step"]<=cfg["first"]]
                assert common(baseline)==common(record),name
            assert baseline["config"]["schedule"]==cfg["schedule"]
    columns=sorted({k for r in training.values() for k in r})
    table(out/"training.csv",[{k:r.get(k) for k in columns} for r in training.values()])
    table(out/"training_cost.csv",costs)
    assert sum(counts[k] for k in ("main_new","outer_new","prefix_phases","teachers"))==freeze["new_runs"]
    return hashes,counts



def audit_training_donors(out, scope, cases, entry, rows):
    schedule=shared_plan(SimpleNamespace(cases=cases))
    chosen=torch.load(out/entry["file"],map_location="cpu",weights_only=True)
    assert chosen.shape==schedule.shape==(6000,8)
    assert sha(out/entry["file"])==entry["sha256"]
    assert tensor_sha(schedule)==entry["sampling_sha256"] and tensor_sha(chosen)==entry["tensor_sha256"]
    assert int(chosen.min())>=0 and int(chosen.max())<len(cases)
    counts=Counter(zip(schedule.flatten().tolist(),chosen.flatten().tolist()))
    registered={(int(r["recipient"]),int(r["donor"])):r for r in rows if r["scope"]==scope}
    assert len(counts)==len(registered) and set(counts)==set(registered)
    legal=groups(cases,("re",))
    closest={}
    pad=read_json(out/"donor_plan.json")["time_slack_frames"]
    for i,case in enumerate(cases):
        different=[abs(c["start"]-case["start"]) for c in legal[(case["re"],)] if c["file"]!=case["file"]]
        closest[i]=min(different)+pad if different else None
    moved=0
    for (i,j),count in counts.items():
        row=registered[i,j]
        left,right=cases[i],cases[j]
        assert int(row["count"])==count and float(row["re"])==left["re"]==right["re"]
        assert (row["recipient_file"],int(row["recipient_start"]))==(left["file"],left["start"])
        assert (row["donor_file"],int(row["donor_start"]))==(right["file"],right["start"])
        assert row["moved"]==str(i!=j)
        if closest[i] is None:
            assert i==j
        else:
            assert left["file"]!=right["file"] and abs(left["start"]-right["start"])<=closest[i]
        moved+=count*(i!=j)
    assert entry["assignments"]==entry["same_re"]==sum(counts.values())==48000
    assert moved==entry["moved"]
    return dict(assignments=entry["assignments"],moved=moved,unique_donors=len(set(chosen.flatten().tolist())))


def audit_cohorts(out,freeze,outer,teachers):
    recorded=read_json(out/"teacher_lineage.json")
    assert recorded["backbone"]==freeze["backbone"]
    train=recorded["main"]["train"]
    assert len(train["keys"])==822 and len({k[0] for k in train["keys"]})==46
    assigned=[]
    for ref in train["refs"]:
        held=set(ref["predicted"])
        assert held.isdisjoint(ref["train"])
        assert held=={train["keys"][i][0] for i in ref["ids"]}
        assigned+=ref["ids"]
    assert sorted(assigned)==list(range(822))
    for scope,record in outer.items():
        identity=record["teacher"]
        excluded=set(identity["train_removed"])
        assert identity==recorded["outer"][scope] and len(identity["refs"])==5
        if scope=="additional":
            normal=identity["normalization"]
            assert normal["training_files"]==identity["refs"][4]["train_files"]
            assert normal==identity["refs"][4]["normalization"]
        for ref in identity["refs"]:
            assert excluded.isdisjoint(ref["train_files"])
            assert set(ref["held_files"])=={k[0] for k in ref["target_windows"]}
            if ref["fold"]==4:
                assert set(ref["held_files"])==excluded
            else:
                assert excluded.isdisjoint(ref["held_files"])
            if scope=="additional":
                teacher=teachers[ref["run"]]
                assert teacher["final"]["sha256"]==ref["final_sha256"]
                normal=ref["normalization"]
                assert normal==teacher["config"]["origin"]["normalization"]
                assert normal["source"]=="trajectory_equal_observed_windows"
                assert normal["training_files"]==ref["train_files"]
                assert len(normal["mean"])==len(normal["std"])==2
                assert all(isfinite(v) for v in (*normal["mean"],*normal["std"])) and min(normal["std"])>0
            else:
                assert sha(ROOT/ref["verified_checkpoint"])==ref["final_sha256"]
    cases=read_json(out/"triplet_manifest.json")["train"]["cases"]
    plan=read_json(out/"outer_plan.json")
    donor_rows=read_rows(out/"history_donors_main.csv")+read_rows(out/"history_donors_outer.csv")
    coverage={"main":audit_training_donors(out,"main",cases,read_json(out/"donor_main.json"),donor_rows)}
    for scope,entry in read_json(out/"donor_outer.json").items():
        if scope in plan:
            selected=[cases[i] for i in plan[scope]["active"]]
        else:
            held=set(outer[scope]["teacher"]["train_removed"])
            selected=[c for c in cases if c["file"] not in held]
        coverage[scope]=audit_training_donors(out,scope,selected,entry,donor_rows)
    assert guard(out)==freeze
    return dict(training_trajectories=46,training_windows=822,valid_outer=list(outer),donor_coverage=coverage)


def release(evaluator,figures,outer,teachers,verification,independent):
    out=evaluator.study.out
    freeze=evaluator.freeze
    digest=sha(out/"FROZEN.json")
    assert verification["identity_max_error"]<1e-15 and verification["delta_identity_max_error"]<1e-15
    assert max(r["absolute_error"] for r in verification["independent_recomputation"])<1e-15
    cohort=audit_cohorts(out,freeze,outer,teachers)
    hashes,counts=ledger(out,freeze,outer,teachers)
    evidence=release_checks(evaluator)
    assert evidence["backbone_frozen"] and evidence["s3_frozen"]
    assessed=read_rows(out/"history_donors_main.csv")+read_rows(out/"history_donors_outer.csv")
    table(out/"history_donors.csv",assessed+evaluator.assigned)
    table(out/"eligible_history_sensitivity.csv",evaluator.sensitivity)
    excluded={scope:set(v["teacher"]["train_removed"]) for scope,v in outer.items()}
    for row in assessed:
        a,b=row["recipient_file"],row["donor_file"]
        assert a not in excluded.get(row["scope"],()) and b not in excluded.get(row["scope"],())
        assert (a!=b)==(row["moved"]=="True")
    eligible_pairs(out,evaluator.sensitivity)
    paired_metrics(out)
    stratify(out,evaluator.strata)
    summarize_gain(out,evaluator.gain)
    equal_regimes(out)
    contrast=read_rows(out/"contrasts.csv")
    table(out/"early_stopping_results.csv",[r for r in contrast
          if r["comparison"] in ("P8","P1","P4","P5","P6","P7")])
    lookup={(r["split"],r["policy"],r["arm"]):float(r["mse"]) for r in read_rows(out/"scores.csv")}
    for key,rows in groups(read_rows(out/"leads.csv"),("split","policy","arm")).items():
        assert sorted(int(row["lead"]) for row in rows)==list(range(1,21))
        assert abs(mean(float(row["mse"]) for row in rows)-lookup[key])<1e-15
    atomic_json(out/"verification.json",dict(verification,freeze_sha256=digest,publish=evidence,cohorts=cohort,
        checkpoints=evaluator.checkpoints,counts=counts,independent=independent,phase_donor_isolation=True))
    from .report import render, plot

    render(out,freeze,evaluator.plan)
    plot(out,figures,freeze,evaluator.plan)
    images={}
    for f in sorted(figures.glob("*.png")):
        with Image.open(f) as image:
            assert min(image.size)>300
            image.verify()
        images[f.name]=sha(f)
    assert len(images)==12
    module=Path(__file__).parent
    code=sorted(module.glob("*.py"))+[ROOT/"research"/s for s in (
        "corrector_incremental_structure/fit.py","corrector_incremental_structure/model.py",
        "corrector_remaining_residual_transfer/publication.py","corrector_spacetime_validation/fit.py")]
    files={str(f.relative_to(out)):sha(f) for f in out.rglob("*") if f.is_file()
           and f.name!="artifact_audit.json" and f.suffix in (".csv",".json",".md",".pt",".gz")}
    atomic_json(out/"artifact_audit.json",dict(passed=True,freeze_sha256=digest,files=files,figures=images,
        checkpoints=hashes,source_files={str(f.relative_to(ROOT)):sha(f) for f in code},training=counts))
    print("AUDIT COMPLETE",counts,flush=True)
