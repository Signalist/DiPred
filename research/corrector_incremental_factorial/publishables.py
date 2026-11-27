"""Produce and audit the final research artifacts."""
from math import isfinite
from pathlib import Path
from statistics import mean

import torch
from PIL import Image

from research.corrector.run import sha
from research.corrector_residual_feedback.data import ROOT, guard
from research.corrector_residual_feedback.fit import state_digest
from research.corrector_simple_baseline.fit import table
from research.corrector_simple_baseline.measure import groups
from research.corrector_spacetime_optimization.fit import config_hash
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_window_study.data import read_json
from research.corrector_incremental_structure.audit import math_audit, verify_state
from research.corrector_incremental_structure.validation import read_rows

from .core import FACTORIAL
from .mechanisms import effect_terms
from .plot import plot
from .report import report


def equal_regimes(out):
    rows=read_rows(out/"regimes.csv")
    by=groups(rows,("split","policy","arm"))
    cells=[]
    for (split,policy,arm), own in by.items():
        assert len({r["re"] for r in own})==len(own)
        value=mean(float(r["mse"]) for r in own)
        baseline,s3=(by[split,policy,ref] for ref in ("Native","S3"))
        assert {r["re"] for r in own}=={r["re"] for r in baseline}=={r["re"] for r in s3}
        cells.append(dict(split=split,policy=policy,arm=arm,regimes=len(own),mean_mse=value,
                          native_pct=100*(1-value/mean(float(r["mse"]) for r in baseline)),
                          s3_pct=100*(1-value/mean(float(r["mse"]) for r in s3))))
    table(out/"equal_regime_scores.csv",cells)


def audit_scores(out, freeze):
    outcome=read_json(out/"verification.json")
    assert outcome["freeze_sha256"]==sha(out/"FROZEN.json") and outcome["period_windows"]==96 and outcome["later_windows"]==48
    assert read_json(out/"native_test.json")["test_freeze_sha256"]==outcome["freeze_sha256"]
    assert outcome["identity_max_error"]<1e-15 and outcome["delta_identity_max_error"]<1e-15
    assert outcome["previous_reproduction_rows"]>=43
    assert max(r["absolute_error"] for r in outcome["independent_recomputation"])<1e-15
    assert {(r["arm"],r["policy"]) for r in outcome["future_invariance"] if r["bitwise"]}=={
        (a,p) for a in freeze["models"] for p in ("selected","final")}
    leads=groups(read_rows(out/"leads.csv"),("split","policy","arm"))
    scores={(r["split"],r["policy"],r["arm"]):r for r in read_rows(out/"scores.csv")}
    assert len(scores)==len(outcome["independent_recomputation"])
    assert {r["arm"] for r in read_rows(out/"holdout_results.csv")}=={"Native", "S3", *FACTORIAL}
    for region in ("interpolation", "upper", "lower"):
        assert {arm for split, policy, arm in scores if split==f"holdout_{region}" and policy=="final"}=={
            "Native", "S3", *FACTORIAL}
    for cell, row in scores.items():
        selected=leads[cell]
        assert sorted(int(r["lead"]) for r in selected)==list(range(1,21))
        for metric in ("mse","native_mse","C","correction_energy","G","s3_mse","C_delta","U_delta","G_delta"):
            assert abs(mean(float(r[metric]) for r in selected)-float(row[metric]))<1e-15
    effect_rows=read_rows(out/"factorial_effects.csv")+read_rows(out/"factorial_interactions.csv")
    defined={name: weights for _,name,_,weights in effect_terms()}
    for row in effect_rows:
        a,p,k=(row[x] for x in ("split","policy","effect"))
        observed=sum(w*float(scores[a,p,arm]["mse"]) for arm,w in defined[k].items())
        assert abs(observed-float(row["mse_reduction"]))<1e-15
        assert float(row["lower"])<=float(row["upper"])
    for key, rows in groups(effect_rows,("split","policy")).items():
        assert {r["effect"] for r in rows}==set(defined)
    history=groups(read_rows(out/"history_interventions.csv"),("split","policy","arm","intervention"))
    assert len(history)==2*2*29
    for (split,policy,arm,kind), cells in history.items():
        assert sorted(int(r["lead"]) for r in cells)==list(range(1,21))
        if kind=="original":
            assert abs(mean(float(r["mse"]) for r in cells)-float(scores[split,policy,arm]["mse"]))<1e-15
    return dict(scores=len(scores),factorial_contrasts=len(effect_rows),intervention_cells=len(history)*20)


def audit_initialization(models):
    comparisons = []
    for cohort in ("", "outer_interpolation_", "outer_upper_", "outer_lower_"):
        for suffix in ("0", "B"):
            incremental, direct = (models[cohort + family + suffix] for family in ("I", "D"))
            digests = []
            for record in (incremental, direct):
                first = record["training"][0]
                assert first["step"] == 0
                weight = torch.load(ROOT / record["directory"] / first["checkpoint"],
                                    map_location="cpu", weights_only=True)
                digests.append(state_digest(weight))
            assert digests[0] == digests[1]
            a, b = incremental["training"][0], direct["training"][0]
            for split in ("train", "dev") if not cohort else ("train",):
                assert a[f"{split}_C_delta"] == a[f"{split}_U_delta"] == a[f"{split}_G_delta"] == 0
                assert b[f"{split}_mse"] == b[f"{split}_native"]
                assert abs(a[f"{split}_mse"] - b[f"{split}_mse"] - b[f"{split}_G_delta"]) < 1e-15
            comparisons.append((incremental["config"]["name"], direct["config"]["name"]))
    return comparisons


def audit_gradients(out, freeze):
    groups_by_probe = groups(read_rows(out / "gradient_diagnostics.csv"), ("arm", "step", "term"))
    calibration = read_json(out / "initial_calibration.json")
    targets = {name: detail["term"] for name, detail in freeze["selection"]["extensions"].items()}
    targets.update({name: "match2" for name in freeze["models"] if name.endswith("_rho")})
    ratios = {}
    for name, term in targets.items():
        rows = groups_by_probe[name, "1000", term]
        assert len(rows) == len(calibration["batches"])
        actual = mean(float(r["weight"]) * float(r["norm"]) for r in rows) / mean(float(r["mse_norm"]) for r in rows)
        assert abs(actual - calibration["rho"]) < 1e-6, (name, actual)
        ratios[name] = actual
    for rows in groups_by_probe.values():
        assert all(isfinite(float(r[k])) for r in rows for k in
                   ("loss", "weight", "norm", "mse_norm", "weighted_ratio", "cosine_to_mse", "total_norm", "clip_factor"))
    return ratios


def audit_models(out, f, outer):
    all_models={**f["models"],**outer}
    checkpoints={}
    fitted={}
    for name,d in all_models.items():
        parent=ROOT/d["directory"]
        cfg=d["config"]
        assert cfg["seed"]==42 and cfg["updates"]==d["final"]["step"]==6000
        assert cfg["layout"]=="NCDHW" and cfg["batch"]==8 and cfg["clip"]==1
        assert tuple(cfg["steps"])==tuple(r["step"] for r in d["training"])==tuple(r["step"] for r in d["states"])
        assert (cfg["lr"], cfg["lr_end"], cfg["weight_decay"]) == (.001, 1e-6, 1e-5)
        assert d["config_hash"]==config_hash(cfg)
        if parent==out:
            fitted[name]=d
        for row,state in zip(d["training"],d["states"]):
            assert row["step"]==state["step"]
            assert sha(parent/row["checkpoint"])==row["sha256"]
            assert sha(parent/state["path"])==state["sha256"]
            if parent==out and not state["borrowed"]:
                verify_state(out,cfg,state,parent/row["checkpoint"])
            assert isfinite(row["train_mse"])
            if row["step"]:
                assert all(isfinite(row[k]) for k in ("loss","grad","postclip","clip_frequency"))
            checkpoints[str((parent/row["checkpoint"]).relative_to(ROOT))]=row["sha256"]
            checkpoints[str((parent/state["path"]).relative_to(ROOT))]=state["sha256"]
        if cfg["shared"] and parent==out:
            base_name=cfg["name"]
            if base_name.startswith("outer_"):
                base_name=base_name.rsplit("_match2",1)[0]
            else:
                base_name=base_name.split("_")[0]
            plain=all_models[base_name]
            saved=[r for r in plain["states"] if r["step"]==1000][0]
            assert saved["sha256"]==cfg["shared"]
            assert cfg["first"]==1000 and cfg["prefix_schedule"]==plain["config"]["schedule"]==cfg["schedule"]
            for field in ("training", "states"):
                assert [(r["step"], r["sha256"]) for r in d[field] if r["step"]<=1000] == [
                    (r["step"], r["sha256"]) for r in plain[field] if r["step"]<=1000]
            assert {k:v for k,v in cfg["model"].items() if k!="weights"}=={
                k:v for k,v in plain["config"]["model"].items() if k!="weights"}
    n_outer=sum(k.startswith("outer_") for k in fitted)
    core=len(fitted)-n_outer
    assert n_outer==15 and core==3+3*int(f["selection"]["gradient_supplement"])+len(f["selection"]["extensions"])
    assert len(fitted)<=33
    for name in FACTORIAL:
        expected={"match2":f["anchor"]} if name.endswith("_match2") else {}
        assert f["models"][name]["config"]["model"]["weights"]==expected
    lineage=read_json(out/"outer_lineage.json")
    assert lineage["freeze_sha256"]==sha(out/"FROZEN.json")
    plan = read_json(out/"outer_plan.json")
    assert set(lineage["regimes"]) == set(plan) == {"interpolation", "upper", "lower"}
    for region, meta in lineage["regimes"].items():
        assert len(meta["refs"])==5 and set(meta["candidates"])==set(FACTORIAL)
        held=set(meta["train_removed"])
        active = set(plan[region]["active_files"])
        assert held==set(plan[region]["held_files"])==set(meta["backbone_seen"])
        assert active.isdisjoint(held) and meta["phys_re"]==plan[region]["re"]
        for ref in meta["refs"]:
            trained, scored = (set(ref[key]) for key in ("train_files", "held_files"))
            assert trained==active-scored and scored.isdisjoint(trained)
            assert {f for f, _ in ref["target_windows"]}==scored
            assert scored==held if ref["fold"]==4 else scored<=active
            assert sha(ROOT/ref["verified_checkpoint"])==ref["final_sha256"]
        for name,candidate in meta["candidates"].items():
            assert candidate["target_invariance"] and candidate["final"]["step"]==6000
            assert candidate["config"]["model"]==f["models"][name]["config"]["model"]
            assert candidate["config"]["scale"]==meta["scale"] and candidate["config"]["inputs"]["development"] is None
    return checkpoints,dict(new_main=core,new_outer=n_outer,new_total=len(fitted),reused_main=len(f["models"])-core,
                            reused_outer=len(outer)-n_outer,reused_teachers=20)


def finish(study, figures, freeze, outer):
    out=study.out
    assert guard(out)==freeze
    equal_regimes(out)
    plot(out,figures,freeze)
    report(out,freeze)
    evidence=audit_scores(out,freeze)
    checkpoints,cost=audit_models(out,freeze,outer)
    evidence["zero_output_pairs"]=audit_initialization({**freeze["models"], **outer})
    evidence["calibrated_initial_gradient_ratios"]=audit_gradients(out,freeze)
    basis = torch.load(ROOT / "results/corrector_incremental_structure/temporal_basis.pt", map_location="cpu", weights_only=True)
    evidence["loss_math"]=math_audit(basis)
    images={}
    for path in sorted(figures.glob("*.png")):
        with Image.open(path) as im:
            assert im.format=="PNG" and min(im.size)>300
            im.verify()
        images[path.name]=sha(path)
    assert len(images)==15
    files={str(p.relative_to(out)):sha(p) for p in out.rglob("*")
           if p.is_file() and p.suffix in (".md",".json",".csv",".gz",".pt") and p.name!="artifact_audit.json"}
    source={str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))}
    atomic_json(out/"artifact_audit.json",dict(passed=True,freeze_sha256=sha(out/"FROZEN.json"),files=files,
                                               source_files=source,figures=images,checkpoints=checkpoints,
                                               training=cost,evaluation=evidence))
    print("AUDIT COMPLETE",cost,evidence,flush=True)
