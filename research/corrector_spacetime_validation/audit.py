"""Audit published identities, times, physical invariants and holdout isolation."""
import csv
from math import isfinite
from pathlib import Path
from statistics import mean

from PIL import Image

from research.corrector.run import sha
from research.corrector_residual_feedback.data import guard
from research.corrector_simple_baseline.measure import groups
from research.corrector_window_study.data import read_json

from .data import PREVIOUS,ROOT
from .fit import atomic_json,checkpoint_file,config_hash
from .objectives import verify_math


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def audit_training(out,frozen):
    fits = [read_json(p) for p in sorted((out/"fits").glob("*.json"))]
    outer_count = sum(r["config"]["name"].startswith("outer_") for r in fits)
    assert outer_count==39 and len(fits)-outer_count==frozen["core_new_fits"]<=32
    assert len(fits)<=80
    prefixes = {"B":PREVIOUS/"shared_B.pt","E":PREVIOUS/"shared_E.pt","S3":out/"shared_S3.pt"}
    prefixes.update((p.stem[7:],p) for p in out.glob("shared_outer_*.pt"))
    by_hash = {sha(p):(name,p) for name,p in prefixes.items()}
    all_checkpoints = dict(frozen["global_checkpoints"])
    for run in fits:
        cfg = run["config"]
        assert run["config_hash"]==config_hash(cfg)
        assert cfg["seed"]==42 and cfg["updates"]==6000
        assert cfg["weight_decay"]==1e-5 and cfg["lr"]==.001 and cfg["lr_end"]==1e-6
        assert cfg["clip"]==1 and cfg["batch"]==8
        assert [r["step"] for r in run["training"]]==cfg["steps"]
        assert run["final"]["step"]==6000
        for row in run["training"]:
            assert row["train_identity_error"]<1e-15
            assert row["step"]==0 or all(isfinite(row[k]) for k in ("train_mse","loss","gradient_norm"))
            prefix = by_hash[cfg["shared"]][1] if cfg["shared"] is not None else None
            checkpoint = checkpoint_file(out,row,prefix)
            all_checkpoints[str(checkpoint.relative_to(ROOT))] = row["sha256"]
    assert all(sha(ROOT/p)==digest for p,digest in all_checkpoints.items())
    import torch
    from research.corrector_residual_feedback.fit import state_digest
    states = {}
    for digest,(name,p) in by_hash.items():
        st = torch.load(p,map_location="cpu",weights_only=False)
        assert all(int(v["step"])==1000 for v in st["optimizer"]["state"].values())
        assert st["step"]==st["scheduler"]["last_epoch"]==1000
        assert set(st["rng"])=={"python","numpy","torch","cuda"}
        folder = PREVIOUS if name in ("B","E") else out
        checkpoint = folder/st["rows"][-1]["checkpoint"]
        assert sha(checkpoint)==st["rows"][-1]["sha256"]
        assert state_digest(torch.load(checkpoint,map_location="cpu",weights_only=True))==state_digest(st["model"])
        states[digest] = st
    for r in fits:
        cfg = r["config"]
        if cfg["weights"]:
            k = states[cfg["shared"]]
            assert cfg["first"]==1000 and cfg["schedule"]==k["schedule_sha256"]
            assert r["training"][:len(k["rows"])]==[dict(x,run=cfg["name"],phase="shared_MSE") for x in k["rows"]]
        else:
            assert cfg["first"]==0
    return all_checkpoints,dict(core=len(fits)-outer_count,holdout=outer_count,total=len(fits),
                                shared_states=len(states),seconds=sum(r["elapsed_seconds"] for r in fits))


def audit_outer(out,freeze):
    lineage = read_json(out/"reference_lineage.json")
    manifests = read_json(out/"outer_split_manifest.json")
    assert lineage["freeze_sha256"]==sha(out/"FROZEN.json")
    assert set(lineage["regimes"])=={"interpolation","upper","lower"}
    for name,item in lineage["regimes"].items():
        active,held = (set(item[k]) for k in ("train_files","held_files"))
        assert active.isdisjoint(held)
        assert held==set(manifests[name]["held_files"])
        assert held<=set(lineage["backbone"]["training_files"])
        assert len(item["references"])==5 and len(item["candidates"])==8
        assigned,issues = set(),set()
        for r,plan in zip(item["references"],manifests[name]["reference_plan"]):
            assert r["train_files"]==plan["training_files"] and r["held_files"]==plan["prediction_files"]
            assert r["schedule"]["tensor_sha256"]==plan["sampling_sha256"]
            r_active,r_held = (set(r[k]) for k in ("train_files","held_files"))
            assert r_active.isdisjoint(held|r_held)
            assert {f for f,t in r["prediction_ids"]}==r_held
            meta = read_json(out/"fits"/f"{r['run']}.json")
            assert meta["config"]["origin"]["files"]==r["train_files"]
            assert r["scale"]==meta["config"]["scale"]
            assert r["final_sha256"]==meta["final"]["sha256"]
            assert sha(out/r["schedule"]["file"])==r["schedule"]["sha256"]
            if r["fold"]<4:
                assert assigned.isdisjoint(r_held) and r_active==active-r_held
                assigned |= r_held
                assert issues.isdisjoint(map(tuple,r["prediction_ids"]))
                issues.update(map(tuple,r["prediction_ids"]))
            else:
                assert r_held==held and r_active==active
        assert assigned==active
        for tag,r in item["candidates"].items():
            assert r["frozen_from"]==freeze["selection"]["outer"][tag]
            assert r["weights"]==freeze["models"][r["frozen_from"]]["config"]["weights"]
            assert r["step"]==6000 and r["scale"]==item["train_scale"] and r["future_perturbation_bitwise"]
            assert sha(out/r["checkpoint"])==r["sha256"]


def audit_scoring(out,freeze):
    verify = read_json(out/"verification.json")
    digest = sha(out/"FROZEN.json")
    assert verify["passed"] and verify["freeze_sha256"]==digest
    assert read_json(out/"native_test.json")["test_freeze_sha256"]==digest
    assert all(x["bitwise"] for x in verify["future_invariance"])
    assert {x["model"] for x in verify["future_invariance"]}==set(freeze["models"])
    assert verify["identity_max_error"]<1e-15
    historical = verify["historical_reproduced"]
    assert len(historical)==40 and max(v["absolute_error"] for v in historical)<1e-12
    assert max(x["absolute_error"] for x in verify["independent_recomputation"])<1e-15
    feedback = read(out/"feedback_rollout.csv")
    for _,rows in groups(feedback,("model","file")).items():
        rows = sorted(rows,key=lambda r:int(r["start"]))
        assert 2<=len(rows)<=10 and rows[0]["prior_source"]=="fixed_reset"
        assert all(r["perturbation_unchanged"]=="True" for r in rows)
        for a,b in zip(rows,rows[1:]):
            assert int(b["start"])-int(a["start"])==20 and a["score_frame"]==b["issue_frame"]
            assert b["prior_source"]=="published" and a["issued_sha256"]==b["state_sha256"]
    evidence = read_json(out/"period_provenance.json")
    assert evidence["freeze_sha256"]==digest
    assert len({(r["file"],r["start"]) for r in evidence["releases"]})==len(evidence["releases"])
    assert evidence["planned_cache_binding"]==freeze["bindings"]["period_plan.json"]
    assert verify["feedback_windows"]==len(feedback)==4*verify["period_windows"]
    rows = read(out/"scores.csv")
    leads = groups(read(out/"leads.csv"),("split","policy","arm"))
    for r in rows:
        k=(r["split"],r["policy"],r["arm"])
        assert sorted(int(x["lead"]) for x in leads[k])==list(range(1,21))
        m,C,U,G,b=(float(r[x]) for x in ("mse","C","correction_energy","G","native_mse"))
        assert abs(mean(float(v["mse"]) for v in leads[k])-m)<1e-15
        assert abs(G-(b-m))<1e-15 and abs(G-(2*C-U))<1e-15
    return dict(scores=len(rows),feedback_rows=len(feedback),recomputed=len(verify["independent_recomputation"]))


def audit(study,figures):
    out = study.out
    f = guard(out)
    assert read_json(out/"math_verification.json")==verify_math()
    assert read_json(out/"selections.json")==f["selection"]
    a=read_json(out/"reuse_audit.json")
    assert all(sha(ROOT/p)==d for p,d in a["historical_evidence"].items())
    for name in ("train","development","test"):
        identity=read_json(out/f"native_{name}.json")
        prior=read_json(PREVIOUS/f"native_{name}.json")
        assert {k:v for k,v in identity.items() if k!="test_freeze_sha256"}=={
                k:v for k,v in prior.items() if k!="test_freeze_sha256"}
    checks,costs = audit_training(out,f)
    audit_outer(out,f)
    scores = audit_scoring(out,f)
    fig_hashes = {}
    for p in sorted(figures.glob("*.png")):
        with Image.open(p) as im:
            assert im.format=="PNG" and min(im.size)>=300
            im.verify()
        fig_hashes[p.name] = sha(p)
    assert len(fig_hashes)==13
    names = ("PROTOCOL.md","hypotheses.json","experiment_plan.json","data_roles.json","sources.json","reuse_audit.json",
             "training.csv","mechanism_search.csv","gradient_diagnostics.csv","gradient_clipping.csv",
             "temporal_mode_scores.csv","selection_sensitivity.csv","outer_split_manifest.json","reference_lineage.json",
             "holdout_results.csv","feedback_rollout.csv","scores.csv","leads.csv","trajectories.csv","regimes.csv",
             "channels.csv","contrasts.csv","lead_primary_contrasts.csv","selections.json","FROZEN.json","exploration_log.md","verification.json",
             "DECISION_REPORT.md","joint_interaction.csv","period_provenance.json","period_plan.json")
    verified = {k:sha(out/k) for k in names}
    atomic_json(out/"artifact_audit.json",dict(passed=True,freeze_sha256=sha(out/"FROZEN.json"),
                                               training=costs,evaluation=scores,files=verified,
                                               checkpoints=checks,figures=fig_hashes))
    print("AUDITED",costs,scores,flush=True)
