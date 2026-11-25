"""Validate frozen data, optimizer provenance, numerical results and deliverables."""
import csv
import gzip
import json
from math import isfinite
from statistics import mean

import numpy as np
from PIL import Image
import torch

from research.corrector.run import dump,sha
from research.corrector_consistency.model import ARMS,loss as original
from research.corrector_residual_feedback.data import guard
from research.corrector_residual_feedback.fit import state_digest
from research.corrector_residual_optimization.fit import track_weights
from research.corrector_simple_baseline.measure import groups
from research.corrector_window_study.data import read_json,tensor_sha

from .data import BRANCHES
from .diagnostics import calibration
from .fit import START,STEPS,config_hash
from .loss import NAMES,extensions,objective
from .search import ORIGINAL


def check_objectives(labels,scale):
    """Compare original implementations and gradients on real target patches."""
    rng = torch.Generator().manual_seed(42)
    y = labels[:3,:,::4,::4,::4].detach().cpu()
    r = (torch.randn(y.shape,generator=rng)*.1+y*.5).requires_grad_()
    errors = {}
    for arm,terms in ARMS.items():
        proposed = objective(r,y,{name:.1 for name in terms})
        expected = original(r,y,arm)
        a, = torch.autograd.grad(proposed,r)
        b, = torch.autograd.grad(expected,r)
        e = dict(value=float((proposed-expected).detach().abs()),gradient=float((a-b).abs().max()))
        assert e["value"]<=3e-7 and e["gradient"]<=3e-8,arm
        errors[arm] = e
    assert torch.equal(objective(r,y,{}),(r-y).square().mean())
    value = extensions(r,y,("match2","relation"))
    assert torch.allclose(value["match2"],(r.diff(n=2,dim=2)-y.diff(n=2,dim=2)).square().mean())
    assert extensions(y,y,("match2","relation"))["match2"]==0
    assert extensions(y,y,("match2","relation"))["relation"]==0
    gradients = torch.autograd.grad(value["match2"]+value["relation"],r)[0]
    assert torch.isfinite(gradients).all()
    lead1_left = r[:,:,0]-y[:,:,1]
    lead1_right = y[:,:,0]-y[:,:,1]
    boundary = (lead1_left-lead1_right).square().mean()
    assert torch.allclose(boundary,(r[:,:,0]-y[:,:,0]).square().mean())
    reconstructed = ((r*scale-y*scale).square().mean()/scale**2)
    assert torch.allclose(reconstructed,objective(r,y,{}))
    return errors


def audit_training(study,frozen,train):
    out = study.out
    models = frozen["selections"]["models"]
    cfg = read_json(out/"config.json")
    decisions = read_json(out/"stage_decisions.json")
    assert cfg["seed"]==frozen["seed"]==42 and cfg["scale"]==study.scale
    actual_rms = ((train.y.double()-train.p.double()).square().mean((1,2,3,4))*track_weights(train)).sum().sqrt()
    assert abs(float(actual_rms)-study.scale)<1e-14
    checked = {}
    shared = {}
    for b in BRANCHES:
        path = out/f"shared_{b}.pt"
        st = torch.load(path,map_location="cpu",weights_only=False)
        shared[b] = st
        assert st["step"]==st["scheduler"]["last_epoch"]==START
        assert st["branch"]==b and st["schedule_sha256"]==tensor_sha(study.schedule)
        assert st["config_hash"]==models[f"{b}_MSE"]["config_hash"]
        assert set(st["rng"])=={"python","numpy","torch","cuda"}
        optim = st["optimizer"]
        assert len(optim["state"])==6
        assert all(int(v["step"])==START for v in optim["state"].values())
        assert all(torch.isfinite(v[t]).all() for v in optim["state"].values() for t in ("exp_avg","exp_avg_sq"))
        assert optim["param_groups"][0]["weight_decay"]==1e-5
        assert optim["param_groups"][0]["lr"]==st["scheduler"]["_last_lr"][0]
        row = st["rows"][-1]
        assert sha(out/row["checkpoint"])==row["sha256"]
        assert state_digest(st["model"])==state_digest(torch.load(out/row["checkpoint"],
                                                                 map_location="cpu",weights_only=True))
    extension = sum(run["config"]["stage"]=="extension" for run in models.values())
    assert extension==decisions["extension_fits"]<=cfg["extension_budget"]
    assert len(models)-extension==decisions["core_fits"]<=cfg["core_budget"]
    ledger = {}
    for name,summary in models.items():
        record = read_json(out/f"run_{name}.json")
        assert all(record[k]==v for k,v in summary.items())
        job = record["config"]
        b = job["branch"]
        weights = job["weights"]
        assert all(0<v<=100 for v in weights.values())
        assert job["seed"]==42 and job["updates"]==6000 and job["scale"]==study.scale
        assert job["schedule_sha256"]==tensor_sha(study.schedule)
        assert job["previous_train_input"]==study.old_runs[b]["config"]["train_input"]
        assert job["previous_dev_input"]==study.old_runs[b]["config"]["dev_input"]
        assert record["config_hash"]==config_hash(job)
        rows = record["training"]
        assert [r["step"] for r in rows]==list(STEPS)
        assert record["final"]==rows[-1]
        assert record["selected"]==min((r for r in rows if r["step"] and r["dev_E"]>0),
                                      key=lambda r:(r["dev_mse"],r["step"]))
        if weights:
            assert job["first"]==START and job["shared_sha256"]==sha(out/f"shared_{b}.pt")
            expected = [dict(r,run=name,phase="shared_MSE") for r in shared[b]["rows"]]
            assert rows[:len(expected)]==expected
        else:
            assert job["first"]==0 and job["arm"]=="MSE"
        for r in rows:
            assert r["run"]==name
            assert r["sha256"]==frozen["checkpoints"][r["checkpoint"]]
            assert max(r["train_identity_error"],r["dev_identity_error"])<1e-15
            if r["step"]:
                assert all(isfinite(r[k]) for k in ("loss","batch_mse","gradient_norm","gradient_after_clip"))
                assert 0<=r["gradient_after_clip"]<=1
            ledger[name,r["step"]] = r
        checked[name] = dict(config_hash=record["config_hash"],shared_sha256=job["shared_sha256"],
                             finite_checkpoint_intervals=len(rows)-1)
    with (out/"training.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows)==len(ledger)
    for r in rows:
        ref = ledger[r["run"],int(r["step"])]
        assert all(r[k]==("" if v is None else str(v)) for k,v in ref.items())
    assert all(models[frozen["selections"]["fixed"][b]["JOINT"]]["config"]["weights"]==ORIGINAL for b in BRANCHES)
    return checked


def audit_search(study,frozen):
    """Verify every mandatory arm and independently derive the frozen selections."""
    out = study.out
    choices = frozen["selections"]
    models = choices["models"]

    def pick(names):
        return min(names,key=lambda n:(models[n]["selected"]["dev_mse"],n))

    def table_rows(name):
        with (out/name).open() as f:
            return list(csv.DictReader(f))

    def weights(name):
        return models[name]["config"]["weights"]

    main = table_rows("loss_search.csv")
    joint = table_rows("joint_search.csv")
    gradients = table_rows("gradient_diagnostics.csv")
    weighted = table_rows("weighted_gradients.csv")
    exploration = read_json(out/"exploration_results.json")
    assert all(isfinite(float(row[key])) for row in gradients for key in
               ("raw_loss","gradient_l2","ratio_to_mse","cosine_to_mse","fixed_total_postclip"))
    assert all(float(r["postclip"])<=1 and abs(float(r["auxiliary_cosine"]))<=1.000001 for r in weighted)
    assert all(models[a]["config"]["branch"]==r["branch"] for r in (*main,*joint) for a in (r["run"],))
    for row in (*main,*joint):
        actual = models[row["run"]]["selected"]
        assert json.loads(row["weights"])==weights(row["run"])
        assert int(row["step"])==actual["step"] and float(row["dev_mse"])==actual["dev_mse"]
        assert float(row["gain_native_pct"])==100*(1-actual["dev_mse"]/actual["dev_native"])
    alias = {r["run"] for r in (*main,*joint)}
    assert alias | {"B_MSE","E_MSE"}==set(models)
    assert {(r["run"],int(r["step"]),int(r["batch"]),r["loss"]) for r in weighted}=={
        (n,step,batch,loss) for n in models if weights(n) for step in (2000,4000,6000)
        for batch in (1,2,3) for loss in ("mse",*weights(n))}
    branches = {}
    for branch in BRANCHES:
        plain = f"{branch}_MSE"
        fixed = choices["fixed"][branch]
        assert fixed["MSE"]==plain and not weights(plain)
        assert set(fixed)=={"MSE","JOINT",*NAMES}
        assert weights(fixed["JOINT"])==ORIGINAL
        branch_rows = [r for r in main if r["branch"]==branch]
        g = groups([r for r in gradients if r["run"]==plain and r["step"]==str(START)],("loss",))
        g_norm = {key:mean(float(r["gradient_l2"]) for r in rows) for (key,),rows in g.items()}
        assert all(len(rows)==3 for rows in g.values())
        counts = {}
        for label,loss in NAMES.items():
            assert weights(fixed[label])=={loss:.1}
            raw = {float(r["coordinate"]):r["run"] for r in branch_rows if r["loss"]==loss
                   and r["stage"] in ("raw","raw_boundary")}
            assert {0.,.01,.1,1.}<=raw.keys()<= {0.,.001,.01,.1,1.,10.}
            assert len(raw)<=5 and raw[0.]==plain and raw[.1]==fixed[label]
            assert all(weights(n)==({loss:v} if v else {}) for v,n in raw.items())
            calibrated = {float(r["coordinate"]):r["run"] for r in branch_rows
                          if r["loss"]==loss and r["stage"]=="balanced"}
            for ratio in (.1,.3):
                expected,_,status = calibration(g_norm[loss],g_norm["mse"],ratio)
                if status=="ok":
                    assert weights(calibrated[ratio])=={loss:expected}
                else:
                    assert ratio not in calibrated
            r = choices["single"][branch][loss]
            assert r["raw"]==pick(raw.values())
            assert r["balanced"]==pick([plain,*calibrated.values()])
            assert r["best"]==pick([r["raw"],r["balanced"]])
            assert r["nonzero"]==pick(n for n in (*raw.values(),*calibrated.values()) if n!=plain)
            assert r["weight"]==weights(r["best"]).get(loss,0.)
            counts[loss] = len(raw)+len(calibrated)
        names = choices["joint"][branch]
        assert set(names)=={"original","tuned","balanced","top2","best"}
        assert names["original"]==fixed["JOINT"]
        assert names["best"]==pick(v for k,v in names.items() if k!="best")
        tuned = {k:v["weight"] for k,v in choices["single"][branch].items() if v["weight"]}
        assert weights(names["tuned"])==tuned
        candidates = choices["single"][branch]
        two = sorted(candidates,key=lambda k:(models[candidates[k]["nonzero"]]["selected"]["dev_mse"],k))[:2]
        assert weights(names["top2"])=={k:weights(candidates[k]["nonzero"])[k] for k in two}
        base = tuned if len(tuned)>=2 else weights(names["top2"])
        assert set(weights(names["balanced"]))<=set(base)
        assert choices["core"][branch]==pick([plain,*fixed.values(),
                                              *(r["best"] for r in candidates.values()),*names.values()])
        ext = [r["run"] for r in exploration if r["branch"]==branch and r["executed"]]
        assert choices["best"][branch]==pick([choices["core"][branch],*ext])
        assert {r["loss"] for r in exploration if r["branch"]==branch}>={"relation","match2"}
        branches[branch] = counts
    ext = [r["run"] for r in exploration if r["executed"]]
    assert choices["best_extension"]==(pick(ext) if ext else None)
    winner = pick(choices["best"].values())
    s3 = study.prior["reference"]["selected"]["dev_mse"]
    assert choices["global_best"]==("S3" if s3<=models[winner]["selected"]["dev_mse"] else winner)
    return dict(search_rows=len(main),joint_rows=len(joint),gradient_rows=len(gradients),
                weighted_gradient_rows=len(weighted),weights_by_branch=branches)


def audit_scores(out,frozen):
    verification = read_json(out/"verification.json")
    assert verification["passed"] and verification["freeze_sha256"]==sha(out/"FROZEN.json")
    assert read_json(out/"native_test.json")["test_freeze_sha256"]==verification["freeze_sha256"]
    choices = frozen["selections"]
    nominees = {"Native","S3","B_MSE","E_MSE",*choices["best"].values(),
                *(choices["joint"][b]["best"] for b in BRANCHES),
                *(v["best"] for b in BRANCHES for v in choices["single"][b].values())}
    if choices["best_extension"]:
        nominees.add(choices["best_extension"])
    assert set(verification["future_mutation_bitwise"])==nominees
    assert {(r["split"],r["arm"]) for r in verification["numpy_recomputations"]}=={
        (s,n) for s in ("development","test") for n in nominees}
    assert all(r["absolute_error"]<1e-15 for r in verification["numpy_recomputations"])
    assert len(verification["historical_baselines"])==16
    assert all(r["absolute_error"]<1e-12 for r in verification["historical_baselines"])
    assert verification["zero_exact"] and verification["source_weights_unchanged"]
    assert verification["restoration_max_error"]<1e-14 and verification["score_max_identity_error"]<1e-15
    keys = ("split","policy","arm")
    summary = {}
    with (out/"scores.csv").open() as f:
        for r in csv.DictReader(f):
            summary[tuple(r[k] for k in keys)] = r
    with (out/"leads.csv").open() as f:
        lead = groups(list(csv.DictReader(f)),keys)
    with (out/"channels.csv").open() as f:
        channel = groups(list(csv.DictReader(f)),keys)
    with (out/"trajectories.csv").open() as f:
        trajectory = groups(list(csv.DictReader(f)),keys)
    with (out/"regimes.csv").open() as f:
        regimes = groups(list(csv.DictReader(f)),keys)
    with (out/"windows.csv").open() as f:
        windows = {tuple(r[k] for k in (*keys,"trajectory","start")):r for r in csv.DictReader(f)}
    for key,r in summary.items():
        assert sorted(int(v["lead"]) for v in lead[key])==list(range(1,21))
        assert sorted(v["channel"] for v in channel[key])==["u","v"]
        assert len(trajectory[key])==12
        for rows in (lead[key],channel[key],trajectory[key]):
            for metric in ("mse","native_mse","C","correction_energy","G"):
                assert abs(mean(float(v[metric]) for v in rows)-float(r[metric]))<1e-15
                for v in rows:
                    assert abs(2*float(v["C"])-float(v["correction_energy"])-float(v["G"]))<1e-15
        for v in regimes[key]:
            cases = [t for t in trajectory[key] if t["re"]==v["re"]]
            assert abs(mean(float(t["mse"]) for t in cases)-float(v["mse"]))<1e-15
    lead_sum = {}
    with gzip.open(out/"lead_windows.csv.gz","rt") as f:
        for r in csv.DictReader(f):
            key = tuple(r[k] for k in (*keys,"trajectory","start"))
            a = lead_sum.setdefault(key,{})
            t = int(r["lead"])
            assert t not in a
            a[t] = float(r["mse"])
            assert (not r["first_difference_match"])==(t==1)
            assert (not r["second_difference_match"])==(t<=2)
    assert windows.keys()==lead_sum.keys()
    for k,part in lead_sum.items():
        assert sorted(part)==list(range(1,21))
        assert abs(mean(part.values())-float(windows[k]["mse"]))<1e-15
    assert len(windows)==verification["window_rows"] and len(lead_sum)*20==verification["lead_rows"]
    assert len(summary)==2*2*(2+len(choices["models"]))
    with (out/"contrasts.csv").open() as f:
        resample = np.random.default_rng(42).integers(12,size=(10000,12))
        for r in csv.DictReader(f):
            key = tuple(r[k] for k in keys)
            a,b = (np.array([float(t["mse"]) for t in sorted(trajectory[h],key=lambda v:v["trajectory"])])
                   for h in (key,(*key[:2],r["control"])))
            v = summary[key]
            assert int(r["improved"])==int((a<b).sum())
            assert abs(float(r["gain_pct"])-100*(1-float(v["mse"])/b.mean()))<1e-12
            low,high = np.quantile(100*(1-a[resample].mean(1)/b[resample].mean(1)),[.025,.975])
            assert abs(low-float(r["lower_pct"]))<1e-12 and abs(high-float(r["upper_pct"]))<1e-12
    return dict(scores=len(summary),windows=len(windows),lead_windows=20*len(lead_sum),
                numpy_candidates=len(nominees),paired_samples=10000)


def audit(study,figures):
    out = study.out
    frozen = guard(out)
    assert read_json(out/"selections.json")==frozen["selections"]
    assert all(sha(out/p)==digest for p,digest in frozen["checkpoints"].items())
    train,_,_ = study.get("train")
    dev,_,_ = study.get("development")
    assert len(train.cases)==822 and len(dev.cases)==24
    assert len({c["file"] for c in train.cases})==46 and len({c["file"] for c in dev.cases})==12
    assert read_json(out/"sources.json")["input_identity"]==study.guards
    for data in (train,dev):
        assert all(c["earlier"]==[c["start"]-20,c["start"]]
                   and c["issue_frame"]==c["start"]+20 and c["score_frame"]==c["start"]+40 for c in data.cases)
    assert sha(out/"native_train.json")==frozen["bindings"]["native_train.json"]
    assert sha(out/"native_development.json")==frozen["bindings"]["native_development.json"]
    checked = audit_training(study,frozen,train)
    formulas = check_objectives((train.y[:3]-train.p[:3])/study.scale,study.scale)
    search = audit_search(study,frozen)
    counts = audit_scores(out,frozen)
    hashes = {}
    for p in sorted(figures.glob("*.png")):
        with Image.open(p) as img:
            assert img.format=="PNG" and min(img.size)>=300
            img.verify()
        hashes[p.name] = sha(p)
    assert sorted(int(k.split("_")[0]) for k in hashes)==list(range(1,16))
    assert guard(out)==frozen
    audit_outputs = ("scores.csv","leads.csv","lead_windows.csv.gz","windows.csv","trajectories.csv","channels.csv",
                     "regimes.csv","contrasts.csv","verification.json","fixed_fields.csv","inference_cost.csv")
    dump(out/"artifact_audit.json",dict(passed=True,freeze_sha256=sha(out/"FROZEN.json"),
                                       training=checked,original_loss_errors=formulas,counts=counts,search=search,
                                       frozen_checkpoint_count=len(frozen["checkpoints"]),figure_sha256=hashes,
                                       scored_sha256={k:sha(out/k) for k in audit_outputs}))
    print("AUDITED",len(checked),"fits and",len(hashes),"figures",flush=True)
