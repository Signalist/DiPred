"""Independent artifact checks for the sealed and scored experiment."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
import torch

from reliability.backbone import configure
from research.corrector.run import dump, sha
from research.corrector_consistency.model import ARMS, loss
from research.corrector_simple_baseline.measure import groups
from research.corrector_window_study.data import read_json, tensor_sha

from .data import CACHE, OUT, RAW, SIMPLE, Data, split as load_split
from .fit import load, predict, read
from .model import CHANNELS, LOSSES, components, inputs


def audit(out,cache,raw):
    frozen = read_json(out/"FROZEN.json")
    decision = read_json(out/"stage_decisions.json")
    selected = read_json(out/"selections.json")
    verify = read_json(out/"verification.json")
    assert sha(out/"FROZEN.json")==verify["frozen_sha256"]
    assert frozen["candidates"]==selected["runs"] and frozen["decision"]==decision
    for file,digest in frozen["bindings"].items():
        assert sha(out/file)==digest
    cfg = read_json(out/"config.json")
    assert frozen["seed"]==cfg["seed"]==42 and cfg["correction_strength"]==1
    assert decision["stage1_new_fits"]==4 and decision["stage2_new_fits"]<=6
    assert decision["stage3_holdout_fits"]==2
    assert decision["stage4_enabled"] == (decision["passed_development_gate"] and decision["passed_migration_gate"])
    assert bool(decision["stage4_new_fits"])==decision["stage4_enabled"]
    assert verify["native_s0_max_mse_error"]<2e-13 and verify["prediction_unchanged_by_future"]
    for file,digest in frozen["training_and_diagnostic_weights"].items():
        assert sha(out/file)==digest
        parameters = torch.load(out/file,map_location="cpu",weights_only=True)
        assert all(v.isfinite().all() for v in parameters.values())
    schedule = torch.load(out/"sampling.pt",map_location="cpu",weights_only=True)
    assert schedule.shape==(6000,8)
    assert schedule.min()>=0 and schedule.max()<822
    records = groups(read(out/"training.csv"),("run",))
    for name,run in selected["runs"].items():
        rows = records[(name,)]
        sub = min((r for r in rows if float(r["dev_E"])>0),key=lambda r:(float(r["dev_mse"]),int(r["step"])))
        assert sub["sha256"]==run["selected"]["sha256"]
        assert all(r["schedule_sha256"]==tensor_sha(schedule[:run["config"]["updates"]]) for r in rows)
        for r in rows:
            p = Path(r["external_checkpoint"]) if r["external_checkpoint"] else out/r["checkpoint"]
            assert sha(p)==r["sha256"]
            for split in ("train","dev"):
                b,m,c,e,g = (float(r[f"{split}_{key}"]) for key in ("native","mse","C","E","G"))
                assert abs(b-m-g)<1e-17 and abs(2*c-e-g)<1e-17
            if int(r["step"]):
                assert np.isfinite([float(r["gradient_norm"]),float(r["batch_loss"])]).all()
    source = {}
    for part in ("train","development","test"):
        old,new = (read_json(p/f"native_{part}.json") for p in (SIMPLE,out))
        assert {k:v for k,v in old.items() if k!="test_freeze_sha256"}=={
            k:v for k,v in new.items() if k!="test_freeze_sha256"}
        file = cache/f"{part}.pt"
        assert sha(file)==new["file_sha256"]
        assert new["test_freeze_sha256"]==(verify["frozen_sha256"] if part=="test" else None)
        ids = {r["file"] for r in new["identity"]["cases"]}
        assert len(ids)=={"train":46,"development":12,"test":12}[part]
        source[part] = ids
    assert len(set.union(*source.values()))==70
    train_manifest = read_json(out/"native_train.json")["identity"]["cases"]
    holdout_leads = groups(read(out/"holdout_leads.csv"),("holdout",))
    holdouts = read_json(out/"holdouts.json")
    assert {h["kind"]:h["re"] for h in holdouts}==cfg["holdout_re"]
    assert read_json(out/"pre_holdout_selection.json")["model"]==frozen["best_base"]
    for holdout in holdouts:
        kind = holdout["kind"]
        fit,held = (set(holdout[k]) for k in ("training_files","heldout_files"))
        expected = [case for case in train_manifest if case["re"]==holdout["re"]]
        assert not fit&held and fit|held==source["train"]
        assert held=={case["file"] for case in expected} and holdout["windows"]==len(expected)
        assert holdout["fixed_step"]==frozen["candidates"][frozen["best_base"]]["selected"]["step"]
        held_schedule = torch.load(out/f"holdout_{kind}_schedule.pt",map_location="cpu",weights_only=True)
        assert held_schedule.shape==(holdout["fixed_step"],8)
        assert int(held_schedule.min())>=0 and int(held_schedule.max())<holdout["subtrain_windows"]
        assert tensor_sha(held_schedule)==holdout["checkpoint"]["schedule_sha256"]
        per_lead = holdout_leads[(kind,)]
        assert {int(r["lead"]) for r in per_lead}==set(range(1,21))
        for metric in ("mse","native"):
            assert abs(mean(float(r[metric]) for r in per_lead)-holdout["metrics"][metric])<1e-17
    improvement = decision["development"]
    dev_pass = (decision["dev_gain_pct"]>=cfg["gate"]["gain_pct"]
                and improvement["improved"]>=cfg["gate"]["min_improved"]
                and improvement["concentration"] is not None
                and improvement["concentration"]<=cfg["gate"]["max_concentration"])
    migration_pass = all(h["improvement_pct"]>0 and 2*h["improved"]>h["trajectories"] for h in holdouts)
    assert dev_pass==decision["passed_development_gate"]
    assert migration_pass==decision["passed_migration_gate"]
    dep = read(out/"input_dependency.csv")
    assert Counter(r["run"] for r in dep)==dict.fromkeys(CHANNELS,400)
    for r in dep:
        if r["run"]=="S0" and abs(int(r["history_frame"])-int(r["lead"]))>3:
            assert float(r["response_rms"])==0.
    grad = read(out/"gradient_diagnostics.csv")
    assert Counter(r["loss"] for r in grad)==dict.fromkeys(("mse",*LOSSES),3)
    assert all(np.isfinite([float(r[k]) for k in ("raw_loss","gradient_l2","cosine_to_mse",
                                                  "layer_1","layer_2","layer_3")]).all() for r in grad)
    rng = torch.Generator().manual_seed(42)
    p,t = (torch.randn((2,2,5,6,7),generator=rng).requires_grad_() for _ in range(2))
    for name,aux in ARMS.items():
        terms = components(p,t,aux)
        value = terms["mse"]+sum(.1*terms[k] for k in aux)
        ref = loss(p,t,name)
        torch.testing.assert_close(value,ref)
        torch.testing.assert_close(torch.autograd.grad(value,p)[0],torch.autograd.grad(ref,p)[0])
    windows = read(out/"windows.csv")
    leads = read(out/"lead_windows.csv")
    key = ("split","run","trajectory","start")
    assert len(groups(windows,key))==len(windows)==verify["windows"]
    assert len(groups(leads,(*key,"lead")))==len(leads)==20*len(windows)
    by_window = groups(leads,key)
    for row in windows:
        actual = by_window[tuple(row[k] for k in key)]
        assert set(int(r["lead"]) for r in actual)==set(range(1,21))
        for k in ("mse","native_mse","C","E","G"):
            assert abs(mean(float(r[k]) for r in actual)-float(row[k]))<1e-17
        b,m,c,e,g = (float(row[k]) for k in ("native_mse","mse","C","E","G"))
        assert abs(b-m-g)<1e-17 and abs(2*c-e-g)<1e-17
    by_run = groups(windows,("split","run"))
    def track_values(rows):
        grouped = groups(rows,("trajectory",))
        return np.array([mean(float(r["mse"]) for r in grouped[k]) for k in sorted(grouped)])
    comparisons = read(out/"contrasts.csv")
    for r in comparisons:
        a,b = (track_values(by_run[r["split"],r[k]]) for k in ("run","control"))
        n = len(a)
        index = np.random.default_rng(42).choice(n,size=(10000,n))
        ci = np.percentile(100*(1-a[index].mean(1)/b[index].mean(1)),[2.5,97.5])
        np.testing.assert_allclose([100*(1-a.mean()/b.mean()),*ci],
                                   [float(r[k]) for k in ("improvement_pct","ci_low_pct","ci_high_pct")],atol=1e-12)
    assert verify["passed"]
    data = Data(raw,cache,out)
    test = load_split(data,"test")
    onscreen = verify["shown_in_development_fields"]
    independent_errors = {}
    for name in dict.fromkeys((frozen["best_base"],onscreen)):
        config = selected["runs"][name]["config"]
        chosen = selected["runs"][name]["selected"]
        net = load(config,chosen,out,test.p.device)
        r = predict(net,inputs(test,data.norm,config["input"]),cfg["scale"]).double().cpu().numpy()
        point, truth = (x.double().cpu().numpy() for x in (test.p,test.y))
        calc = np.mean((point+r-truth)**2,axis=(1,2,3,4))
        ledger = {(row["trajectory"],int(row["start"])):float(row["mse"])
                  for row in by_run["test",name]}
        result = np.array([ledger[case["file"],case["start"]] for case in test.cases])
        np.testing.assert_allclose(calc,result,rtol=0,atol=1e-16)
        independent_errors[name] = float(np.max(np.abs(calc-result)))
    pngs = {}
    for path in sorted(out.glob("*.png")):
        with Image.open(path) as plot:
            assert plot.format=="PNG"
            pngs[path.name] = list(plot.size)
            plot.verify()
    expected = {f"{i:02d}" for i in (*range(1,10),12)}
    if decision["stage4_enabled"]:
        expected |= {"10","11"}
    assert {k[:2] for k in pngs}==expected
    dump(out/"artifact_audit.json",dict(passed=True,phase_gate_consistent=True,loss_definitions_match_previous=True,
                                        checked_checkpoints=len(frozen["training_and_diagnostic_weights"]),
                                        reproduced_checkpoints=len(records[("S0",)]),paired_contrasts=len(comparisons),
                                        window_rows=len(windows),lead_window_rows=len(leads),figures=pngs,
                                        independent_numpy_test_window_mse_max_error=independent_errors))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--cache",type=Path,default=CACHE)
    parser.add_argument("--raw",type=Path,default=RAW)
    args = parser.parse_args()
    configure()
    audit(args.out,args.cache,args.raw)
