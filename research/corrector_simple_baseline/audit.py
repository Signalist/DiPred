"""Independently check artifact identities, trajectory reductions and field scores."""
import argparse
import csv
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
import torch

from reliability.backbone import configure
from research.corrector.run import dump, sha
from research.corrector_window_study.data import read_json, tensor_sha

from .data import CACHE, OUT, RAW, WORKSPACE, Data
from .fit import STEPS
from .measure import groups
from .model import fields, network, physical_residual


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def audit(out,cache,raw):
    frozen = read_json(out/"FROZEN.json")
    freeze_sha = sha(out/"FROZEN.json")
    verification = read_json(out/"verification.json")
    assert verification["passed"] and freeze_sha==verification["frozen_sha256"]
    for name,digest in frozen["sources_sha256"].items():
        assert sha(out/name)==digest
    cfg = read_json(out/"config.json")
    proof = read_json(out/"sources.json")
    assert proof["experiment_protocol_sha256"]==sha(out/"PROTOCOL.md")
    assert cfg["correction_strength"]==1 and cfg["seed"]==42
    training = read(out/"training.csv")
    assert tuple(int(r["step"]) for r in training)==STEPS
    assert len({r["schedule_sha256"] for r in training})==1
    assert float(training[0]["lr"])==3e-4 and float(training[-1]["lr"])==1e-6
    for r in training:
        path = out/r["checkpoint"]
        assert sha(path)==r["sha256"]
        w = torch.load(path,map_location="cpu",weights_only=True)
        assert sum(v.numel() for v in w.values())==3042 and all(torch.isfinite(v).all() for v in w.values())
        for split in ("train","development"):
            c,e,g,m,b = (float(r[f"{split}_{k}"]) for k in ("C","E","G","mse","native_mse"))
            assert abs(2*c-e-g)<1e-17 and abs(b-m-g)<1e-17
    choose = lambda r:(float(r["development_mse"]),int(r["step"]))
    selected = frozen["checkpoints"]
    for k,r in (("selected",min(training,key=choose)),
                ("best_nonzero",min((r for r in training if float(r["development_E"])>0),key=choose)),
                ("native",training[0]),("final",training[-1])):
        assert selected[k]["sha256"]==r["sha256"] and selected[k]["step"]==int(r["step"])
    case_sets = {}
    for name,n in (("train",822),("development",24),("test",24)):
        item = read_json(out/f"native_{name}.json")
        assert item["test_freeze_sha256"]==(freeze_sha if name=="test" else None)
        cases = item["identity"]["cases"]
        ids = {(r["file"],r["start"]) for r in cases}
        assert len(cases)==len(ids)==n
        assert all(c["history"]==[c["start"],c["start"]+20]
                   and c["target"]==[c["start"]+20,c["start"]+40] for c in cases)
        reused = set(map(tuple,item["reused_windows"]))
        new = set(map(tuple,item["generated_windows"]))
        assert new | reused == ids and not new & reused
        assert set(map(tuple,item["replay_verified"])) <= reused
        assert item["identity"]["denoising_labels"]==[99,66,33,0]
        for path,digest in item["legacy_source_hashes"].items():
            assert sha(WORKSPACE/path)==digest
        path = cache/f"{name}.pt"
        assert sha(path)==item["file_sha256"] and path.stat().st_size==item["cache_bytes"]
        p = torch.load(path,map_location="cpu",weights_only=True)
        assert p.shape==(n,2,20,32,64) and tensor_sha(p)==item["P_sha256"]
        case_sets[name] = {r["file"] for r in cases}
    assert len(set.union(*case_sets.values()))==sum(map(len,case_sets.values()))==70
    assert all(sorted(case_sets[k])==v for k,v in proof["splits"].items())
    window,lead = (read(out/f"{f}.csv") for f in ("windows","lead_windows"))
    key = ("split","policy","step","trajectory","start")
    assert len(window)==len(groups(window,key))==3480
    assert len(lead)==len(groups(lead,(*key,"lead")))==69600
    leading = groups(lead,key)
    for row in window:
        rows = leading[tuple(row[k] for k in key)]
        assert {int(r["lead"]) for r in rows}==set(range(1,21))
        for name in ("mse","native_mse","C","E","G"):
            assert abs(mean(float(r[name]) for r in rows)-float(row[name]))<1e-17
        p,q,c,e,g = (float(row[k]) for k in ("native_mse","mse","C","E","G"))
        assert abs(p-q-g)<1e-17 and abs(2*c-e-g)<1e-17
    idx = np.random.default_rng(42).choice(12,size=(10000,12))
    by_key = groups(window,("split","policy"))
    for row in read(out/"scores.csv"):
        own = groups(by_key[row["split"],row["policy"]],("trajectory",))
        proposed = np.array([mean(float(r["mse"]) for r in own[t]) for t in sorted(own)])
        baseline = np.array([mean(float(r["native_mse"]) for r in own[t]) for t in sorted(own)])
        gain = 100*(1-proposed.mean()/baseline.mean())
        assert abs(gain-float(row["correction_pct"]))<1e-12
        assert int(row["improved_trajectories"])==int((proposed<baseline).sum())
        if row["split"]=="test":
            expected = np.quantile(100*(1-proposed[idx].mean(1)/baseline[idx].mean(1)),(.025,.975))
            np.testing.assert_allclose(expected,[float(row["ci95_low_pct"]),float(row["ci95_high_pct"])],atol=1e-12)
    data = Data(raw,cache,out)
    # The frozen audit path reads the already validated cache and does not run the backbone again.
    split = data.split("test")
    model = network(split.p.device)
    model.load_state_dict(torch.load(out/selected["selected"]["checkpoint"],map_location=split.p.device,weights_only=True))
    r = physical_residual(model,fields(split,data.norm),cfg["physical_residual_rms"]).double().cpu().numpy()
    p,y = (v.double().cpu().numpy() for v in (split.p,split.y))
    independent = np.mean((p+r-y)**2,axis=(1,2,3,4))
    expected = [float(v["mse"]) for v in by_key["test","selected"]]
    np.testing.assert_allclose(independent,expected,rtol=0,atol=1e-16)
    fields_csv = read(out/"fixed_development_field.csv")
    assert len(fields_csv)==2*32*64
    for row in fields_csv:
        assert abs(float(row["truth"])-float(row["prediction"])-float(row["remaining"]))<1e-8
    figures = {}
    for p in sorted(out.glob("*.png")):
        with Image.open(p) as image:
            assert image.format=="PNG"
            figures[p.name] = list(image.size)
            image.verify()
    assert len(figures)==5
    dump(out/"artifact_audit.json",dict(passed=True,test_freeze_binding_verified=True,
                                        checkpoints=len(training),trajectory_splits={k:len(v) for k,v in case_sets.items()},
                                        windows=len(window),lead_windows=len(lead),test_intervals_recomputed=4,
                                        independent_numpy_test_mse_max_error=float(np.max(np.abs(independent-expected))),
                                        figures=figures))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--cache",type=Path,default=CACHE)
    parser.add_argument("--raw",type=Path,default=RAW)
    args = parser.parse_args()
    configure()
    audit(args.out,args.cache,args.raw)
