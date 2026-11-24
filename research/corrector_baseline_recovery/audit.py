"""Verify released artifacts independently of model selection or training."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
import torch

from research.corrector.run import dump, sha
from research.corrector_predictability.data import OUT as PREVIOUS
from research.corrector_window_study.data import read_json

from .data import CACHE
from .fit import KINDS, SEEDS, STEPS
from .legacy import OUT, old_sources


def table(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def audit(out, cache):
    old_sources()
    cfg = read_json(out/"config.json")
    frozen = read_json(out/"FROZEN.json")
    check = read_json(out/"verification.json")
    assert check["passed"] and check["frozen_sha256"]==sha(out/"FROZEN.json")
    sources = read_json(out/"sources.json")
    for key, file in (("protocol","PROTOCOL.md"),("legacy_replay","legacy_replay.json"),
                      ("expansion","expansion.json"),("diagnostic_audit","diagnostic_audit.json")):
        assert sources[f"{key}_sha256"]==sha(out/file), file
    for key, file in (("config","config.json"),("training","training.csv"),("sources","sources.json")):
        assert frozen[f"{key}_sha256"]==sha(out/file), file
    data = read_json(out/"expansion.json")
    assert data["legacy_manifest_sha256"]==sha(out/"legacy_windows_819.json")
    cases = {(c["file"],c["start"]) for c in data["cases"]}
    assert len(cases)==len(data["cases"])==cfg["train_windows"]["expanded"]
    assert all(r["target"]==[r["start"]+20,r["start"]+40] for r in data["cases"])
    train, dev, test = (set(sources["splits"][k]["files"]) for k in ("train","development","test"))
    assert not (train & dev or train & test or dev & test)
    assert {f for f,_ in cases}==train
    old = {(c["file"],c["start"]) for c in read_json(out/"legacy_windows_819.json")}
    assert old <= cases
    for r in data["batches"]:
        if r["batch_backbone_calls"]:
            path = cache/f'train_{r["start"]}_mean.pt'
            assert sha(path)==r["cache_sha256"] and path.stat().st_size==r["cache_bytes"]
            assert r["seeds"]==list(range(500000,500032)) and r["sign"]==1
    training = table(out/"training.csv")
    expected = {(m,s,step) for m in KINDS for s in SEEDS for step in (STEPS[:7] if m=="R0" else STEPS)}
    assert len(training)==len(expected)==frozen["checkpoints"]==129
    schedules = defaultdict(set)
    initial, row_by_fit = {}, {}
    for row in training:
        mode,seed,step = row["mode"],int(row["seed"]),int(row["step"])
        expected.remove((mode,seed,step))
        row_by_fit[mode,seed,step] = row
        path = (PREVIOUS if mode=="R0" else out)/row["checkpoint"]
        assert sha(path)==row["sha256"]
        tensors = torch.load(path,map_location="cpu",weights_only=True)
        assert sum(t.numel() for t in tensors.values())==cfg["parameter_counts"][KINDS[mode]]
        assert all(torch.isfinite(t).all() for t in tensors.values())
        if step==0:
            initial[mode,seed] = tensors
        schedules[mode,seed].add(row["schedule_sha256"])
    assert not expected
    for mode in KINDS:
        for seed in SEEDS:
            for policy, picked in frozen["selected"][mode][str(seed)].items():
                row = row_by_fit[mode,seed,picked["step"]]
                assert picked["sha256"]==row["sha256"] and picked["checkpoint"]==row["checkpoint"]
                assert 0<=picked["alpha"]<=1
                if policy.startswith("step1000"): assert picked["step"]==1000
                if policy.startswith("final"): assert picked["step"]==(1000 if mode=="R0" else 2000)
            candidates = []
            for key, row in row_by_fit.items():
                if key[:2] != (mode,seed):
                    continue
                step = int(row["step"])
                for score,alpha in (("dev_fitted_mse",float(row["dev_alpha"])),
                                    ("dev_mse",1.),("dev_native_mse",0.)):
                    candidates.append((float(row[score]),step,alpha))
            expected_mse,expected_step,expected_alpha = min(candidates)
            picked = frozen["selected"][mode][str(seed)]["selected"]
            assert picked["step"]==expected_step
            assert abs(picked["alpha"]-expected_alpha)<1e-8 and abs(picked["dev_mse"]-expected_mse)<2e-11
    assert all(len(v)==1 for v in schedules.values())
    for seed in SEEDS:
        for a,b in (("R1","R2"),("R3","R4")):
            assert schedules[a,seed]==schedules[b,seed]
        for a,b in (("R0","R1"),("R1","R3"),("R2","R4")):
            u,v = initial[a,seed],initial[b,seed]
            assert u.keys()==v.keys() and all(torch.equal(u[k],v[k]) for k in u)
    window = table(out/"windows.csv")
    leads = table(out/"lead_windows.csv")
    keys = ("mode","policy","seed","trajectory","start")
    assert len(window)==len({tuple(r[k] for k in keys) for r in window})==1824
    assert len(leads)==len({tuple(r[k] for k in (*keys,"lead")) for r in leads})==36480
    lead_by_window = defaultdict(list)
    for r in leads:
        lead_by_window[tuple(r[k] for k in keys)].append(r)
    for r in window:
        own = lead_by_window[tuple(r[k] for k in keys)]
        assert {int(v["lead"]) for v in own}==set(range(1,21))
        for field in ("native_mse","mse","C","E","G"):
            assert abs(mean(float(v[field]) for v in own)-float(r[field]))<1e-17
    by_seed = defaultdict(list)
    for r in window:
        c,e,g,p,q = (float(r[k]) for k in ("C","E","G","native_mse","mse"))
        assert abs(2*c-e-g)<1e-17 and abs(p-q-g)<1e-17
        by_seed[tuple(r[k] for k in keys[:-1])].append(q)
    per_track = defaultdict(list)
    for (mode,policy,seed,track), mses in by_seed.items():
        assert len(mses)==2
        per_track[mode,policy,track].append(mean(mses))
    names = sorted(test)
    baseline = np.array([mean(per_track["Native","zero",k]) for k in names])
    idx = np.random.default_rng(42).choice(len(names),size=(10000,len(names)))
    scores = table(out/"scores.csv")
    for row in scores:
        mode,policy = row["mode"],row["policy"]
        data = np.array([mean(per_track[mode,policy,k]) for k in names])
        low,high = np.quantile(100*(1-data[idx].sum(1)/baseline[idx].sum(1)),(.025,.975))
        np.testing.assert_allclose([float(row[k]) for k in ("correction_pct","low95_pct","high95_pct")],
                                   [100*(1-data.sum()/baseline.sum()),low,high],atol=1e-12)
    assert len(scores)==26
    shapes = {}
    for path in sorted(out.glob("*.png")):
        with Image.open(path) as im:
            assert im.format=="PNG"
            shapes[path.name] = list(im.size)
            im.verify()
    assert len(shapes)==3
    dump(out/"artifact_audit.json",dict(passed=True,verified_new_checkpoints=108,replayed_r0_checkpoints=21,
                                        matched_schedules=True,identical_initializations=True,
                                        legacy_source_verified=True,expansion_sources_verified=True,
                                        paired_intervals_recomputed=len(scores),test_windows=len(window),
                                        test_leads=len(leads),figures=shapes))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--cache",type=Path,default=CACHE)
    args = parser.parse_args()
    audit(args.out,args.cache)
