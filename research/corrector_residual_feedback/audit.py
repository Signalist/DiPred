"""Audit published factorial artifacts and reproduce the held-out primary scores."""
import argparse
import csv
import gzip
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
import torch

from reliability.backbone import configure
from research.corrector.run import dump,sha
from research.corrector_simple_baseline.measure import groups
from research.corrector_window_study.data import read_json,tensor_sha

from .data import CACHE,FIGURES,OUT,RAW,Triplets,guard
from .fit import STEPS,read,state_digest
from .model import ARMS,base,encode,predict
from .run import checked_model,feedback,references

KEYS = ("split","policy","arm","trajectory","start")


def audit(out,cache,figures,raw):
    frozen = guard(out)
    config,selected = frozen["config"],frozen["selections"]
    manifest = read_json(out/"triplet_manifest.json")
    crossed = read_json(out/"crossfit_manifest.json")
    feedbacks = read_json(out/"feedback_cache_manifest.json")
    source = read_json(out/"sources.json")
    verification = read_json(out/"verification.json")
    fsha = sha(out/"FROZEN.json")
    assert verification["passed"] and verification["freeze_sha256"]==fsha
    assert verification["restoration_max_error"]<1e-14 and verification["zero_exact"]
    assert set(frozen["arms"])==set(ARMS) and frozen["arms"]=={a:list(v) for a,v in ARMS.items()}
    assert config["seed"]==crossed["seed"]==42 and config["correction_strength"]==1.
    test_auth = read_json(out/"native_test.json")
    assert test_auth["test_freeze_sha256"]==fsha
    ref,initial_file,prior_init = references()
    assert ref==frozen["reference"]
    assert source["initial"]["file"]==initial_file

    for part in ("train","development","test"):
        cases = manifest[part]["cases"]
        keys = {(c["file"],c["start"]) for c in cases}
        assert len(keys)==len(cases)
        for c in cases:
            s = c["start"]
            assert c["earlier"]==[s-20,s] and c["history"]==[s,s+20] and c["target"]==[s+20,s+40]
            assert c["issue_frame"]==s+20 and c["score_frame"]==s+40
        ledger = read_json(out/f"native_{part}.json")
        recipe = ledger["added_recipe"]
        new = set(map(tuple,recipe["keys"]))
        assert recipe["zero_initial_noise"] and recipe["denoising_labels"]==[99,66,33,0]
        assert recipe["backbone"]==source["backbone"] and len(new)==ledger["new_previous"]
        assert not keys & new and ledger["reused_previous"]+len(new)==len(cases)
        if new:
            assert sha(cache/f"previous_native_{part}.pt")==ledger["generated_cache_sha256"]
        proof = feedbacks[part] if part!="test" else read_json(out/"feedback_test_cache_manifest.json")
        assert sha(cache/f"feedback_{part}.pt")==proof["sha256"]
        p = torch.load(cache/f"feedback_{part}.pt",map_location="cpu",weights_only=True)
        assert p.shape==(len(cases),2,20,32,64) and tensor_sha(p)==proof["fields"]["predicted"]["sha256"]
        assert proof["closure_max_error"]<1e-14 and proof["normalized_max_error"]<5e-6
        assert proof["restoration_max_error"]<1e-14

    all_held = []
    all_train_ids = {c["file"] for c in manifest["train"]["cases"]}
    assignment = feedbacks["train"]["fold"]
    folds = crossed["folds"]
    assert len(folds)==4 and len(assignment)==len(manifest["train"]["cases"])
    for f in folds:
        used,held = (set(f[k]) for k in ("training_files","holdout_files"))
        assert not used&held and used|held==all_train_ids
        ids = {j for j,c in enumerate(manifest["train"]["cases"]) if c["file"] in held}
        assert ids==set(f["holdout_indices"])=={j for j,v in enumerate(assignment) if v==f["fold"]}
        assert not ids&set(f["training_indices"]) and len(f["training_indices"])+len(ids)==len(assignment)
        assert f["reference"]["step"]==6000 and f["rms"]>0
        assert feedbacks["train"]["sources"]["reference_sha256"][f["fold"]]==f["reference"]["sha256"]
        assert sha(out/f["schedule"]["file"])==f["schedule"]["sha256"]
        schedule = torch.load(out/f["schedule"]["file"],map_location="cpu",weights_only=True)
        assert schedule.shape==(6000,8) and schedule.min()>=0 and schedule.max()<f["training_count"]
        assert tensor_sha(schedule)==f["reference"]["schedule_sha256"]
        all_held.extend(held)
    assert len(all_held)==len(set(all_held))==46

    training = read(out/"training.csv")
    crosses = {f"oof_{i}" for i in range(4)}
    methods = {*ARMS,"Capacity"}
    assert Counter(r["run"] for r in training)==dict.fromkeys(crosses|methods,len(STEPS))
    assert len(frozen["checkpoints"])==len(training)==len(STEPS)*(len(crosses)+len(methods))
    initial = None
    train_native = read_json(out/"native_train.json")
    controls = [selected["models"][name]["config"] for name in sorted(methods)]
    assert {r["train_P"] for r in controls}=={train_native["matched_P_sha256"]}
    assert len({r["train_Y"] for r in controls})==1
    assert {r["scale"] for r in controls}=={config["rms"]}
    assert all(r["seed"]==42 and r["updates"]==6000 and r["channels"]==12 for r in controls)
    assert sha(out/source["schedule"]["file"])==source["schedule"]["sha256"]
    main_schedule = torch.load(out/source["schedule"]["file"],map_location="cpu",weights_only=True)
    assert main_schedule.shape==(6000,8) and tensor_sha(main_schedule)==source["schedule"]["tensor_sha256"]
    for entry in training:
        arm,step,path = entry["run"],int(entry["step"]),entry["checkpoint"]
        assert sha(out/path)==entry["sha256"]==frozen["checkpoints"][path]
        weights = torch.load(out/path,map_location="cpu",weights_only=True)
        assert all(torch.isfinite(v).all() for v in weights.values())
        if step:
            assert np.isfinite([float(entry["loss"]),float(entry["gradient_norm"])]).all()
        for region in ("train",) if arm in crosses else ("train","dev"):
            b,m,c,e,g = (float(entry[f"{region}_{k}"]) for k in ("native","mse","C","E","G"))
            assert abs(b-m-g)<1e-17 and abs(2*c-e-g)<1e-17
        if arm in methods:
            assert entry["schedule_sha256"]==source["schedule"]["tensor_sha256"]
            assert sum(v.numel() for v in weights.values())==39810
        if step==0:
            assert torch.equal(weights["0.weight"][:,:6],prior_init["0.weight"])
            assert all(torch.equal(v,weights[k]) for k,v in prior_init.items() if k!="0.weight")
            assert not torch.count_nonzero(weights["4.weight"]) and not torch.count_nonzero(weights["4.bias"])
            if arm in methods:
                digest = state_digest(weights)
                initial = digest if initial is None else initial
                assert digest==initial==selected["models"][arm]["initial_digest"]
    for (name,),values in groups(training,("run",)).items():
        assert [int(r["step"]) for r in values]==list(STEPS)
        if name in methods:
            nonzero = [r for r in values if float(r["dev_E"])>0]
            candidate = min(nonzero,key=lambda r:(float(r["dev_mse"]),int(r["step"])))
            assert candidate["sha256"]==selected["models"][name]["selected"]["sha256"]
            assert values[-1]["sha256"]==selected["models"][name]["final"]["sha256"]
    for field,candidates in (("best_single","BCD"),("best_pair","EFG"),("best_feedback","BCDEFGH")):
        expected = min(candidates,key=lambda a:(selected["models"][a]["selected"]["dev_mse"],a))
        assert expected==selected[field]

    windows = read(out/"windows.csv")
    keyed = {tuple(r[k] for k in KEYS):r for r in windows}
    assert len(keyed)==len(windows)==verification["window_rows"]
    columns = ("native_mse","mse","C","correction_energy","G")
    aggregates = {}
    count = 0
    with gzip.open(out/"lead_windows.csv.gz","rt",newline="") as file:
        for row in csv.DictReader(file):
            key = tuple(row[k] for k in KEYS)
            sums,seen = aggregates.setdefault(key,([0.]*len(columns),set()))
            lead = int(row["lead"])
            assert lead not in seen
            seen.add(lead)
            for j,k in enumerate(columns):
                sums[j] += float(row[k])
            count += 1
    assert count==verification["lead_rows"]==20*len(windows)
    for key,row in keyed.items():
        sums,seen = aggregates[key]
        assert seen==set(range(1,21))
        assert all(abs(v/20-float(row[k]))<1e-17 for v,k in zip(sums,columns))
        p,q,c,e,g = [float(row[k]) for k in columns]
        assert abs(p-q-g)<1e-17 and abs(2*c-e-g)<1e-17
    rows = groups(windows,("split","policy","arm"))
    def by_track(own,field):
        tracks = groups(own,("trajectory",))
        return np.array([mean(float(r[field]) for r in tracks[k]) for k in sorted(tracks)])
    contrasts = read(out/"contrasts.csv")
    assert {(r["split"],r["policy"]) for r in contrasts
            if r["arm"]==selected["capacity_matches"] and r["control"]=="Capacity"}=={
                (split,policy) for split in ("train","development","test") for policy in ("selected","final")}
    for r in contrasts:
        key = (r["split"],r["policy"])
        a,b = (by_track(rows[(*key,r[k])],"mse") for k in ("arm","control"))
        n = len(a)
        draws = np.random.default_rng(42).choice(n,size=(10000,n))
        interval = np.percentile(100*(1-a[draws].mean(1)/b[draws].mean(1)),[2.5,97.5])
        np.testing.assert_allclose([100*(1-a.mean()/b.mean()),*interval],
                                   [float(r[k]) for k in ("gain_pct","lower_pct","upper_pct")],atol=1e-12)
    factors = read(out/"factorial_effects.csv")
    effects = dict(true=(0,),predicted=(1,),remaining=(2,),true_predicted=(0,1),
                   true_remaining=(0,2),predicted_remaining=(1,2),three_way=(0,1,2))
    for r in factors:
        key = (r["split"],r["policy"])
        contrast = sum(np.prod([2*mask[j]-1 for j in effects[r["effect"]]])*by_track(rows[(*key,arm)],"G")/4
                       for arm,mask in ARMS.items())
        draws = np.random.default_rng(42).choice(len(contrast),size=(10000,len(contrast)))
        bounds = np.percentile(contrast[draws].mean(1),[2.5,97.5])
        np.testing.assert_allclose([contrast.mean(),*bounds],
                                   [float(r[k]) for k in ("G_effect","lower","upper")],rtol=0,atol=1e-17)
    for row in read(out/"input_dependency.csv"):
        if abs(int(row["previous_lead"])-int(row["current_lead"]))>3:
            assert float(row["response_rms"])==0.
    assert len(read(out/"temporal_correlations.csv"))==1200
    rollout_rows = read(out/"feedback_rollout.csv")
    for r in rollout_rows:
        assert int(r["score_frame"])-int(r["issue_frame"])==20
        if r["reset"]=="True":
            assert float(r["fixed_reference_mse"])==float(r["self_mse"])
    rollout_groups = groups(rollout_rows,("split","arm"))
    for row in read(out/"feedback_rollout_summary.csv"):
        own = rollout_groups[row["split"],row["arm"]]
        if row["chain_window"]!="all":
            own = [r for r in own if r["chain_window"]==row["chain_window"]]
        fixed,self_score = (by_track(own,k) for k in ("fixed_reference_mse","self_mse"))
        assert len(own)==int(row["windows"]) and len(fixed)==int(row["trajectories"])
        assert abs(100*(1-self_score.mean()/fixed.mean())-float(row["self_gain_pct"]))<1e-12
        if row["chain_window"]=="1":
            assert np.array_equal(fixed,self_score)
    pngs = {}
    for p in sorted(figures.glob("*.png")):
        with Image.open(p) as image:
            assert image.format=="PNG"
            pngs[p.name]=list(image.size)
            image.verify()
    assert {k[:2] for k in pngs}=={f"{i:02d}" for i in range(1,14)}
    dataset = Triplets(out,cache,raw)
    test = dataset.get("test")
    norm = dataset.source.norm
    previous = feedback(test,norm,out,cache,"test",[],ref,config["external_rms"])
    current = base(test.x,test.p,norm)
    point,truth = [v.double().cpu().numpy() for v in (test.p,test.y)]
    independent = {}
    for name in (*ARMS,"Capacity"):
        mask = ARMS[selected["capacity_matches"]] if name=="Capacity" else ARMS[name]
        z = encode(current,previous,mask,config["rms"],capacity=name=="Capacity")
        assert all(torch.count_nonzero(z[:,6+2*i:8+2*i])==0 for i,on in enumerate(mask) if not on)
        assert torch.equal(z[:,:6],current)
        for policy in ("selected","final"):
            net = checked_model(12,selected["models"][name][policy],out,test.p.device)
            prediction = predict(net,z,config["rms"]).double().cpu().numpy()
            error = np.mean((point+prediction-truth)**2,axis=(1,2,3,4))
            expected = [float(keyed["test",policy,name,c["file"],str(c["start"])]["mse"]) for c in test.cases]
            np.testing.assert_allclose(error,expected,rtol=0,atol=1e-16)
            independent[f"{name}:{policy}"] = float(np.max(np.abs(error-expected)))
    dump(out/"artifact_audit.json",dict(passed=True,freeze_sha256=fsha,figures=pngs,
                                        checkpoints=len(training),windows=len(windows),leads=count,
                                        contrasts=len(contrasts),factorial_effects=len(factors),
                                        shared_P_targets_and_schedule=True,zero_slots_verified=True,
                                        independent_numpy_mse_max_error=independent))



if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--cache",type=Path,default=CACHE)
    parser.add_argument("--figures",type=Path,default=FIGURES)
    parser.add_argument("--raw",type=Path,default=RAW)
    args=parser.parse_args()
    configure()
    audit(args.out,args.cache,args.figures,args.raw)
