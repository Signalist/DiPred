"""Prepare causal data, cross-fit references, train the factorial, freeze, score."""
import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import torch

from reliability.backbone import configure
from research.corrector.run import dump,sha
from research.corrector_residual_optimization.fit import load as old_load, predict as old_predict, shared_plan, track_weights
from research.corrector_simple_baseline.data import RAW
from research.corrector_window_study.data import read_json,tensor_sha

from .data import OUT,CACHE,PRIOR,SIMPLE,FIGURES,Triplets,guard,write_tensor
from .fit import STEPS,fit,ledger
from .model import ARMS,base,encode,model,predict

REFERENCE_COMMIT = "e78f1160c63eafef9c8963b0a68ac9617ca125fb"
TEST_REF = "df3cf523cb205d9e31c8ebb4770e993d018ea992:results/corrector_simple_baseline/native_test.json"


def folds(cases):
    regimes = defaultdict(set)
    for c in cases:
        regimes[c["re"]].add(c["file"])
    groups = [[] for _ in range(4)]
    for re in sorted(regimes):
        order = sorted(range(4),key=lambda i:(len(groups[i]),i))
        for i,file in enumerate(sorted(regimes[re])):
            groups[order[i%4]].append(file)
    assert len({f for g in groups for f in g})==sum(map(len,groups))==46
    return groups


def subset(source,ix):
    from types import SimpleNamespace
    return SimpleNamespace(cases=[source.cases[i] for i in ix],x=source.x[ix],p=source.p[ix],y=source.y[ix])


def rms(data):
    mse = (data.y.double()-data.p.double()).square().mean((1,2,3,4))
    return float((mse*track_weights(data)).sum().sqrt())


def save_plan(out,name,sampling):
    file = out/f"schedule_{name}.pt"
    if file.exists():
        assert torch.equal(torch.load(file,map_location="cpu",weights_only=True),sampling)
    else:
        write_tensor(file,sampling)
    return dict(file=file.name,sha256=sha(file),tensor_sha256=tensor_sha(sampling))


def references():
    old = read_json(PRIOR/"FROZEN.json")
    choice = old["candidates"][old["best_base"]]
    assert choice["config"]==dict(width=32,lr=.001,weight_decay=1e-5,updates=6000,aux={},input="S3")
    ini = "checkpoints/budget_6000_0.pt"
    assert sha(PRIOR/ini)==old["training_and_diagnostic_weights"][ini]
    assert sha(PRIOR/choice["selected"]["checkpoint"])==choice["selected"]["sha256"]
    initial = {k.removeprefix("net.") if hasattr(k,"removeprefix") else k[4:]:v
               for k,v in torch.load(PRIOR/ini,map_location="cpu",weights_only=True).items()}
    return choice,ini,initial


def oof(train,prepared,norm,initial,out,cache):
    assignment = folds(train.cases)
    entries = []
    for i,files in enumerate(assignment):
        held = [j for j,c in enumerate(train.cases) if c["file"] in files]
        active = [j for j,c in enumerate(train.cases) if c["file"] not in files]
        name = f"oof_{i}"
        data = subset(train,active)
        scale = rms(data)
        schedule = shared_plan(data)
        sample = save_plan(out,name,schedule)
        summary = fit(name,data,prepared[active],None,None,scale,initial,schedule,out,cache)
        ledger(out)
        entry = dict(fold=i,holdout_files=sorted(files),holdout_indices=held,training_indices=active,
                     training_files=sorted({c["file"] for c in data.cases}),rms=scale,schedule=sample,
                     reference=summary["final"],training_count=len(active))
        entries.append(entry)
        dump(out/"crossfit_manifest.json",dict(seed=42,folds=entries))
    assert sorted(j for e in entries for j in e["holdout_indices"])==list(range(len(train.cases)))
    return entries


def checked_model(cfg,row,out,device):
    path = out/row["checkpoint"]
    assert sha(path)==row["sha256"]
    head = model(cfg,device)
    head.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    return head


@torch.inference_mode()
def feedback(data,norm,out,cache,kind,folds,reference,scale):
    spec = read_json(out/f"native_{kind}.json")
    manifest_file = out/f"feedback_{kind}.json"
    cache_file = cache/f"feedback_{kind}.pt"
    # The historical target is the newest OBSERVED block x, never y.
    true = data.x.double()-data.previous_p.double()
    source = dict(previous=spec["previous_P_sha256"],history=spec["previous_X_sha256"],
                  observed=tensor_sha(data.x),reference_sha256=[f["reference"]["sha256"] for f in folds] if folds else
                  [reference["selected"]["sha256"]],rms=[f["rms"] for f in folds] if folds else [scale])
    if manifest_file.exists() and cache_file.exists():
        metadata = read_json(manifest_file)
        assert metadata["sources"]==source and sha(cache_file)==metadata["sha256"]
        pred = torch.load(cache_file,map_location=data.x.device,weights_only=True)
    else:
        inp = base(data.previous_x,data.previous_p,norm)
        if folds:
            pred = torch.empty_like(data.x)
            for f in folds:
                ix = f["holdout_indices"]
                assert set(f["holdout_files"]).isdisjoint(f["training_files"])
                assert all(data.cases[j]["file"] in f["holdout_files"] for j in ix)
                head = checked_model(6,f["reference"],out,inp.device)
                pred[ix] = predict(head,inp[ix],f["rms"])
        else:
            head = old_load(reference["config"],reference["selected"],PRIOR,inp.device)
            pred = old_predict(head,inp,scale)
        write_tensor(cache_file,pred.cpu())
    remain = true-pred.double()
    assert torch.isfinite(remain).all()
    native = data.previous_p.double()
    observed = data.x.double()
    diff = float((observed-(native+pred.double())-remain).abs().max())
    restoration_error = float((native+true-observed).abs().max())
    assert restoration_error<1e-14 and diff<1e-14
    normalized = [(t/scale).float() for t in (true,pred,remain)]
    normal_error = float((normalized[0]-normalized[1]-normalized[2]).abs().max())
    assert normal_error<5e-6
    per_example = {}
    for name,values in (("true",true),("predicted",pred),("remaining",remain)):
        per_example[name] = dict(sha256=tensor_sha(values),rms=float(values.double().square().mean().sqrt()))
    dump(manifest_file,dict(sources=source,sha256=sha(cache_file),fields=per_example,
                            cases=[[c["file"],c["start"]-20] for c in data.cases],
                            fold=[next(f["fold"] for f in folds if j in f["holdout_indices"]) for j in range(len(data.cases))]
                            if folds else None,restoration_max_error=restoration_error,
                            closure_max_error=diff,normalized_max_error=normal_error))
    return true,pred,remain


def train_experiment(data):
    out,cache = data.out,data.cache
    assert not (out/"FROZEN.json").exists()
    (out/"checkpoints").mkdir(exist_ok=True)
    train,dev = (data.get(s) for s in ("train","development"))
    norm = data.source.norm
    reference,initial_path,initial = references()
    external_scale = read_json(PRIOR/"config.json")["scale"]
    scale = rms(train)
    current = [base(v.x,v.p,norm) for v in (train,dev)]
    schedule = shared_plan(train)
    schedule_rec = save_plan(out,"main",schedule)
    cfg = dict(seed=42,steps=STEPS,updates=6000,lr=.001,weight_decay=1e-5,batch=8,
               gradient_clip=1.,correction_strength=1.,layout="contiguous NCDHW, FP32",
               parameters=39810,primary_inputs=ARMS,rms=scale,external_rms=external_scale,
               normalization={k:v.flatten().tolist() for k,v in norm.items()},
               case=dict(file="26700_0.h5",start=400,lead=10),
               chain=dict(spacing=20,reset_on_gap=True,release_before_scoring=True))
    dump(out/"config.json",cfg)
    source = dict(prior_commit=REFERENCE_COMMIT,prior_freeze_sha256=sha(PRIOR/"FROZEN.json"),
                  external=reference,initial=dict(file=initial_path,sha256=sha(PRIOR/initial_path)),
                  backbone=data.source.identity,schedule=schedule_rec,protocol_sha256=sha(out/"PROTOCOL.md"))
    dump(out/"sources.json",source)
    crossfit = oof(train,current[0],norm,initial,out,cache)
    features = [feedback(v,norm,out,cache,s,crossfit if s=="train" else [],reference,
                         scale if s=="train" else external_scale)
                for v,s in ((train,"train"),(dev,"development"))]
    # Use the main train RMS for all normalized arm inputs (including development).
    def execute(name,slots,capacity=False):
        z = [encode(x,f,slots,scale,capacity=capacity) for x,f in zip(current,features)]
        for zz in z:
            for i,on in enumerate(slots):
                if not on:
                    assert not torch.count_nonzero(zz[:,6+2*i:8+2*i])
        value = fit(name,train,z[0],dev,z[1],scale,initial,schedule,out,cache)
        ledger(out)
        return value

    trials = {name:execute(name,slots) for name,slots in ARMS.items()}
    def best(names):
        return min(names,key=lambda n:(trials[n]["selected"]["dev_mse"],n))
    best_single,best_pair,best_feedback = (best(names) for names in ("BCD","EFG","BCDEFGH"))
    trials["Capacity"] = execute("Capacity",ARMS[best_feedback],capacity=True)
    assert len({str(v["initial_digest"]) for v in trials.values()})==1
    assert len({v["config"]["schedule_sha256"] for v in trials.values()})==1
    selections = dict(best_single=best_single,best_pair=best_pair,best_feedback=best_feedback,
                      best_factorial=best("ABCDEFGH"),capacity_matches=best_feedback,
                      rollout=list(dict.fromkeys((best_single,best_pair,"H"))),
                      models={k:{field:v[field] for field in ("selected","final","config","initial_digest","elapsed_seconds")}
                              for k,v in trials.items()})
    dump(out/"selections.json",selections)
    from .diagnostics import correlations, dependency
    correlations(train,features[0],out)
    dependency(dev,current[1],features[1],cfg,selections,out)
    dump(out/"feedback_cache_manifest.json",{part:read_json(out/f"feedback_{part}.json")
                                              for part in ("train","development")})
    files = ["triplet_manifest.json","crossfit_manifest.json","feedback_cache_manifest.json","config.json","sources.json",
             "selections.json","training.csv","input_dependency.csv","temporal_correlations.csv","native_train.json",
             "native_development.json","PROTOCOL.md","residual_correlations.csv","spatial_correlations.csv",
             *(x.name for x in sorted(out.glob("schedule_*.pt"))),*(x.name for x in sorted(out.glob("fit_*.json")))]
    blob = subprocess.check_output(["git","rev-parse",TEST_REF],cwd=PRIOR,text=True).strip()
    dump(out/"FROZEN.json",dict(bindings={k:sha(out/k) for k in files},prior_test_manifest_blob=blob,
                               arms=ARMS,reference=reference,selections=selections,config=cfg,
                               checkpoints={str(p.relative_to(out)):sha(p) for p in sorted((out/"checkpoints").glob("*.pt"))}))
    print("FROZEN BEFORE TEST",best_single,best_pair,best_feedback,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",default="all",choices=("all","train","score"))
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--cache",type=Path,default=CACHE)
    parser.add_argument("--raw",type=Path,default=RAW)
    parser.add_argument("--figures",type=Path,default=FIGURES)
    args=parser.parse_args()
    configure()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.out!=OUT and not (args.out/"PROTOCOL.md").exists():
        (args.out/"PROTOCOL.md").write_text((OUT/"PROTOCOL.md").read_text())
    data=Triplets(args.out,args.cache,args.raw)
    if args.action in ("all","train") and not (args.out/"FROZEN.json").exists():
        train_experiment(data)
    if args.action in ("all","score"):
        from .evaluate import score
        from .plot import plot
        score(data)
        plot(args.out,args.figures)


if __name__=="__main__":
    main()
