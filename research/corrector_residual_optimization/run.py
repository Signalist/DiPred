"""Complete development-only stages, freeze configurations, then score."""
import argparse
import subprocess
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import torch

from reliability.backbone import configure
from research.corrector.run import dump, sha
from research.corrector_simple_baseline.fit import SEED, table
from research.corrector_window_study.data import read_json, save_tensor, tensor_sha

from .data import CACHE, OUT, RAW, SIMPLE, Data, split, subset
from .diagnose import analytic, dependency, residual_invariants
from .fit import DEFAULT, Experiment, checks, load, plan, predict, shared_plan, track_weights
from .model import CHANNELS, inputs, network


def assessment(r,data):
    benefit = (data.y.double()-data.p.double()).square().mean((1,2,3,4))
    benefit -= (data.y.double()-(data.p.double()+r.double())).square().mean((1,2,3,4))
    tracks = {}
    from research.corrector_predictability.alpha import trajectory_indices
    for k,ix in trajectory_indices(data.cases).items():
        tracks[k] = float(benefit[ix].mean())
    positive = [v for v in tracks.values() if v>0]
    return dict(improved=len(positive),trajectories=len(tracks),
                concentration=max(positive)/sum(positive) if positive else None,trajectory_G=tracks)


def holdout(run,train,out):
    conf,picked = run["config"],run["selected"]
    results, lead_rows = [], []
    for kind,re in (("interpolation",17812.),("extrapolation",25482.)):
        ix = [i for i,r in enumerate(train.cases) if r["re"]==re]
        it = [i for i,r in enumerate(train.cases) if r["re"]!=re]
        own,held = (subset(train,sel) for sel in (it,ix))
        assert {r["file"] for r in own.cases}.isdisjoint(r["file"] for r in held.cases)
        wd = track_weights(own)
        weight = wd.view(-1,1,1,1,1)
        mu = (own.x.double()*weight).sum(0).mean((1,2,3)).view(1,2,1,1,1)
        var = ((own.x.double()-mu).square()*weight).sum(0).mean((1,2,3)).view_as(mu)
        norm = {"mean":mu.float(),"std":var.sqrt().float()}
        scale = float((((own.y.double()-own.p.double()).square().mean((1,2,3,4)))*wd).sum().sqrt())
        schedule = shared_plan(own)
        save_tensor(out/f"holdout_{kind}_schedule.pt",schedule[:picked["step"]])
        exp = Experiment(own,None,norm,scale,out,schedule,namespace=f"holdout_{kind}_")
        exp.fit(f"H_{kind}",conf,"holdout",stop=picked["step"],record_steps=(0,picked["step"]))
        selected = exp.runs[f"H_{kind}"]["selected"]
        net = load(conf,selected,out,held.p.device)
        r = predict(net,inputs(held,norm,conf["input"]),scale)
        metrics,gap = checks(r,held,track_weights(held))
        status = assessment(r,held)
        result = dict(kind=kind,re=re,windows=len(ix),subtrain_windows=len(it),
                      fixed_step=picked["step"],checkpoint=selected,metrics=metrics,**status,
                      normalization={k:v.flatten().tolist() for k,v in norm.items()},scale=scale,
                      heldout_files=sorted({c["file"] for c in held.cases}),
                      training_files=sorted({c["file"] for c in own.cases}),
                      improvement_pct=100*metrics["G"]/metrics["native"],decomposition_error=gap)
        results.append(result)
        weights = track_weights(held)[:,None]
        base = (held.y.double()-held.p.double()).square().mean((1,3,4))
        mse = (held.y.double()-held.p.double()-r.double()).square().mean((1,3,4))
        for t,(a,b) in enumerate(zip((mse*weights).sum(0),(base*weights).sum(0))):
            lead_rows.append(dict(holdout=kind,re=re,lead=t+1,mse=float(a),native=float(b),gain_pct=float(100*(1-a/b))))
    dump(out/"holdouts.json",results)
    table(out/"holdout_leads.csv",lead_rows)
    return results


def small_fits(run,train,norm,scale,out):
    started = perf_counter()
    result = []
    for size in (4,16):
        own = subset(train,list(range(size)))
        # These probes fit the same fixed windows repeatedly via replacement.
        schedule = shared_plan(own)
        exp = Experiment(own,None,norm,scale,out,schedule,namespace=f"small_{size}_")
        conf = dict(run["config"],updates=2000)
        exp.fit(f"small_{size}",conf,"diagnostic",record_steps=(0,250,500,1000,2000))
        first,last = exp.rows[0],exp.rows[-1]
        result.append(dict(windows=size,initial_mse=first["train_mse"],final_mse=last["train_mse"],
                           improvement_pct=100*(1-last["train_mse"]/first["train_mse"]),cases=own.cases,
                           checkpoints=exp.rows,schedule_sha256=tensor_sha(schedule[:2000])))
    dump(out/"small_fit_audit.json",dict(results=result,elapsed_seconds=perf_counter()-started))


def stages(data,resume=False):
    started = perf_counter()
    out = data.out
    assert not (out/"FROZEN.json").exists()
    if out.resolve()!=OUT.resolve() and not resume:
        copyfile(OUT/"PROTOCOL.md",out/"PROTOCOL.md")
    (out/"checkpoints").mkdir(exist_ok=resume)
    train,dev = (split(data,s) for s in ("train","development"))
    prior = read_json(SIMPLE/"config.json")
    scale = prior["physical_residual_rms"]
    assert prior["normalization"]=={k:v.flatten().tolist() for k,v in data.norm.items()}
    schedule = shared_plan(train)
    if resume:
        prior_schedule = torch.load(out/"sampling.pt",map_location="cpu",weights_only=True)
        assert torch.equal(prior_schedule,schedule)
    else:
        save_tensor(out/"sampling.pt",schedule)
    config = dict(seed=SEED,correction_strength=1.,normalization=prior["normalization"],
                  scale=scale,plot=prior["plot"],backbone=data.identity,
                  batch=8,gradient_clip=1.,min_lr=1e-6,checkpoint_steps=[0,50,100,250,500,750,1000,1500,2000,3000,4000,5000,6000],
                  holdout_re=dict(interpolation=17812.,extrapolation=25482.),
                  gate=dict(gain_pct=.5,min_improved=8,max_concentration=.5),
                  data_windows=dict(train=len(train.cases),development=len(dev.cases),test=len(data.cases["test"])),
                  torch=torch.__version__,gpu=torch.cuda.get_device_name())
    if resume:
        assert read_json(out/"config.json")==config
    else:
        dump(out/"config.json",config)
    sources = dict(protocol_sha256=sha(out/"PROTOCOL.md"),
                   prior={f:sha(SIMPLE/f) for f in ("config.json","training.csv","diagnostic_audit.json",
                                                  "native_train.json","native_development.json","artifact_audit.json")},
                   train_and_dev_backbone_cache={s:read_json(out/f"native_{s}.json")["P_sha256"]
                                                 for s in ("train","development")},
                   test_identity_pending_freeze=True,backbone=data.identity,
                   sampling_sha256=tensor_sha(schedule))
    if resume:
        assert read_json(out/"sources.json")==sources
    else:
        dump(out/"sources.json",sources)
        residual_invariants(train,out)
        analytic(train,out)
    xp = Experiment(train,dev,data.norm,scale,out,schedule)
    xp.replay(SIMPLE)
    dependency(train,data.norm,scale,out,("S0",),xp.runs)
    for name in tuple(CHANNELS)[1:]:
        xp.fit(name,dict(DEFAULT,input=name),"input")
    inputs_best = xp.best(CHANNELS)
    dependency(train,data.norm,scale,out,CHANNELS,xp.runs)
    initial = xp.runs[inputs_best]["config"]
    lr_runs = [inputs_best]
    for name,rate in (("lr_1e-4",1e-4),("lr_1e-3",1e-3)):
        lr_runs.append(xp.fit(name,dict(initial,lr=rate),"learning_rate"))
    lr_best = xp.best(lr_runs)
    width_runs = [lr_best]
    for width in (16,32):
        width_runs.append(xp.fit(f"width_{width}",dict(xp.runs[lr_best]["config"],width=width),"width"))
    width_best = xp.best(width_runs)
    xp.fit("budget_6000",dict(xp.runs[width_best]["config"],updates=6000),"budget")
    prelim = xp.best(xp.runs)
    conf = xp.runs[prelim]["config"]
    before = xp.runs[prelim]["selected"]
    end = next(r for r in xp.rows if r["run"]==prelim and r["step"]==conf["updates"])
    gap = 100*(end["train_G"]/end["train_native"]-end["dev_G"]/end["dev_native"])
    relapse = 100*(end["dev_mse"]-before["dev_mse"])/end["dev_native"]
    regularize = gap>=1. and relapse>=.25
    if regularize:
        xp.fit("wd_1e-3",dict(conf,weight_decay=1e-3),"regularization")
    base = xp.best(xp.runs)
    run = xp.runs[base]
    path = out/"pre_holdout_selection.json"
    if path.exists():
        prior_selection = read_json(path)
        assert prior_selection["model"]==base and prior_selection["holdout_re"]==config["holdout_re"]
        assert prior_selection["selection"]["config"]==run["config"]
        assert prior_selection["selection"]["selected"]["sha256"]==run["selected"]["sha256"]
    else:
        dump(path,dict(model=base,selection=run,holdout_re=config["holdout_re"]))
    small_fits(run,train,data.norm,scale,out)
    held = holdout(run,train,out)
    net = load(run["config"],run["selected"],out,dev.p.device)
    r = predict(net,inputs(dev,data.norm,run["config"]["input"]),scale)
    clinical = assessment(r,dev)
    best_row = run["selected"]
    gain = 100*best_row["dev_G"]/best_row["dev_native"]
    qualified = (best_row["dev_E"]>0 and gain>=.5 and clinical["improved"]>=8 and
                 clinical["concentration"] is not None and clinical["concentration"]<=.5)
    migration = all(h["improvement_pct"]>0 and 2*h["improved"]>h["trajectories"] for h in held)
    stage4 = qualified and migration
    stages = dict(stage1_new_fits=4,stage2_new_fits=5+int(regularize),stage3_holdout_fits=2,
                  stage4_enabled=stage4,base=base,dev_gain_pct=gain,development=clinical,
                  regularization=dict(executed=regularize,gap_percentage_points=gap,relapse_percentage_points=relapse),
                  passed_development_gate=qualified,passed_migration_gate=migration,
                  stopping_reason=None if stage4 else "development or internal migration screening condition unmet")
    dump(out/"stage_decisions.json",stages)
    print("GATE",stages,flush=True)
    from .loss_search import gradient_diagnostics, search
    gradients = gradient_diagnostics(xp,base)
    choices = search(xp,base,gradients) if stage4 else {}
    stages["stage4_new_fits"] = sum(r["stage"].startswith("loss_") for r in xp.runs.values())
    dump(out/"stage_decisions.json",stages)
    table(out/"hyperparameter_search.csv",[
        dict(run=k,stage=v["stage"],input=v["config"]["input"],width=v["config"]["width"],
             lr=v["config"]["lr"],updates=v["config"]["updates"],weight_decay=v["config"]["weight_decay"],
             step=v["selected"]["step"],train_gain_pct=100*v["selected"]["train_G"]/v["selected"]["train_native"],
             dev_gain_pct=100*v["selected"]["dev_G"]/v["selected"]["dev_native"],
             elapsed_seconds=v["elapsed_seconds"],
             parameters=sum(p.numel() for p in network(v["config"],"cpu").parameters()))
        for k,v in xp.runs.items() if not v["stage"].startswith("loss_")])
    selection = dict(input_best=inputs_best,lr_best=lr_best,width_best=width_best,base=base,
                     auxiliary=choices,runs=xp.runs,controls=["Native","B1","B2"],
                     rule="minimum development physical MSE among nonzero checkpoint corrections; ties by earliest step")
    dump(out/"selections.json",selection)
    files = ("PROTOCOL.md","config.json","sources.json","native_train.json","native_development.json","sampling.pt",
             "training.csv","selections.json","stage_decisions.json","linear_controls.json","holdouts.json",
             "gradient_diagnostics.csv","input_dependency.csv","small_fit_audit.json","hyperparameter_search.csv")
    dump(out/"FROZEN.json",dict(seed=SEED,best_base=base,stage4=choices,decision=stages,
                               bindings={f:sha(out/f) for f in files},elapsed_seconds=perf_counter()-started))
    seal(out)
    print("FROZEN BEFORE TEST",base,stage4,flush=True)


def seal(out):
    """Co-locate immutable selection evidence before any test cache is opened."""
    assert not (out/"native_test.json").exists()
    path = out/"FROZEN.json"
    frozen = read_json(path)
    for name,digest in frozen["bindings"].items():
        assert sha(out/name)==digest
    selected = read_json(out/"selections.json")
    # Bind the design preceding the two refits, together with their exact schedules.
    extra = ["pre_holdout_selection.json", "holdout_leads.csv", "diagnostic_invariants.json",
             "small_4_training.csv", "small_16_training.csv", "training_residual_stats.json",
             "lead_residual_stats.csv", "spatial_residual_stats.csv", "regime_residual_stats.csv"]
    for kind in ("interpolation","extrapolation"):
        extra.extend((f"holdout_{kind}_schedule.pt", f"holdout_{kind}_training.csv"))
    if selected["auxiliary"]:
        extra.append("loss_search.csv")
    frozen["bindings"].update({file:sha(out/file) for file in extra})
    provenance = {}
    for split_name in ("train","development"):
        entry = read_json(out/f"native_{split_name}.json")
        provenance[split_name] = dict(x=entry["identity"]["x_sha256"],y=entry["identity"]["y_sha256"],
                                      point=entry["P_sha256"],cache=entry["file_sha256"])
    test_manifest_ref = "df3cf523cb205d9e31c8ebb4770e993d018ea992:results/corrector_simple_baseline/native_test.json"
    test_manifest_blob = subprocess.check_output(["git","rev-parse",test_manifest_ref],
                                                 cwd=SIMPLE,text=True).strip()
    assert len(test_manifest_blob)==40
    frozen.update(selection_rule=selected["rule"],candidates=selected["runs"],controls=selected["controls"],
                  previous_test_manifest_ref=test_manifest_ref,previous_test_manifest_blob=test_manifest_blob,
                  known_data_sha256=provenance,training_and_diagnostic_weights={
                      t.relative_to(out).as_posix():sha(t) for t in sorted((out/"checkpoints").glob("*.pt"))})
    dump(path,frozen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",choices=("all","train","resume","seal","score"),default="all")
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--cache",type=Path,default=CACHE)
    parser.add_argument("--raw",type=Path,default=RAW)
    args = parser.parse_args()
    if args.action=="seal":
        seal(args.out)
        return
    configure()
    args.out.mkdir(parents=True,exist_ok=True)
    data = Data(args.raw,args.cache,args.out)
    if args.action in ("all","train","resume"):
        stages(data,resume=args.action=="resume")
    if args.action in ("all","score"):
        from .evaluate import evaluate
        from .plot import plot
        evaluate(data)
        plot(data.out)


if __name__=="__main__":
    main()
