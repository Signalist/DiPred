"""DiPred incremental-residual research pipeline."""
import argparse
from pathlib import Path
from shutil import copyfile
from statistics import mean

import torch

from reliability.backbone import configure
from research.corrector.run import sha
from research.corrector_residual_feedback.data import ROOT,guard
from research.corrector_residual_feedback.fit import state_digest
from research.corrector_residual_feedback.run import references
from research.corrector_spacetime_optimization.data import Study as HistoricalStudy, OUT as OPTIMIZATION
from research.corrector_spacetime_validation.data import OUT as VALIDATION
from research.corrector_spacetime_validation.fit import Fits,atomic_json,load_net
from research.corrector_spacetime_validation.gradients import monitor_indices,vectors
from research.corrector_window_study.data import RAW,read_json

OUT = ROOT/"results/corrector_incremental_structure"
CACHE = ROOT/".cache/corrector_incremental_structure"
FIGURES = ROOT/".User/corrector_incremental_structure"


def prepare(out,cache,raw):
    for item in ("PROTOCOL.md","hypotheses.json","experiment_plan.json","data_roles.json","reuse_manifest.json"):
        assert (out/item).is_file(),item
    reuse=read_json(out/"reuse_manifest.json")
    assert all(sha(ROOT/p)==v for p,v in reuse["bound_sources"].items())
    return HistoricalStudy(out,cache,raw)


def calibrated_e(study,train,dev,z,dz):
    frozen=guard(OPTIMIZATION)
    baseline=read_json(OPTIMIZATION/"run_E_MSE.json")
    state=OPTIMIZATION/"shared_E.pt"
    point=next(r for r in baseline["training"] if r["step"]==1000)
    assert frozen["checkpoints"][point["checkpoint"]]==point["sha256"]
    assert frozen["bindings"][state.name]==sha(state)
    net=load_net(OPTIMIZATION/point["checkpoint"],point["sha256"],"E",train.p.device)
    names=("match2","diag","perm")
    target=(train.y-train.p)/study.scale
    batches=monitor_indices(train.cases)
    draws=vectors(net,z["E"],target,batches,names)
    norms={k:mean(float(p[k][1].norm()) for p in draws) for k in draws[0]}
    assert all(v>1e-12 for v in norms.values())
    weights={f"Ecal_{k}_{rho}":dict(term=k,lambda_=rho*norms["mse"]/norms[k],rho=rho)
             for k in names for rho in (.1,.3)}
    actual=read_json(VALIDATION/"math_verification.json")["permutation"]
    from research.corrector_spacetime_validation.objectives import ORDER,verify_math
    assert list(ORDER)==actual
    atomic_json(study.out/"E_calibration.json",dict(weights=weights,norms=norms,batches=batches,
                                                  permutation=actual,shared=sha(state),math=verify_math()))
    engine=Fits(study.out,study.cache,train,z["E"],study.scale,study.schedule,references()[2],dev,dz["E"])
    return {k:engine.fit(k,"E",{v["term"]:v["lambda_"]},shared=state,origin="matched-gradient E")
            for k,v in weights.items()}


def archive_calibrated_states(study,records):
    """Bind surviving legacy journals to their originally fitted weight snapshots."""
    out=study.out
    archive=out/"training_states"
    archive.mkdir(exist_ok=True)
    cache=study.cache/"legacy_journals"
    coverage={}
    for name,record in records.items():
        expected={r["step"]:r for r in record["training"] if r["step"]>1000}
        saved={int(p.stem.rsplit("_",1)[1]):p for p in archive.glob(f"{name}_*.pt")}
        paths=sorted(cache.glob(f"*_{name}.pt"))
        for path in paths:
            state=torch.load(path,map_location="cpu",weights_only=False)
            step=state["step"]
            if step not in expected or step in saved:
                continue
            destination=archive/f"{name}_{step}.pt"
            pending=destination.with_suffix(".tmp")
            copyfile(path,pending)
            pending.replace(destination)
            saved[step]=destination
        retained=[]
        for step,path in sorted(saved.items()):
            state=torch.load(path,map_location="cpu",weights_only=False)
            row=expected[step]
            assert state["config_hash"]==record["config_hash"]
            assert state["step"]==state["scheduler"]["last_epoch"]==step
            assert state["rows"][-1]==row
            assert {int(v["step"]) for v in state["optimizer"]["state"].values()}=={step}
            assert state["schedule_sha256"]==record["config"]["schedule"]
            assert set(state["rng"])=={"python","numpy","torch","cuda"}
            weights=torch.load(out/row["checkpoint"],map_location="cpu",weights_only=True)
            assert state_digest(state["model"])==state_digest(weights)
            retained.append(dict(step=step,path=str(path.relative_to(out)),sha256=sha(path)))
        coverage[name]=dict(retained=retained,unavailable=sorted(set(expected)-set(saved)),
                            shared_step=1000,shared_sha256=record["config"]["shared"])
    atomic_json(out/"calibrated_state_archive.json",coverage)
    return coverage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",default="all",choices=("all","calibration","core"))
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--cache",type=Path,default=CACHE)
    parser.add_argument("--raw",type=Path,default=RAW)
    parser.add_argument("--figures",type=Path,default=FIGURES)
    args=parser.parse_args()
    configure()
    study=prepare(args.out,args.cache,args.raw)
    if not (args.out/"FROZEN.json").exists():
        train,train_base,z=study.get("train")
        dev,dev_base,dz=study.get("development")
        direct=calibrated_e(study,train,dev,z,dz)
        if args.action!="calibration":
            from .core import core
            calibration_states=archive_calibrated_states(study,direct)
            core(study,train,dev,train_base,dev_base,direct,z,dz,calibration_states)
    if args.action=="all":
        from .validation import validate
        validate(study,args.figures)


if __name__=="__main__":
    main()
