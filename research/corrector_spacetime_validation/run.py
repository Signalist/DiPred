"""Run the validation experiment in its preregistered data order."""
import argparse
from pathlib import Path

from reliability.backbone import configure
from research.corrector.run import sha
from research.corrector_window_study.data import read_json

from .data import CACHE,FIGURES,OUT,RAW,S3_ROOT,Study
from .fit import Fits
from .objectives import verify_math


def s3_prefix(study,prepared):
    train,dev,z,dz = prepared
    engine = Fits(study.out,study.cache,train,z["S3"],study.scale,study.schedule,study.initial,dev,dz["S3"])
    result = engine.fit("S3_M0","S3",{},save_shared=study.out/"shared_S3.pt")
    old = study.reference["selected"]
    assert abs(result["final"]["dev_mse"]-old["dev_mse"])<1e-12
    import torch
    assert sha(S3_ROOT/old["checkpoint"])==old["sha256"]
    old_state = torch.load(S3_ROOT/old["checkpoint"],map_location="cpu",weights_only=True)
    new_state = torch.load(study.out/result["final"]["checkpoint"],map_location="cpu",weights_only=True)
    assert all(torch.equal(old_state[k],v) for k,v in new_state.items())
    return engine,result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("s3","core","all"),default="all",nargs="?")
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--cache",type=Path,default=CACHE)
    parser.add_argument("--raw",type=Path,default=RAW)
    parser.add_argument("--figures",type=Path,default=FIGURES)
    args = parser.parse_args()
    configure()
    for file in ("PROTOCOL.md","hypotheses.json","experiment_plan.json","data_roles.json"):
        assert (args.out/file).exists(),file
    verify_math()
    study = Study(args.out,args.cache,args.raw)
    if args.action=="s3":
        s3_prefix(study,study.prepare())
        return
    if not (args.out/"FROZEN.json").exists():
        from .core import core
        core(study,study.prepare())
    if args.action=="all":
        from .validation import validate
        validate(study,args.figures)


if __name__=="__main__":
    main()
