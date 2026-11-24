"""Run the fixed simple Corrector protocol, then evaluate after the freeze."""
import argparse
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import torch

from reliability.backbone import configure
from research.corrector.run import dump, sha
from research.corrector_window_study.data import RAW, read_json
from research.corrector_baseline_recovery.legacy import old_sources

from .data import BASE, CACHE, DYNAMICS, OUT, WINDOW, Data
from .fit import SEED, STEPS, preflight, train
from .measure import weighted


def fit(data):
    out = data.out
    assert not (out/"FROZEN.json").exists() and not (out/"checkpoints").exists()
    if out.resolve()!=OUT.resolve():
        copyfile(OUT/"PROTOCOL.md",out/"PROTOCOL.md")
    old_sources()
    training,development = (data.split(s) for s in ("train","development"))
    native_error = (training.y.double()-training.p.double()).square().mean((1,2,3,4))
    scale = float(weighted(native_error,training.cases).sqrt())
    cfg = dict(seed=SEED,steps=STEPS,updates=2000,batch=8,initial_lr=3e-4,final_lr=1e-6,
               weight_decay=1e-5,gradient_clip=1.,parameters=3042,correction_strength=1.,
               normalization={k:v.flatten().tolist() for k,v in data.norm.items()},physical_residual_rms=scale,
               native="single zero-noise 4-call frozen DiPred",torch=torch.__version__,
               cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),
               plot=dict(file="26700_0.h5",start=400,lead=10))
    assert sum(c["file"]==cfg["plot"]["file"] and c["start"]==cfg["plot"]["start"]
               for c in development.cases)==1
    sources = dict(backbone=data.identity,legacy_source=data.previous["pinned"],
                   expansion_sha256=sha(BASE/"expansion.json"),window_manifest_sha256=sha(WINDOW),
                   record_protocol_sha256=sha(DYNAMICS),experiment_protocol_sha256=sha(out/"PROTOCOL.md"),
                   raw=str(data.raw),splits={k:sorted({c["file"] for c in v}) for k,v in data.cases.items()})
    dump(out/"config.json",cfg)
    dump(out/"sources.json",sources)
    preflight(training,data.norm,scale,out)
    start = perf_counter()
    selected = train(training,development,data.norm,scale,out)
    bindings = {file:sha(out/file) for file in ("config.json","sources.json","diagnostic_audit.json",
                                               "native_train.json","native_development.json","training.csv")}
    dump(out/"FROZEN.json",dict(checkpoints=selected,sources_sha256=bindings,
                               elapsed_training_seconds=perf_counter()-start))
    print("FROZEN BEFORE TEST",{k:v["step"] for k,v in selected.items()},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",choices=("all","train","score"),default="all")
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--cache",type=Path,default=CACHE)
    parser.add_argument("--raw",type=Path,default=RAW)
    args = parser.parse_args()
    configure()
    args.out.mkdir(parents=True,exist_ok=True)
    data = Data(args.raw,args.cache,args.out)
    if args.action in ("all","train"):
        fit(data)
    if args.action in ("all","score"):
        from .evaluate import score
        from .plot import plot
        score(data)
        plot(args.out)


if __name__=="__main__":
    main()
