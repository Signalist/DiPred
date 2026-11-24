"""Legacy replay, diagnostics, controlled fitting, freeze, then test scoring."""
import argparse
from pathlib import Path
from shutil import copyfile
from statistics import mean
from time import perf_counter

import torch

from reliability.backbone import configure
from research.corrector.run import dump, sha, table
from research.corrector_predictability.data import EXPANDED, SMALL, Split, normalization, sources
from research.corrector_window_study.data import RAW, read_json, tensor_sha

from .data import CACHE, expansion
from .diagnostics import run as diagnostics
from .fit import Experiment, KINDS, MODES, SEEDS, STEPS, choices
from .legacy import OUT, replay
from .model import network


def train(out, cache, raw):
    assert not (out/"FROZEN.json").exists()
    out.mkdir(parents=True,exist_ok=True)
    (out/"checkpoints").mkdir(exist_ok=False)
    if out.resolve()!=OUT.resolve():
        copyfile(OUT/"PROTOCOL.md",out/"PROTOCOL.md")
    manifest, lineage = sources()
    base = Split(manifest,"train",EXPANDED)
    dev = Split(manifest,"development",SMALL)
    avg, std, scale, _ = normalization(base.x.device)
    if not (out/"legacy_replay.json").exists():
        replay(out)
    if not (out/"diagnostic_audit.json").exists():
        diagnostics(base,avg,std,scale,out)
    assert read_json(out/"diagnostic_audit.json")["passed"]
    full = expansion(base,manifest,out,cache,raw)
    cfg = dict(seeds=SEEDS,checkpoints=STEPS,batch=8,lr_max=3e-4,lr_min=1e-6,
               updates=2000,weight_decay=1e-5,clip=1.,parameter_counts={
                   kind:sum(p.numel() for p in network(kind,base.x.device).parameters())
                   for kind in ("Point","Block")},
               architecture=KINDS,train_windows={"183":len(base.cases),"expanded":len(full.cases)},
               development_windows=len(dev.cases),device=torch.cuda.get_device_name(),
               torch=torch.__version__,cudnn=torch.backends.cudnn.version(),
               input_mean=avg.flatten().tolist(),input_std=std.flatten().tolist(),output_rms=scale)
    lineage.update(protocol_sha256=sha(OUT/"PROTOCOL.md"),
                   legacy_replay_sha256=sha(out/"legacy_replay.json"),
                   expansion_sha256=sha(out/"expansion.json"),
                   diagnostic_audit_sha256=sha(out/"diagnostic_audit.json"),
                   train_P_sha256={"base":tensor_sha(base.p),"expanded":tensor_sha(full.p)},
                   development_P_sha256=tensor_sha(dev.p))
    dump(out/"config.json",cfg)
    dump(out/"sources.json",lineage)
    rows, selected = [], {}
    experiments = {True:Experiment(base,dev,(avg,std,scale),out,True),
                   False:Experiment(full,dev,(avg,std,scale),out,False)}
    start = perf_counter()
    for mode in MODES:
        by_seed = {}
        for seed in SEEDS:
            fit_start = perf_counter()
            result = experiments[mode in ("R0","R1","R2")].run(mode,seed)
            by_seed[seed] = choices(result)
            rows.extend(result)
            table(out/"training.csv",rows)
            print("FINISHED",mode,seed,round(perf_counter()-fit_start,3),flush=True)
        selected[mode] = by_seed
    screening = {}
    for mode, candidates in selected.items():
        picked = [x["selected"] for x in candidates.values()]
        gains = [x["dev_gain_pct"] for x in picked]
        screening[mode] = dict(mean_dev_gain_pct=mean(gains),seed_dev_gain_pct=gains,
                               positive_nonzero_seeds=sum(x["alpha"]>0 and x["dev_gain_pct"]>0 for x in picked))
    eligible = [m for m,g in screening.items() if g["positive_nonzero_seeds"]>=2 and g["mean_dev_gain_pct"]>0]
    winner = min(eligible,key=lambda m:(-screening[m]["mean_dev_gain_pct"],m)) if eligible else None
    frozen = dict(config_sha256=sha(out/"config.json"),sources_sha256=sha(out/"sources.json"),
                  training_sha256=sha(out/"training.csv"),checkpoints=len(rows),
                  new_core_fits=12,selected=selected,development=screening,winner=winner,
                  elapsed_seconds=perf_counter()-start)
    dump(out/"FROZEN.json",frozen)
    print("FROZEN BEFORE CURRENT TEST",winner,screening,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",choices=("all","train","score"),default="all")
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--cache",type=Path,default=CACHE)
    parser.add_argument("--raw",type=Path,default=RAW)
    args = parser.parse_args()
    configure()
    if args.action in ("all","train"):
        train(args.out,args.cache,args.raw)
    if args.action in ("all","score"):
        from .evaluate import score
        score(args.out)


if __name__=="__main__":
    main()
