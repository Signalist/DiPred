"""Run bounded Stages 0–3, then open the test set once."""
import argparse
import shlex
import sys
import time
from pathlib import Path

import torch

from reliability.backbone import configure
from research.corrector.run import dump, sha, table
from research.corrector_window_study.data import tensor_sha

from .alpha import grid
from .data import CACHE, EXPANDED, OLD, OUT, SMALL, Split, normalization, sources
from .evaluate import evaluate
from .features import produce
from .fit import BATCH, SEEDS, STEPS, Loop, correction, directional, gate, network, read_model, standard
from .model import FEATURE_GROUPS, PROXY_INDICES, inputs
from .verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    a = parser.parse_args()
    out = a.out
    out.mkdir(exist_ok=True, parents=True)
    assert not (out/"FROZEN.json").exists() and not (out/"checkpoints").exists()
    started = time.monotonic()
    print("COMMAND", shlex.join([sys.executable,"-m",__spec__.name,*sys.argv[1:]]),flush=True)
    configure()
    device="cuda"
    manifest,lineage=sources()
    train=Split(manifest,"train",EXPANDED)
    small=train.small()
    dev=Split(manifest,"development",SMALL)
    mean,units,scale,thresholds=normalization(device)
    fx=produce(train,"train",manifest,a.cache,out,None)
    fd=produce(dev,"development",manifest,a.cache,out,None)
    stats=fx.mean((0,2,3),keepdim=True),fx.std((0,2,3),unbiased=False,keepdim=True)
    assert all(torch.isfinite(v).all() for v in stats) and (stats[1]>0).all()
    values={id(train):fx,id(dev):fd}
    cached_inputs={}
    def lookup(data, mode):
        key=id(data),mode
        if key not in cached_inputs:
            cached_inputs[key]=inputs(data.x,data.p,mean,units,mode,values.get(id(data)),stats)
        return cached_inputs[key]

    checks=verify(dev,lookup(dev,"XPF"),scale)
    print("PREFLIGHT",checks,flush=True)
    dump(out/"sources.json",lineage)
    metadata=dict(pytorch=torch.__version__, python=sys.version.split()[0],
        cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),
        train_small=len(small.x), train_expanded=len(train.x), development=len(dev.x),
        batch=BATCH,seeds=list(SEEDS),steps=list(STEPS),adam_lr=.001,
        feature_slot_channels=48,unused_input_weights_xp_zero=48*16,proxy_indices=list(PROXY_INDICES),
        parameters={m:sum(v.numel() for v in network(m,device).parameters())
                    for m in ("Small",*FEATURE_GROUPS)},
        train_only_feature_mean=stats[0].flatten().tolist(),
        train_only_feature_std=stats[1].flatten().tolist(),
        common_raw_residual_rms=scale, channel_mean=mean.flatten().tolist(),
        channel_scale=units.flatten().tolist(),thresholds=thresholds,
        plot_case=dict(trajectory=dev.cases[0]["file"],start=dev.cases[0]["start"],lead=20,seed=42),
        previous_head="b1b8b8f8050e86f2dcab0e05519b809768f6002c",
        protocol_sha256=sha(OUT/"PROTOCOL.md"))
    dump(out/"config.json",metadata)
    old_alphas, train_log, grids = {}, [], []
    for arm,row in lineage["frozen_correctors"].items():
        old=read_model(OLD,"XP",row,device)
        r=correction(old,lookup(dev,"XP"),scale)
        terms=grid(r,dev.y.double()-dev.p.double(),arm)
        grids.extend(terms)
        old_alphas[arm]=terms[0]["alpha"]
    table(out/"stage0_development_alpha.csv",grids)

    candidates={}
    for mode in ("Small","Expanded",*FEATURE_GROUPS):
        dataset=small if mode=="Small" else train
        loop=Loop(dataset,dev,lookup(dataset,mode),lookup(dev,mode),scale,out)
        candidates[mode]=[standard(loop,mode,seed) for seed in SEEDS]
        train_log.extend(loop.log)
        table(out/"training.csv",train_log)
    screening=gate(candidates,dev)
    passed=[mode for mode,r in screening.items() if r["passes"]]
    direction_base=None
    if passed:
        direction_base=min(passed,key=lambda mode:(-screening[mode]["improvement_pct"],mode))
        dataset=small if direction_base=="Small" else train
        loop=Loop(dataset,dev,lookup(dataset,direction_base),lookup(dev,direction_base),scale,out)
        for seed in SEEDS:
            for arm,result in directional(loop,direction_base,seed).items():
                candidates.setdefault(arm,[]).append(result)
        train_log.extend(loop.log)
        table(out/"training.csv",train_log)
    stage=dict(stage0_new_fits=0,stage1_fits=6,stage2_fits=9,stage3_arms=12 if passed else 0,
               screening=screening,direction_base=direction_base,
               stage3_status="executed" if passed else "not triggered by development",
               training_rows=len(train_log))
    dump(out/"STAGES.json",stage)
    print("STAGE GATE",stage,flush=True)
    frozen=dict(old_checkpoints=lineage["frozen_correctors"],stage0_alpha=old_alphas,
                training_candidates=candidates,direction_base=direction_base, stages=stage,
                config_sha256=sha(out/"config.json"), sources_sha256=sha(out/"sources.json"),
                train_features_sha256=sha(out/"F_train.json"),dev_features_sha256=sha(out/"F_development.json"))
    dump(out/"FROZEN.json",frozen)
    print("FROZEN before test cache access",flush=True)
    test=Split(manifest,"test",SMALL)
    values[id(test)]=produce(test,"test",manifest,a.cache,out,sha(out/"FROZEN.json"))
    p_before=tensor_sha(test.p)
    checks.update(evaluate(test,dev,frozen,out,units,scale,thresholds,lookup))
    assert p_before==tensor_sha(test.p)
    assert lineage==sources()[1]
    checks.update(frozen_weights_unchanged=True,P_test_unchanged=True,test_P_sha256=p_before,
                  elapsed_seconds=time.monotonic()-started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    dump(out/"verification.json",checks)
    print("PASS",checks,flush=True)


if __name__=="__main__":
    main()
