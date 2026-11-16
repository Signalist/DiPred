"""Generic complete-window replay with low-dimensional separable calibration."""

import json
import time

import torch

from clean_reliability_common import dump,manifest,digest
from experiment_reliability_transport_e12 import target
from experiment_direct_error_uncertainty import metrics
from evaluate_reliability_transport_e12 import ranking,summarize
from experiment_reliability_structured_e3 import grouped_quantile


STRATEGIES = ("raw","fixed","online_global","online_lead","online_full_lead","online_separable")


def scales(ratio, pooled=False):
    g=grouped_quantile(ratio,"global",pooled=pooled).clamp(1e-6,1e3)
    lead=grouped_quantile(ratio,"lead",pooled=pooled).clamp(1e-6,1e3)
    patch=grouped_quantile(ratio,"patch",pooled=pooled).clamp(1e-6,1e3)
    separable=lead*patch/g
    correction=grouped_quantile(ratio/separable,"global",pooled=pooled)
    # The extra global correction targets the revealed block's marginal rate;
    # it does not guarantee future or conditional coverage.
    separable=(separable*correction).clamp(1e-6,1e3)
    return dict(fixed=g,online_global=g,online_lead=(g*lead).sqrt(),
                online_full_lead=lead,online_separable=separable)


@torch.no_grad()
def replay(out,provider,names):
    scores={name:[] for name in names}
    for start in range(0,161,20):
        p,bounds=provider.bounds("calibration",start)
        residual=(target("calibration",start)-provider.norm["mean"])/provider.norm["std"]-p
        for name,(lo,hi) in bounds.items():
            mid,width=.5*(lo+hi),(.5*(hi-lo)).clamp_min(1e-6)
            scores[name].append(((residual-mid).abs()/width).cpu())
    initial={name:scales(torch.cat(values).cuda(),pooled=True) for name,values in scores.items()}
    torch.save({name:{key:value.cpu() for key,value in values.items()} for name,values in initial.items()},out/"calibration.pt")
    dump(out/"calibration_locked.json",dict(time=time.time(),sha256=digest(out/"calibration.pt")))
    rows=[]
    for split in ("development","test"):
        records=manifest()["files"][split]
        state={name:{key:value.expand(len(records),*value.shape[1:]).clone() for key,value in values.items()} for name,values in initial.items()}
        for start in range(0,761,20):
            p,raw=provider.bounds(split,start)
            intervals={}
            for name,(lo,hi) in raw.items():
                mid,width=.5*(lo+hi),.5*(hi-lo)
                intervals[name]=dict(raw=(lo,hi))
                for strategy in STRATEGIES[1:]:
                    q=initial[name][strategy] if strategy=="fixed" else state[name][strategy]
                    intervals[name][strategy]=(mid-q*width,mid+q*width)
            with (out/"events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event="issued",split=split,start=start,feedback=start//20,time=time.time()))+"\n")
            residual=(target(split,start)-provider.norm["mean"])/provider.norm["std"]-p
            with (out/"events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event="revealed",split=split,start=start,time=time.time()))+"\n")
            mse=(residual*provider.norm["std"]).square().mean((2,3,4))
            for name,methods in intervals.items():
                lo,hi=raw[name]
                mid,width=.5*(lo+hi),(.5*(hi-lo)).clamp_min(1e-6)
                location=ranking(width,residual.abs())
                for strategy,(lower,upper) in methods.items():
                    values=dict(**metrics(lower,upper,residual),**location)
                    for i,record in enumerate(records):
                        for c,channel in enumerate(("u","v")):
                            rows.append(dict(kind="diffusion",split=split,file=record["file"],group=record["group"],
                                 start=start,period="confirmation" if start>=580 else "main",channel=channel,map=name,variant=strategy,
                                 mse=float(mse[i,c]),**{key:float(value[i,c]) for key,value in values.items()}))
                observed=scales((residual-mid).abs()/width)
                for strategy in STRATEGIES[2:]:
                    state[name][strategy]=(state[name][strategy]*observed[strategy]).sqrt()
            if start in (180,380,580,760):
                print("REPLAY",split,start,flush=True)
    dump(out/"per_case.json",rows)
    dump(out/"summary.json",summarize(rows))
