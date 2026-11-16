"""Shared complete-window replay for adaptive risk coefficient studies."""

import json
import time

import torch

from clean_reliability_common import dump,manifest
from experiment_reliability_transport_e12 import history,target
from experiment_direct_error_uncertainty import metrics
from evaluate_reliability_transport_e12 import ranking,summarize
from conditional_online_risk import risk_features


@torch.no_grad()
def replay(out,provider,names,modes,factory,initial):
    rows=[]
    for split in ("development","test"):
        records=manifest()["files"][split]
        controllers={name:{mode:factory(mode,initial[name]["separable_integral"],len(records)) for mode in modes} for name in names}
        for start in range(0,761,20):
            p,bounds=provider.bounds(split,start)
            h=(history(split,start)-provider.norm["mean"])/provider.norm["std"]
            features={name:risk_features(h,p,.5*(bounds[name][1]-bounds[name][0])) for name in names}
            issued={name:{mode:cal.issue(*bounds[name],features[name]) for mode,cal in controls.items()} for name,controls in controllers.items()}
            with (out/"events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event="issued",split=split,start=start,feedback=start//20,time=time.time()))+"\n")
            residual=(target(split,start)-provider.norm["mean"])/provider.norm["std"]-p
            with (out/"events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event="revealed",split=split,start=start,time=time.time()))+"\n")
            mse=(residual*provider.norm["std"]).square().mean((2,3,4))
            for name,controls in issued.items():
                raw={"raw_"+key:value for key,value in ranking(.5*(bounds[name][1]-bounds[name][0]),residual.abs()).items()}
                for mode,(lo,hi) in controls.items():
                    values=dict(**metrics(lo,hi,residual),**ranking(.5*(hi-lo),residual.abs()),**raw)
                    for i,record in enumerate(records):
                        for c,channel in enumerate(("u","v")):
                            rows.append(dict(kind="diffusion",split=split,file=record["file"],group=record["group"],start=start,
                                 period="confirmation" if start>=580 else "main",channel=channel,map=name,variant=mode,
                                 mse=float(mse[i,c]),**{key:float(value[i,c]) for key,value in values.items()}))
                    controllers[name][mode].reveal(residual)
            if start in (180,380,580,760):
                print("REPLAY",split,start,flush=True)
    dump(out/"per_case.json",rows)
    dump(out/"summary.json",summarize(rows))
