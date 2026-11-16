"""Shared independent-calibration and later-training-ID diagnostic protocol."""

import json
import time
from statistics import mean

import torch

from clean_reliability_common import dump,digest
from experiment_reliability_transport_e12 import target,geometry
from experiment_fullblock_memory_risk import load
from diagnose_train_distribution_drift import OUT as PILOT_DATA
from pilot_fullblock_memory_risk import OUT as KNOWN_MEMORY
from causal_shape_calibration import initial_state
from experiment_continue_rich_uncertainty import MODES,controller
from conditional_online_risk import risk_features
from experiment_temporal_error_uncertainty import normalized
from reliability_risk_transport import base_features
from experiment_direct_error_uncertainty import metrics
from evaluate_reliability_transport_e12 import ranking,summarize


@torch.no_grad()
def calibrate(provider,out,names,reference_name,reference_state):
    scores = {name:[] for name in names}
    norm = provider.norm
    for start in range(0,161,20):
        point,bounds = provider.bounds("calibration",start)
        residual = (target("calibration",start)-norm["mean"])/norm["std"]-point
        for name,(lower,upper) in bounds.items():
            scores[name].append(((residual-.5*(lower+upper)).abs()/(.5*(upper-lower)).clamp_min(1e-6)).cpu())
    initial = {name:{"separable_integral":initial_state(torch.cat(values).cuda(),"separable_integral")}
               for name,values in scores.items()}
    assert all(torch.equal(value,reference_state[key]) for key,value in initial[reference_name]["separable_integral"].items())
    torch.save({name:{mode:{key:value.cpu() for key,value in state.items()} for mode,state in modes.items()}
                for name,modes in initial.items()},out/"calibration.pt")
    dump(out/"calibration_locked.json",dict(time=time.time(),sha256=digest(out/"calibration.pt"),
                                           reference_recomputed_exact=True))
    return initial


@torch.no_grad()
def pilot(provider,out,names,initial):
    protocol = json.loads((PILOT_DATA/"protocol.json").read_text())
    records,norm = protocol["records"],provider.norm
    geo = geometry(records)
    states = {name:{mode:controller(mode,initial[name]["separable_integral"],len(records)) for mode in MODES}
              for name in names}
    rows = []
    for start in protocol["starts"]:
        known = load(KNOWN_MEMORY/"known"/(str(start)+".pt"))
        source = load(PILOT_DATA/"known"/(str(start)+".pt"))
        point,observed = known["prediction"],source["observed"]
        assert torch.equal(point,source["prediction"])
        history = (observed-norm["mean"])/norm["std"]
        base = base_features(observed,point*norm["std"]+norm["mean"],norm["mean"],norm["std"],geo)
        base,last = normalized(base,known["last"].float(),provider.stats)
        bounds = {name:head.bounds(base,last,history,known["previous_error"],known["available"])
                  for name,head in provider.heads.items()}
        issued = {name:{mode:state.issue(*bounds[name],risk_features(history,point,.5*(bounds[name][1]-bounds[name][0])))
                       for mode,state in modes.items()} for name,modes in states.items()}
        with (out/"pilot_events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event="issued",start=start,time=time.time()))+"\n")
        residual = load(PILOT_DATA/"labels"/(str(start)+".pt"))
        with (out/"pilot_events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event="revealed",start=start,time=time.time()))+"\n")
        mse = (residual*norm["std"]).square().mean((2,3,4))
        for name,modes in issued.items():
            raw = {"raw_"+key:value for key,value in ranking(.5*(bounds[name][1]-bounds[name][0]),residual.abs()).items()}
            for mode,(lower,upper) in modes.items():
                values = dict(**metrics(lower,upper,residual),**ranking(.5*(upper-lower),residual.abs()),**raw)
                for i,record in enumerate(records):
                    for c,channel in enumerate(("u","v")):
                        rows.append(dict(kind="diffusion",split="train_time_pilot",group=record["group"],file=record["file"],start=start,
                            period="later" if start>=700 else "post_cold" if start>=600 else "prefix",channel=channel,map=name,variant=mode,
                            mse=float(mse[i,c]),**{key:float(value[i,c]) for key,value in values.items()}))
                states[name][mode].reveal(residual)
        print("CAUSAL_RISK_PILOT",start,flush=True)
    dump(out/"per_case.json",rows)
    dump(out/"summary.json",summarize(rows))
    values = {}
    metrics_to_average = ("standardized_interval_score","standardized_width","relative_aurc","top10_energy",
                          "per_frame_top10_energy","worst_frame_coverage","worst_patch_coverage")
    for mode in MODES:
        values[mode] = {}
        for name in names:
            groups = {}
            for row in rows:
                if row["map"]==name and row["variant"]==mode and row["start"]>=600:
                    groups.setdefault((row["group"],row["channel"]),[]).append(row)
            result = {metric:mean(mean(r[metric] for r in group) for group in groups.values()) for metric in metrics_to_average}
            result["coverage_error"] = mean(abs(mean(r["coverage"] for r in group)-.9) for group in groups.values())
            values[mode][name] = result
    dump(out/"pilot_macro.json",values)
    return values
