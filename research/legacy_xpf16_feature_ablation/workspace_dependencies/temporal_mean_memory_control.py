"""Same-budget temporal-mean control before evaluating heldout PDE identities.

This removes temporal variation while preserving the encoded per-pixel mean.
It distinguishes a full error trajectory from simple multi-frame averaging.
Original three-model artifacts remain frozen and are never overwritten.
"""

import hashlib
import json
import math
import time

import torch

from clean_reliability_common import ROOT, configure, digest, dump
from experiment_fullblock_memory_risk import OUT as TRAIN, SOURCE, HEAD_SOURCE, Provider as OriginalProvider, load
from experiment_reliability_transport_e12 import FIT_RECORDS, FIT_STARTS, history, target, geometry, normalization
from experiment_extended_train_error_adapters import LATE_RECORDS, LATE_STARTS
from experiment_temporal_error_uncertainty import pack, normalized
from reliability_risk_transport import base_features
from fullblock_memory_risk_head import FullBlockMemoryRiskHead
from causal_shape_calibration import initial_state
from experiment_continue_rich_uncertainty import MODES, controller
from conditional_online_risk import risk_features
from experiment_direct_error_uncertainty import metrics
from evaluate_reliability_transport_e12 import ranking, summarize
from diagnose_train_distribution_drift import OUT as DATA
from pilot_fullblock_memory_risk import OUT as PILOT


OUT = ROOT / "runs/fullblock_temporal_mean_control_20261109"


class TemporalMeanMemoryRiskHead(FullBlockMemoryRiskHead):
    def __init__(self, memory_scale):
        super().__init__("full", memory_scale)

    def encode_memory(self, previous_error):
        encoded = torch.log1p(previous_error.abs() / self.memory_scale).clamp(0,5)
        return encoded.mean(dim=2,keepdim=True).expand_as(encoded)


class Provider(OriginalProvider):
    def __init__(self, norm):
        super().__init__(norm)
        head = TemporalMeanMemoryRiskHead(self.scale).cuda().eval().requires_grad_(False)
        head.load_state_dict(load(OUT / "mean.pt")["model"])
        self.heads["mean"] = head


@torch.no_grad()
def training_data(norm):
    stats = load(HEAD_SOURCE / "statistics.pt")["statistics"]
    values = [[] for _ in range(5)]
    index = {r["file"]: i for i,r in enumerate(FIT_RECORDS)}
    for later, starts, records in ((False,FIT_STARTS,FIT_RECORDS),(True,LATE_STARTS,LATE_RECORDS)):
        previous = load(TRAIN / ("previous_fit380.pt" if later else "previous_fit180.pt"))["prediction"]
        if later:
            previous = previous[[index[r["file"]] for r in records]]
        geo = geometry(records)
        for start in starts:
            if later:
                saved = load(SOURCE / "later_data" / (str(start) + ".pt"))
                observed, point, last, residual = saved["observed"],saved["prediction"],saved["last"].float(),saved["residual"]
                base = base_features(observed,point*norm["std"]+norm["mean"],norm["mean"],norm["std"],geo)
            else:
                point,base,last = pack("fit",start,norm,geo)
                observed = history("fit",start)
                residual = (target("fit",start)-norm["mean"])/norm["std"]-point
            h = (observed-norm["mean"])/norm["std"]
            base,last = normalized(base,last,stats)
            memory = h-previous
            for destination,value in zip(values,(base,last,h,memory,residual)):
                destination.append(value)
            previous = point
    return [torch.cat(parts) for parts in values]


def train(norm):
    base,last,h,memory,residual = training_data(norm)
    scale = load(TRAIN / "statistics.pt")["memory_scale"]
    torch.manual_seed(42)
    head = TemporalMeanMemoryRiskHead(scale).cuda()
    original = load(SOURCE / "expanded_rich.pt")["model"]
    missing = head.load_state_dict(original,strict=False)
    assert set(missing.missing_keys)=={"memory_scale","memory_projection.weight","memory_projection.bias"}
    assert sum(p.numel() for p in head.parameters())==48340
    sampler = torch.Generator(device="cuda").manual_seed(42)
    optimizer = torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-5)
    schedule = hashlib.sha256()
    logs,begin = [],time.time()
    for step in range(1,2001):
        ix = torch.randint(819,(4,),device="cuda",generator=sampler)
        available = (torch.rand(4,device="cuda",generator=sampler)>=.2).float()
        schedule.update(ix.cpu().numpy().tobytes())
        schedule.update(available.cpu().numpy().tobytes())
        rate = 1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/2000))
        for group in optimizer.param_groups:
            group["lr"]=rate
        optimizer.zero_grad(set_to_none=True)
        loss = head.loss(base[ix],last[ix],h[ix],memory[ix],residual[ix],available)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(head.parameters(),1))
        optimizer.step()
        if step==1 or step%500==0:
            logs.append(dict(step=step,loss=float(loss),elapsed=time.time()-begin))
            print("MEAN_MEMORY_TRAIN",logs[-1],flush=True)
    source_schedule = json.loads((TRAIN / "heads_frozen.json").read_text())["minibatches_and_availability_masks_sha256"]["full"]
    assert schedule.hexdigest()==source_schedule
    torch.save(dict(model=head.state_dict(),mode="mean",seed=42,additional_steps=2000,examples=819),OUT/"mean.pt")
    dump(OUT/"training.json",logs)
    dump(OUT/"head_frozen.json",dict(time=time.time(),sha256=digest(OUT/"mean.pt"),parameters=48340,
        same_minibatches_availability_masks_sha256=schedule.hexdigest()))


@torch.no_grad()
def calibrate_and_pilot(norm):
    provider = Provider(norm)
    scores=[]
    for start in range(0,161,20):
        point,bounds=provider.bounds("calibration",start)
        lo,hi=bounds["mean"]
        residual=(target("calibration",start)-norm["mean"])/norm["std"]-point
        scores.append(((residual-.5*(lo+hi)).abs()/(.5*(hi-lo)).clamp_min(1e-6)).cpu())
    initial=initial_state(torch.cat(scores).cuda(),"separable_integral")
    torch.save({"mean":{"separable_integral":{key:value.cpu() for key,value in initial.items()}}},OUT/"calibration.pt")
    protocol=json.loads((DATA/"protocol.json").read_text())
    records=protocol["records"]
    geo=geometry(records)
    states={mode:controller(mode,initial,len(records)) for mode in MODES}
    rows=[]
    for start in protocol["starts"]:
        known=load(PILOT/"known"/(str(start)+".pt"))
        observed=load(DATA/"known"/(str(start)+".pt"))["observed"]
        point=known["prediction"]
        h=(observed-norm["mean"])/norm["std"]
        base=base_features(observed,point*norm["std"]+norm["mean"],norm["mean"],norm["std"],geo)
        base,last=normalized(base,known["last"].float(),provider.stats)
        lo,hi=provider.heads["mean"].bounds(base,last,h,known["previous_error"],known["available"])
        risk=risk_features(h,point,.5*(hi-lo))
        issued={mode:state.issue(lo,hi,risk) for mode,state in states.items()}
        with (OUT/"events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event="issued",start=start,time=time.time()))+"\n")
        residual=load(DATA/"labels"/(str(start)+".pt"))
        with (OUT/"events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event="revealed",start=start,time=time.time()))+"\n")
        mse=(residual*norm["std"]).square().mean((2,3,4))
        raw={"raw_"+key:value for key,value in ranking(.5*(hi-lo),residual.abs()).items()}
        for mode,(lower,upper) in issued.items():
            values=dict(**metrics(lower,upper,residual),**ranking(.5*(upper-lower),residual.abs()),**raw)
            for i,record in enumerate(records):
                for c,channel in enumerate(("u","v")):
                    rows.append(dict(kind="diffusion",split="train_time_pilot",group=record["group"],file=record["file"],
                        start=start,period="later" if start>=700 else "post_cold" if start>=600 else "prefix",
                        channel=channel,map="mean",variant=mode,mse=float(mse[i,c]),
                        **{key:float(value[i,c]) for key,value in values.items()}))
            states[mode].reveal(residual)
        print("MEAN_MEMORY_PILOT",start,flush=True)
    dump(OUT/"per_case.json",rows)
    combined=json.loads((PILOT/"per_case.json").read_text())+rows
    dump(OUT/"combined_per_case.json",combined)
    dump(OUT/"combined_summary.json",summarize(combined))
    dump(OUT/"complete.json",dict(time=time.time(),old_models_unchanged=True,mean_is_encoded_temporal_average=True))


def main():
    configure()
    assert json.loads((PILOT/"gate.json").read_text())["proceed_full_evaluation"]
    OUT.mkdir(exist_ok=False)
    original_hashes={str(path.relative_to(ROOT)):digest(path) for path in TRAIN.glob("*.pt")}
    dump(OUT/"protocol.json",dict(time=time.time(),source_sha256=digest(ROOT/"temporal_mean_memory_control.py"),
        source_train_protocol_sha256=digest(TRAIN/"protocol.json"),original_artifacts=original_hashes,
        comparison="Same48340allocatedparameters,seed42,819pairs,2000updates,initialcore,optimizer and exactavailability/minibatchschedule. Mean encodedmemory over20previousframes and repeat; preserves per-pixel/channelencodedmean,removes temporalcontrast.",
        reason="Original train-only pilot passed forfull andlast. Meancontrol added before any development/test field evaluation to distinguish averaging from full-history temporal structure. Not a gain or seed scan.",
        limits="Same training IDs/later times reused. No claim that temporalcontrast or non-diagonalprojection is useful without this control."))
    norm=normalization()
    train(norm)
    calibrate_and_pilot(norm)
    assert all(digest(ROOT/path)==sha for path,sha in original_hashes.items())
    print("TEMPORAL_MEAN_CONTROL_COMPLETE",flush=True)


if __name__=="__main__":
    main()
