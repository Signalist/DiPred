"""Continue fitting the existing error head without adding inference modules."""

import json
import math
import time
from pathlib import Path

import torch

from clean_reliability_common import ROOT, configure, digest, dump
from experiment_reliability_transport_e12 import FIT_RECORDS, FIT_STARTS, PREDICTOR, geometry, history, target, normalization
from experiment_structured_quantile_uncertainty import OUT as HEAD_SOURCE, Provider as FrozenProvider
from experiment_temporal_error_uncertainty import pack, normalized
from structured_quantile_uncertainty import StructuredQuantileHead
from causal_shape_calibration import initial_state
from memory_quantile_online_risk import controller as memory_controller
from smoothed_scale_online_risk import controller as scale_controller
from replay_conditional_risk import replay


OUT = ROOT / "runs/continue_rich_uncertainty_20261109"
NAMES = ("rich", "continued_rich")
MODES = ("memory_dynamic", "scale_fastcal")


def controller(mode, initial, batch):
    factory = scale_controller if mode == "scale_fastcal" else memory_controller
    return factory(mode, initial, batch)


def train(norm):
    stats = torch.load(HEAD_SOURCE/"statistics.pt", map_location="cuda", weights_only=False)["statistics"]
    geo = geometry(FIT_RECORDS)
    bases, lasts, histories, labels = [], [], [], []
    with torch.no_grad():
        for start in FIT_STARTS:
            p, base, last = pack("fit", start, norm, geo)
            base, last = normalized(base, last, stats)
            bases.append(base)
            lasts.append(last)
            histories.append((history("fit",start)-norm["mean"])/norm["std"])
            labels.append((target("fit",start)-norm["mean"])/norm["std"]-p)
        base,last,h,residual=[torch.cat(values) for values in (bases,lasts,histories,labels)]
    del bases,lasts,histories,labels
    torch.manual_seed(42)
    head=StructuredQuantileHead("rich").cuda()
    original=torch.load(HEAD_SOURCE/"rich.pt",map_location="cuda",weights_only=False)["model"]
    head.load_state_dict(original)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-5)
    sampler=torch.Generator(device="cuda").manual_seed(42)
    scale=torch.ones_like(residual[:1])
    logs=[]
    begin=time.time()
    for step in range(1,2001):
        ix=torch.randint(len(base),(4,),device="cuda",generator=sampler)
        rate=1e-6+(1e-4-1e-6)*0.5*(1+math.cos(math.pi*(step-1)/2000))
        for group in optimizer.param_groups:
            group["lr"]=rate
        optimizer.zero_grad(set_to_none=True)
        loss=head.loss(base[ix],last[ix],h[ix],residual[ix],scale)
        assert torch.isfinite(loss)
        loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(head.parameters(),1)
        assert torch.isfinite(gradient)
        optimizer.step()
        if step==1 or step%500==0:
            logs.append(dict(step=step,loss=float(loss),elapsed=time.time()-begin))
            print("TRAIN",logs[-1],flush=True)
    assert list(head.state_dict())==list(original)
    assert any(not torch.equal(value,original[key]) for key,value in head.state_dict().items())
    torch.save(dict(model=head.state_dict(),seed=42,original_steps=2000,additional_steps=2000),OUT/"continued_rich.pt")
    dump(OUT/"training.json",logs)
    dump(OUT/"head_frozen.json",dict(time=time.time(),sha256=digest(OUT/"continued_rich.pt"),
         parameters=sum(value.numel() for value in head.parameters()),examples=len(base),
         same_state_keys_and_shapes=True,original_head_sha256=digest(HEAD_SOURCE/"rich.pt")))


class Provider:
    def __init__(self,norm):
        self.norm=norm
        self.frozen=FrozenProvider(norm)
        self.head=StructuredQuantileHead("rich").cuda().eval().requires_grad_(False)
        self.head.load_state_dict(torch.load(OUT/"continued_rich.pt",map_location="cuda",weights_only=False)["model"])

    @torch.no_grad()
    def bounds(self,split,start):
        p,bounds=self.frozen.bounds(split,start)
        _,base,last=pack(split,start,self.norm,self.frozen.geo[split])
        base,last=normalized(base,last,self.frozen.stats)
        h=(history(split,start)-self.norm["mean"])/self.norm["std"]
        continued=self.head.bounds(base,last,h,torch.ones_like(p[:1]))
        return p,dict(rich=bounds["rich"],continued_rich=continued)


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    frozen=json.loads((HEAD_SOURCE/"heads_frozen.json").read_text())["artifacts"]
    predictor_sha=digest(PREDICTOR/"augmented_diffusion/model.pt")
    dump(OUT/"protocol.json",dict(time=time.time(),source_sha256=digest(Path(__file__)),
        predictor_sha256=predictor_sha,head_sha256=frozen,heads=NAMES,modes=MODES,
        architecture="Identical45060-parameter rich head; no added adapter, representation or diffusion calls.",
        training="Load original2000-step rich weights; reset AdamW;2000 additional true-residual pinball steps,LR1e-4->1e-6 cosine,batch4,dedicated seed42 sampler. Not an uninterrupted4000-step optimizer run.",
        fit_files=[r["file"] for r in FIT_RECORDS],fit_frames=[200,400],
        point="Frozen original4-call99,66,33,0, full20->20; labels only supervise error head, no distillation.",
        comparison="Same additional update count as previous adapter study, but parameter counts/LR/minibatches differ; not an exact compute-matched architecture comparison.",
        calibration="Original10separate trajectories, recomputed after head freeze. Same memory_dynamic and scale_fastcal rules for original/continued heads.",
        limits="All rules fixed before replay; no folds,multi-seed sweeps,development/test fitting or policy selection. Historically reused exploratory data, no formal coverage guarantee."))
    norm=normalization()
    train(norm)
    provider=Provider(norm)
    scores={name:[] for name in NAMES}
    for start in range(0,161,20):
        with torch.no_grad():
            p,bounds=provider.bounds("calibration",start)
            residual=(target("calibration",start)-norm["mean"])/norm["std"]-p
            for name,(lo,hi) in bounds.items():
                scores[name].append(((residual-0.5*(lo+hi)).abs()/(0.5*(hi-lo)).clamp_min(1e-6)).cpu())
    initial={name:{"separable_integral":initial_state(torch.cat(values).cuda(),"separable_integral")} for name,values in scores.items()}
    torch.save({name:{mode:{k:v.cpu() for k,v in value.items()} for mode,value in modes.items()} for name,modes in initial.items()},OUT/"calibration.pt")
    dump(OUT/"calibration_locked.json",dict(time=time.time(),sha256=digest(OUT/"calibration.pt")))
    replay(OUT,provider,NAMES,MODES,controller,initial)
    assert digest(PREDICTOR/"augmented_diffusion/model.pt")==predictor_sha
    assert all(digest(HEAD_SOURCE/name)==sha for name,sha in frozen.items())
    dump(OUT/"complete.json",dict(time=time.time(),predictor_and_original_risk_head_unchanged=True))


if __name__=="__main__":
    main()
