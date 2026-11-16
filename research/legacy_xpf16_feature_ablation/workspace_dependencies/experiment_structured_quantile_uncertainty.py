"""Predeclared richer-context and spatial-normalization risk ablations."""

import json
import math
from pathlib import Path
import time

import torch

from clean_reliability_common import ROOT,configure,digest,dump,manifest
from experiment_reliability_transport_e12 import (
    FIT_RECORDS,FIT_STARTS,PREDICTOR,geometry,history,target,normalization,
)
from experiment_temporal_error_uncertainty import (
    OUT as PREVIOUS,Provider as PreviousProvider,pack,normalized,
)
from structured_quantile_uncertainty import MODES,StructuredQuantileHead
from structured_interval_replay import replay,STRATEGIES


OUT=ROOT/"runs/structured_quantile_uncertainty_20261109"
NAMES=(*MODES,"previous_temporal","previous_2d","spread32")


def train(norm):
    stats=torch.load(PREVIOUS/"statistics.pt",map_location="cuda",weights_only=False)
    geo=geometry(FIT_RECORDS)
    bases,lasts,histories,residuals=[],[],[],[]
    for start in FIT_STARTS:
        p,base,last=pack("fit",start,norm,geo)
        base,last=normalized(base,last,stats)
        bases.append(base)
        lasts.append(last)
        histories.append((history("fit",start)-norm["mean"])/norm["std"])
        residuals.append((target("fit",start)-norm["mean"])/norm["std"]-p)
    base,last,h,residual=[torch.cat(values) for values in (bases,lasts,histories,residuals)]
    del bases,lasts,histories,residuals
    template=residual.square().mean(0,keepdim=True).sqrt().clamp_min(.01)
    torch.save(dict(template=template.cpu(),statistics={key:value.cpu() for key,value in stats.items()}),OUT/"statistics.pt")
    parameters={}
    for mode in MODES:
        torch.manual_seed(42)
        head=StructuredQuantileHead(mode).cuda()
        parameters[mode]=sum(p.numel() for p in head.parameters())
        scale=torch.ones_like(template) if mode=="rich" else template
        optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-5)
        begin=time.time()
        logs=[]
        for step in range(1,2001):
            ix=torch.randint(len(base),(4,),device="cuda")
            rate=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/2000))
            for group in optimizer.param_groups:
                group["lr"]=rate
            optimizer.zero_grad(set_to_none=True)
            loss=head.loss(base[ix],last[ix],h[ix],residual[ix],scale)
            assert torch.isfinite(loss)
            loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            assert torch.isfinite(gradient)
            optimizer.step()
            if step==1 or step%500==0:
                logs.append(dict(step=step,loss=float(loss),elapsed=time.time()-begin))
                print("TRAIN",mode,logs[-1],flush=True)
        torch.save(dict(model=head.state_dict(),mode=mode,steps=2000,seed=42),OUT/f"{mode}.pt")
        dump(OUT/f"{mode}_training.json",logs)
    dump(OUT/"heads_frozen.json",dict(time=time.time(),parameters=parameters,
         artifacts={p.name:digest(p) for p in OUT.glob("*.pt")}))


class Provider:
    def __init__(self,norm):
        self.norm=norm
        self.previous=PreviousProvider(norm)
        stored=torch.load(OUT/"statistics.pt",map_location="cuda",weights_only=False)
        self.template,self.stats=stored["template"],stored["statistics"]
        self.geo={split:geometry(manifest()["files"][split]) for split in ("calibration","development","test")}
        self.heads={}
        for mode in MODES:
            head=StructuredQuantileHead(mode).cuda().eval().requires_grad_(False)
            head.load_state_dict(torch.load(OUT/f"{mode}.pt",map_location="cuda",weights_only=False)["model"])
            self.heads[mode]=head

    @torch.no_grad()
    def bounds(self,split,start):
        p,old=self.previous.bounds(split,start)
        _,base,last=pack(split,start,self.norm,self.geo[split])
        base,last=normalized(base,last,self.stats)
        h=(history(split,start)-self.norm["mean"])/self.norm["std"]
        bounds={mode:head.bounds(base,last,h,torch.ones_like(self.template) if mode=="rich" else self.template)
                for mode,head in self.heads.items()}
        bounds.update(previous_temporal=old["quantile_last"],previous_2d=old["reference_quantile_last"],spread32=old["reference_spread32"])
        return p,bounds


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    predictor=PREDICTOR/"augmented_diffusion/model.pt"
    point_hash=digest(predictor)
    old_frozen=json.loads((PREVIOUS/"heads_frozen.json").read_text())["artifacts"]
    dump(OUT/"protocol.json",dict(time=time.time(),source_sha256=digest(Path(__file__)),
         module_sha256=digest(ROOT/"structured_quantile_uncertainty.py"),
         replay_sha256=digest(ROOT/"structured_interval_replay.py"),predictor_sha256=point_hash,
         modes=MODES,steps=2000,batch_size=4,seed=42,
         architecture="Full-history40->4per-lead context,last48->8per-lead features,2coordinates; two24channel3D convolutions,second spatial dilation2.",
         template="Training-only per-channel/lead/pixel residual RMS, floor0.01; rescales pinball loss and decoded bounds. No sampled teacher/distillation.",
         ablations="rich unscaled pinball; template spatial normalization; template_base removes denoising features with identical allocated parameters.",
         fit_files=[r["file"] for r in FIT_RECORDS],fit_frames=[200,400],predictor_fit_frames=[0,200],
         calibration_files=[r["file"] for r in manifest()["files"]["calibration"]],calibration_frames=[0,200],
         strategies=STRATEGIES,nominal=.9,
         separable="q_lead*q_patch/q_global followed by revealed-block global90% correction; q updated geometrically after complete20 reveal, step0.5.",
         target="Reduce regressions in score,width,frame/patch coverage and risk ranking, audit byRe/channel/trajectory; no claim of universal success in advance.",
         selection="All candidates, rules and endpoints fixed before replay; no development/test fitting, folds, distillation, extra training seeds.",
         limits="Exploratory, all data historically reused; no strict conditional or joint coverage guarantee. Fixed pointMSE cannot improve while predictor is frozen."))
    norm=normalization()
    train(norm)
    frozen=json.loads((OUT/"heads_frozen.json").read_text())["artifacts"]
    replay(OUT,Provider(norm),NAMES)
    assert digest(predictor)==point_hash
    assert all(digest(OUT/name)==sha for name,sha in frozen.items())
    assert all(digest(PREVIOUS/name)==sha for name,sha in old_frozen.items())
    dump(OUT/"complete.json",dict(time=time.time(),predictor_and_heads_unchanged=True))


if __name__=="__main__":
    main()
