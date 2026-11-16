"""Matched risk heads with actual denoising changes, all four calls retained."""

import json
import math
import time
from pathlib import Path

import torch

from clean_reliability_common import ROOT,configure,digest,dump
from experiment_reliability_transport_e12 import FIT_RECORDS,FIT_STARTS,PREDICTOR,geometry,history,target,normalization
from experiment_structured_quantile_uncertainty import OUT as HEAD_SOURCE,Provider as FrozenProvider
from experiment_temporal_error_uncertainty import pack,normalized
from experiment_direct_error_uncertainty import cached
from experiment_physical20_augmented_dynamics import new_model
from infer_structured_fourstep_uncertainty import StructuredFourStepPredictor
from fourstep_denoising_risk_cues import forecast_last_change,cue_inputs,DenoisingCueAdapter
from causal_shape_calibration import initial_state
from experiment_continue_rich_uncertainty import controller,MODES
from replay_conditional_risk import replay


OUT=ROOT/"runs/fourstep_denoising_risk_cues_20261109"
CUES=("none","history_change","denoising_change")
NAMES=("rich",*CUES)


def change_cache(split,start):
    return torch.load(OUT/"changes"/("%s_%d.pt"%(split,start)),map_location="cuda",weights_only=False)["change"].float()


@torch.no_grad()
def cache_changes(norm):
    (OUT/"changes").mkdir()
    model=new_model("augmented_diffusion").eval().requires_grad_(False)
    model.load_state_dict(torch.load(PREDICTOR/"augmented_diffusion/model.pt",map_location="cuda",weights_only=False)["model"])
    checks=[]
    for split,starts in (("fit",FIT_STARTS),("calibration",range(0,161,20)),
                         ("development",range(0,761,20)),("test",range(0,761,20))):
        for start in starts:
            h=(history(split,start)-norm["mean"])/norm["std"]
            point,change,labels=forecast_last_change(model,h)
            assert torch.equal(point,cached(split,start)["prediction"])
            assert torch.isfinite(change.half()).all()
            torch.save(dict(change=change.half().cpu(),labels=labels),OUT/"changes"/("%s_%d.pt"%(split,start)))
            checks.append(dict(split=split,start=start,batch=len(h),labels=labels,
                point_exact=True,current_future_target_read=False))
        print("CACHE",split,flush=True)
    dump(OUT/"change_cache_audit.json",dict(actual_four_call_rollouts=len(checks),cases=checks))


def train(norm):
    forecaster=StructuredFourStepPredictor(geometry(FIT_RECORDS))
    inputs={cue:[] for cue in CUES}
    lowers,uppers,labels=[],[],[]
    with torch.no_grad():
        for start in FIT_STARTS:
            p,base,last=pack("fit",start,norm,geometry(FIT_RECORDS))
            base,last=normalized(base,last,forecaster.stats)
            h=(history("fit",start)-norm["mean"])/norm["std"]
            lo,hi=forecaster.head.bounds(base,last,h,torch.ones_like(p[:1]))
            change=change_cache("fit",start)
            for cue in CUES:
                inputs[cue].append(cue_inputs(h,p,lo,hi,change,cue,True))
            lowers.append(lo)
            uppers.append(hi)
            labels.append((target("fit",start)-norm["mean"])/norm["std"]-p)
        inputs={cue:torch.cat(values) for cue,values in inputs.items()}
        lower,upper,residual=[torch.cat(values) for values in (lowers,uppers,labels)]
    del forecaster,lowers,uppers,labels
    initial=[]
    for cue in CUES:
        torch.manual_seed(42)
        head=DenoisingCueAdapter().cuda()
        initial.append({key:value.clone() for key,value in head.state_dict().items()})
        optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-5)
        logs=[]
        begin=time.time()
        x=inputs[cue]
        for step in range(1,2001):
            ix=torch.randint(len(x),(4,),device="cuda")
            rate=1e-6+(3e-4-1e-6)*0.5*(1+math.cos(math.pi*(step-1)/2000))
            for group in optimizer.param_groups:
                group["lr"]=rate
            optimizer.zero_grad(set_to_none=True)
            loss=head.loss(x[ix],lower[ix],upper[ix],residual[ix])
            assert torch.isfinite(loss)
            loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(head.parameters(),1)
            assert torch.isfinite(gradient)
            optimizer.step()
            if step==1 or step%500==0:
                logs.append(dict(step=step,loss=float(loss),elapsed=time.time()-begin))
                print("TRAIN",cue,logs[-1],flush=True)
        torch.save(dict(model=head.state_dict(),cue=cue,seed=42,steps=2000),OUT/(cue+".pt"))
        dump(OUT/(cue+"_training.json"),logs)
    assert all(torch.equal(initial[0][key],value) for state in initial[1:] for key,value in state.items())
    dump(OUT/"heads_frozen.json",dict(time=time.time(),parameters_each=13090,
        matched_initialization=True,examples=len(residual),
        artifacts={cue+".pt":digest(OUT/(cue+".pt")) for cue in CUES}))


class Provider:
    def __init__(self,norm):
        self.norm=norm
        self.frozen=FrozenProvider(norm)
        self.heads={}
        for cue in CUES:
            head=DenoisingCueAdapter().cuda().eval().requires_grad_(False)
            head.load_state_dict(torch.load(OUT/(cue+".pt"),map_location="cuda",weights_only=False)["model"])
            self.heads[cue]=head

    @torch.no_grad()
    def bounds(self,split,start):
        p,bounds=self.frozen.bounds(split,start)
        lo,hi=bounds["rich"]
        h=(history(split,start)-self.norm["mean"])/self.norm["std"]
        change=change_cache(split,start)
        return p,dict(rich=(lo,hi),**{cue:head.bounds(cue_inputs(h,p,lo,hi,change,cue,start>0),lo,hi) for cue,head in self.heads.items()})


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    frozen=json.loads((HEAD_SOURCE/"heads_frozen.json").read_text())["artifacts"]
    predictor_sha=digest(PREDICTOR/"augmented_diffusion/model.pt")
    dump(OUT/"protocol.json",dict(time=time.time(),source_sha256=digest(Path(__file__)),
        module_sha256=digest(ROOT/"fourstep_denoising_risk_cues.py"),predictor_sha256=predictor_sha,
        head_sha256=frozen,heads=NAMES,modes=MODES,
        fit_files=[r["file"] for r in FIT_RECORDS],fit_frames=[200,400],
        inference="Actual original99,66,33,0 all run; hook clean outputs,cacheabs(clean0-clean33)asfloat16. Every reissued point must exactly match frozenfour-callcache; no3-callbranch.",
        cues="Same2reservedslots: zeros,abs(lastobservedframe-prevobservedframe)broadcast20,or actual lastdenoising clean-output difference; log1p(signal/rawhalfwidth)clamp0..5.",
        training="All13090parameters,initialization,seed42,batches,2000steps,batch4,AdamW,LR3e-4cosine1e-6 matched. True prediction residual targets only, no distillation.",
        calibration="Separate10trajectories,recomputed after freeze; original two causal controllers and full20feedback. Existing cold-startgate retained equally.",
        interpretation="Denoising change is a deterministic inference-process cue,not automatically forecast variance or physical uncertainty.",
        limits="Historically reused exploratory trajectories, no folds,multi-training-seed sweep,future-labelfeatures,new point modelor formal coverage guarantee."))
    norm=normalization()
    cache_changes(norm)
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
    dump(OUT/"complete.json",dict(time=time.time(),all_original_models_unchanged=True))


if __name__=="__main__":
    main()
