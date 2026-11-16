"""Matched cue ablations with an asymmetric tail head; native4steps frozen."""

import json
import math
import time
from pathlib import Path

import torch

from clean_reliability_common import ROOT,configure,digest,dump
from experiment_reliability_transport_e12 import FIT_RECORDS,FIT_STARTS,PREDICTOR,geometry,history,target,normalization
from experiment_structured_quantile_uncertainty import OUT as HEAD_SOURCE,Provider as FrozenProvider
from experiment_temporal_error_uncertainty import pack,normalized
from experiment_fourstep_denoising_risk_cues import OUT as TRACE_SOURCE,change_cache
from infer_structured_fourstep_uncertainty import StructuredFourStepPredictor
from fourstep_denoising_risk_cues import cue_inputs
from fourstep_cue_tail_adapter import CueTailAdapter
from causal_shape_calibration import initial_state
from experiment_continue_rich_uncertainty import controller,MODES
from replay_conditional_risk import replay


OUT=ROOT/"runs/fourstep_cue_tails_20261109"
CUES=dict(asym_none="none",asym_history="history_change",asym_denoising="denoising_change")
NAMES=("rich",*CUES)


def train(norm):
    forecaster=StructuredFourStepPredictor(geometry(FIT_RECORDS))
    inputs={name:[] for name in CUES}
    lowers,uppers,labels=[],[],[]
    with torch.no_grad():
        for start in FIT_STARTS:
            p,base,last=pack("fit",start,norm,geometry(FIT_RECORDS))
            base,last=normalized(base,last,forecaster.stats)
            h=(history("fit",start)-norm["mean"])/norm["std"]
            lo,hi=forecaster.head.bounds(base,last,h,torch.ones_like(p[:1]))
            change=change_cache("fit",start)
            for name,cue in CUES.items():
                inputs[name].append(cue_inputs(h,p,lo,hi,change,cue,True))
            lowers.append(lo)
            uppers.append(hi)
            labels.append((target("fit",start)-norm["mean"])/norm["std"]-p)
        inputs={name:torch.cat(values) for name,values in inputs.items()}
        lower,upper,residual=[torch.cat(values) for values in (lowers,uppers,labels)]
    del forecaster,lowers,uppers,labels
    initial=[]
    for name in CUES:
        torch.manual_seed(42)
        head=CueTailAdapter().cuda()
        initial.append({key:value.clone() for key,value in head.state_dict().items()})
        optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-5)
        logs=[]
        begin=time.time()
        x=inputs[name]
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
                print("TRAIN",name,logs[-1],flush=True)
        torch.save(dict(model=head.state_dict(),cue=CUES[name],seed=42,steps=2000),OUT/(name+".pt"))
        dump(OUT/(name+"_training.json"),logs)
    assert all(torch.equal(initial[0][key],value) for state in initial[1:] for key,value in state.items())
    dump(OUT/"heads_frozen.json",dict(time=time.time(),parameters_each=13124,
        matched_initialization=True,examples=len(residual),
        artifacts={name+".pt":digest(OUT/(name+".pt")) for name in CUES}))


class Provider:
    def __init__(self,norm):
        self.norm=norm
        self.frozen=FrozenProvider(norm)
        self.heads={}
        for name in CUES:
            head=CueTailAdapter().cuda().eval().requires_grad_(False)
            head.load_state_dict(torch.load(OUT/(name+".pt"),map_location="cuda",weights_only=False)["model"])
            self.heads[name]=head

    @torch.no_grad()
    def bounds(self,split,start):
        p,bounds=self.frozen.bounds(split,start)
        lo,hi=bounds["rich"]
        h=(history(split,start)-self.norm["mean"])/self.norm["std"]
        change=change_cache(split,start)
        return p,dict(rich=(lo,hi),**{name:head.bounds(cue_inputs(h,p,lo,hi,change,CUES[name],start>0),lo,hi) for name,head in self.heads.items()})


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    frozen=json.loads((HEAD_SOURCE/"heads_frozen.json").read_text())["artifacts"]
    predictor_sha=digest(PREDICTOR/"augmented_diffusion/model.pt")
    trace_protocol=json.loads((TRACE_SOURCE/"protocol.json").read_text())
    assert trace_protocol["predictor_sha256"]==predictor_sha
    assert digest(ROOT/"fourstep_denoising_risk_cues.py")==trace_protocol["module_sha256"]
    dump(OUT/"protocol.json",dict(time=time.time(),source_sha256=digest(Path(__file__)),
        module_sha256=digest(ROOT/"fourstep_cue_tail_adapter.py"),predictor_sha256=predictor_sha,
        trace_audit_sha256=digest(TRACE_SOURCE/"change_cache_audit.json"),head_sha256=frozen,
        heads=NAMES,modes=MODES,cues=CUES,fit_files=[r["file"] for r in FIT_RECORDS],fit_frames=[200,400],
        architecture="Same13124parameter independent lower/upper scale head; rawrich anchor frozen; no change to pointP.",
        cue_description="Zeros,observedhistorylastincrement,or lastactualclean-outputdifferencefromoriginal99,66,33,0. All4callsretained,notpredictionvariance; reuseauditedfloat16cuecache.",
        training="Sameinitialization,seed42,batches,2000steps,batch4,AdamW,LR3e-4cosine1e-6,actual05/95pinball. No distillation,folds,new trainingseedsor currenttruthfeatures.",
        calibration="Independent10trajectories,recomputed after heads freeze. Same two causal controllers,only entire20feedback.",
        limits="Combination may not improve; preserve no-cue and history controls. Historically reused exploratory data,no fresh independent confirmation or formal coverage guarantee."))
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
    dump(OUT/"complete.json",dict(time=time.time(),all_original_models_unchanged=True))


if __name__=="__main__":
    main()
