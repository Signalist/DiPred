"""Matched continuation controls for the meaning of predicted interval width.

Pinball loss targets quantiles. A weak Gaussian scale auxiliary is evaluated
about either the learned interval center or the fixed DiPred point forecast.
No architecture, point prediction, inference call count, or risk metric changes.
"""

import hashlib
import json
import math
import time
from pathlib import Path
from statistics import mean

import torch

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from fullblock_memory_risk_head import FullBlockMemoryRiskHead
from experiment_fullblock_memory_risk import OUT as MEMORY, HEAD_SOURCE, load
from temporal_mean_memory_control import training_data
from experiment_reliability_transport_e12 import FIT_RECORDS, PREDICTOR, normalization, geometry, history, target
from experiment_temporal_error_uncertainty import pack, normalized
from experiment_direct_error_uncertainty import cached, metrics
from causal_shape_calibration import initial_state
from experiment_continue_rich_uncertainty import MODES, controller
from conditional_online_risk import risk_features
from reliability_risk_transport import base_features
from evaluate_reliability_transport_e12 import ranking, summarize
from diagnose_train_distribution_drift import OUT as PILOT_DATA
from pilot_fullblock_memory_risk import OUT as KNOWN_MEMORY


OUT = ROOT / "runs/point_error_scale_alignment_20261109"
OBJECTIVES = ("quantile", "centered_nll", "point_nll")
NAMES = ("identity", *OBJECTIVES)


def new_head(scale):
    head = FullBlockMemoryRiskHead("full", scale).cuda().eval()
    head.load_state_dict(load(MEMORY / "full.pt")["model"])
    return head


def loss_terms(head, objective, base, last, history, memory, residual, available):
    lower, upper = head.bounds(base, last, history, memory, available)
    e_lower, e_upper = residual-lower, residual-upper
    quantile = (torch.maximum(.05*e_lower, -.95*e_lower)
                + torch.maximum(.95*e_upper, -.05*e_upper)).mean()
    radius = (.5*(upper-lower)).clamp_min(1e-4)
    if objective == "quantile":
        return quantile, radius.new_zeros(()), radius
    # Keep direct midpoint gradients identical in both auxiliary controls.
    # Shared representation gradients may still change their fitted midpoint.
    location = .5*(lower+upper).detach() if objective == "centered_nll" else 0
    error = residual-location
    auxiliary = (torch.log(radius/head.memory_scale) + .5*(error/radius).square()).mean()
    return quantile, auxiliary, radius


def train(norm):
    base, last, h, memory, residual = training_data(norm)
    scale = load(MEMORY / "statistics.pt")["memory_scale"]
    schedules, coefficients = {}, {}
    # Fixed before looking at any pilot: exactly 10% of the initial radius-
    # gradient L2 norm, measured on 32 evenly spaced original fitting examples.
    probe = torch.linspace(0,818,32,device="cuda").round().long()
    for objective in OBJECTIVES:
        torch.manual_seed(42)
        head = new_head(scale)
        if objective == "quantile":
            weight, balance = 0.0, None
        else:
            q, nll, radius = loss_terms(head, objective, base[probe], last[probe], h[probe],
                                        memory[probe], residual[probe], True)
            q_gradient = torch.autograd.grad(q, radius, retain_graph=True, allow_unused=True)[0]
            # lower/upper precede radius in the graph, so differentiate pinball
            # w.r.t. the bounds, then use d(lower,upper)/d(radius)=(-1,+1).
            if q_gradient is None:
                lower, upper = head.bounds(base[probe], last[probe], h[probe], memory[probe], True)
                left, right = residual[probe]-lower, residual[probe]-upper
                pinball = (torch.maximum(.05*left,-.95*left)+torch.maximum(.95*right,-.05*right)).mean()
                grad_lower, grad_upper = torch.autograd.grad(pinball,(lower,upper))
                q_norm = (grad_upper-grad_lower).norm()
            else:
                q_norm = q_gradient.norm()
            nll_norm = torch.autograd.grad(nll, radius)[0].norm()
            assert q_norm > 0 and nll_norm > 0 and torch.isfinite(nll_norm)
            weight = float(.1*q_norm/nll_norm)
            balance = dict(initial_quantile_radius_gradient_l2=float(q_norm),
                           initial_auxiliary_radius_gradient_l2=float(nll_norm), target_gradient_ratio=.1)
        coefficients[objective] = dict(weight=weight, balance=balance)
        sampler = torch.Generator(device="cuda").manual_seed(42)
        optimizer = torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-5)
        schedule = hashlib.sha256()
        logs, begin = [], time.time()
        for step in range(1,2001):
            ix = torch.randint(819,(4,),device="cuda",generator=sampler)
            available = (torch.rand(4,device="cuda",generator=sampler)>=.2).float()
            schedule.update(ix.cpu().numpy().tobytes())
            schedule.update(available.cpu().numpy().tobytes())
            optimizer.param_groups[0]["lr"] = 1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/2000))
            optimizer.zero_grad(set_to_none=True)
            q, nll, _ = loss_terms(head,objective,base[ix],last[ix],h[ix],memory[ix],residual[ix],available)
            loss = q+weight*nll
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(head.parameters(),1))
            optimizer.step()
            if step==1 or step%500==0:
                logs.append(dict(step=step,loss=float(loss),quantile=float(q),auxiliary=float(nll),
                                 auxiliary_weight=weight,elapsed=time.time()-begin))
                print("ERROR_SCALE_TRAIN",objective,logs[-1],flush=True)
        schedules[objective] = schedule.hexdigest()
        torch.save(dict(model=head.state_dict(),objective=objective,auxiliary_weight=weight,seed=42,
                        examples=819,additional_steps=2000),OUT/(objective+".pt"))
        dump(OUT/(objective+"_training.json"),logs)
    old = json.loads((MEMORY/"heads_frozen.json").read_text())
    assert len(set(schedules.values()))==1 and schedules["quantile"]==old["minibatches_and_availability_masks_sha256"]["full"]
    dump(OUT/"heads_frozen.json",dict(time=time.time(),parameters=48340,fit_pairs=819,
        minibatches_and_availability_masks_sha256=schedules,coefficients=coefficients,
        artifacts={name:digest(OUT/(name+".pt")) for name in OBJECTIVES}))


class Provider:
    def __init__(self,norm):
        self.norm = norm
        self.stats = load(HEAD_SOURCE/"statistics.pt")["statistics"]
        self.scale = load(MEMORY/"statistics.pt")["memory_scale"]
        self.geo = {split:geometry(manifest()["files"][split]) for split in ("calibration","development","test")}
        self.heads = {"identity":new_head(self.scale).requires_grad_(False)}
        for name in OBJECTIVES:
            head = new_head(self.scale).requires_grad_(False)
            head.load_state_dict(load(OUT/(name+".pt"))["model"])
            self.heads[name] = head

    @torch.no_grad()
    def bounds(self,split,start):
        point,base,last = pack(split,start,self.norm,self.geo[split])
        base,last = normalized(base,last,self.stats)
        h = (history(split,start)-self.norm["mean"])/self.norm["std"]
        previous = None if start==0 else cached(split,start-20)["prediction"]
        memory = torch.zeros_like(h) if previous is None else h-previous
        return point,{name:head.bounds(base,last,h,memory,previous is not None) for name,head in self.heads.items()}


@torch.no_grad()
def calibrate_and_pilot(norm):
    provider = Provider(norm)
    scores = {name:[] for name in NAMES}
    for start in range(0,161,20):
        point,bounds = provider.bounds("calibration",start)
        residual = (target("calibration",start)-norm["mean"])/norm["std"]-point
        for name,(lo,hi) in bounds.items():
            scores[name].append(((residual-.5*(lo+hi)).abs()/(.5*(hi-lo)).clamp_min(1e-6)).cpu())
    initial = {name:{"separable_integral":initial_state(torch.cat(values).cuda(),"separable_integral")}
               for name,values in scores.items()}
    old = load(MEMORY/"calibration.pt")["full"]["separable_integral"]
    assert all(torch.equal(value,old[key]) for key,value in initial["identity"]["separable_integral"].items())
    torch.save({name:{mode:{key:value.cpu() for key,value in state.items()} for mode,state in modes.items()}
                for name,modes in initial.items()},OUT/"calibration.pt")
    dump(OUT/"calibration_locked.json",dict(time=time.time(),sha256=digest(OUT/"calibration.pt"),identity_exact=True))
    protocol = json.loads((PILOT_DATA/"protocol.json").read_text())
    records = protocol["records"]
    geo = geometry(records)
    states = {name:{mode:controller(mode,initial[name]["separable_integral"],len(records)) for mode in MODES} for name in NAMES}
    rows = []
    for start in protocol["starts"]:
        known = load(KNOWN_MEMORY/"known"/(str(start)+".pt"))
        observed = load(PILOT_DATA/"known"/(str(start)+".pt"))["observed"]
        point = known["prediction"]
        h = (observed-norm["mean"])/norm["std"]
        base = base_features(observed,point*norm["std"]+norm["mean"],norm["mean"],norm["std"],geo)
        base,last = normalized(base,known["last"].float(),provider.stats)
        bounds = {name:head.bounds(base,last,h,known["previous_error"],known["available"]) for name,head in provider.heads.items()}
        issued = {name:{mode:state.issue(*bounds[name],risk_features(h,point,.5*(bounds[name][1]-bounds[name][0])))
                       for mode,state in modes.items()} for name,modes in states.items()}
        with (OUT/"pilot_events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event="issued",start=start,time=time.time()))+"\n")
        residual = load(PILOT_DATA/"labels"/(str(start)+".pt"))
        with (OUT/"pilot_events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event="revealed",start=start,time=time.time()))+"\n")
        mse = (residual*norm["std"]).square().mean((2,3,4))
        for name,modes in issued.items():
            raw = {"raw_"+key:value for key,value in ranking(.5*(bounds[name][1]-bounds[name][0]),residual.abs()).items()}
            for mode,(lo,hi) in modes.items():
                values = dict(**metrics(lo,hi,residual),**ranking(.5*(hi-lo),residual.abs()),**raw)
                for i,record in enumerate(records):
                    for c,channel in enumerate(("u","v")):
                        rows.append(dict(kind="diffusion",split="train_time_pilot",group=record["group"],file=record["file"],start=start,
                            period="later" if start>=700 else "post_cold" if start>=600 else "prefix",channel=channel,map=name,variant=mode,
                            mse=float(mse[i,c]),**{key:float(value[i,c]) for key,value in values.items()}))
                states[name][mode].reveal(residual)
        print("ERROR_SCALE_PILOT",start,flush=True)
    dump(OUT/"per_case.json",rows)
    dump(OUT/"summary.json",summarize(rows))
    result = {}
    for mode in MODES:
        values = {}
        for name in NAMES:
            groups = {}
            for row in rows:
                if row["map"]==name and row["variant"]==mode and row["start"]>=600:
                    groups.setdefault((row["group"],row["channel"]),[]).append(row)
            values[name] = {metric:mean(mean(r[metric] for r in group) for group in groups.values())
                            for metric in ("standardized_interval_score","relative_aurc","top10_energy","worst_frame_coverage",
                                           "worst_patch_coverage","standardized_width")}
            values[name]["coverage_error"] = mean(abs(mean(r["coverage"] for r in group)-.9) for group in groups.values())
        passed = []
        for name in ("centered_nll","point_nll"):
            new,old = values[name],values["quantile"]
            if (new["standardized_interval_score"]<old["standardized_interval_score"]-1e-6
                    and new["relative_aurc"]<old["relative_aurc"]-1e-6
                    and new["top10_energy"]>=old["top10_energy"]-1e-6
                    and new["coverage_error"]<=old["coverage_error"]+1e-6):
                passed.append(name)
        result[mode] = dict(values=values,passed=passed)
    dump(OUT/"gate.json",dict(results=result,proceed_full_evaluation=any(row["passed"] for row in result.values())))
    print(json.dumps(result),flush=True)


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    paths = [MEMORY/"full.pt",MEMORY/"statistics.pt",MEMORY/"calibration.pt",PREDICTOR/"augmented_diffusion/model.pt"]
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in paths}
    ids = {r["file"] for r in FIT_RECORDS}
    held = {r["file"] for split,records in manifest()["files"].items() if split!="train" for r in records}
    assert not ids & held
    dump(OUT/"protocol.json",dict(time=time.time(),source_sha256=digest(Path(__file__)),frozen_artifacts=hashes,
        predictor_sha256=digest(paths[-1]),names=NAMES,modes=MODES,
        hypotheses="Quantile radius orders R-mid while primaryriskranking targets R. Test Gaussian scale loss about detachedmid versus originalfixedP residual. Neither assumes Gaussian future physics nor substitutes centered-error scoring for primaryRrisk.",
        training="Same48340parameters,full-memoryinitialweights,819originaltrainpairs200:600,2000additionalupdates,seed42,batch4,AdamW3e-4->1e-6,clip1,same20%availabilitydropout. Purepinball matchedcontinuedcontrol. No extra inference output/parameters/calls.",
        auxiliary_weight="One prespecified10% initialradiusgradientL2 ratio,computed separately forcentered/pointtargets on32evenlyspaced originalfitexamples. Freezeweightbeforeupdates; no pilot/evalselection or gainscan. Midpoint detached forbothauxiliarycontrols.",
        calibration="Allheadsfreezefirst,recomputeindependent10IDinitialization. Pointpredictor and trainnormalization fixed. Allcurrenttargets read after allissues in pilot.",
        pilot_gate="44originaltrainID queries600..760 after580cold. Centered/pointauxiliary mustloweractualScore and primaryR-AURC versusmatchedpurepinball,withoutworse top10energy or meanabsoluteRe/channelcoverageerror,tolerance1e-6. Retain all4heads/2controllers ifproceeding.",
        invariants="Native99,66,33,0,joint20to20,actualsavedpreviousP,fullwindowfeedback,alloriginal8metrics/96cells and strongreferences unchanged. No folds,distillation,seedscan or heldoutidentity fitting.",
        limits="Historicallyreusedtrainpilot. Per-radiusgradient normalization is not identical parametergradient/FLOP matching. Diagnostic doesnotprove the testAURCgap has this cause. Full-line goal remains unachieved."))
    norm = normalization()
    train(norm)
    calibrate_and_pilot(norm)
    assert all(digest(ROOT/path)==sha for path,sha in hashes.items())
    dump(OUT/"complete.json",dict(time=time.time(),frozen_predictor_and_source_heads_unchanged=True,
                                 development_test_not_evaluated=True))
    print("POINT_ERROR_SCALE_ALIGNMENT_COMPLETE",flush=True)


if __name__=="__main__":
    main()
