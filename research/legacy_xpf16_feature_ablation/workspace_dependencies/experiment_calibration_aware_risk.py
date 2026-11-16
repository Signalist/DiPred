"""Fit the existing risk head under causal online calibration, not raw bounds.

All three controls see the same trajectory episodes, optimizer steps, dropout
masks and true frozen-predictor residuals. Online state gradients are stopped
after feedback; current issued bounds remain differentiable during offline fit.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from clean_reliability_common import ROOT,configure,dump,digest,manifest
from experiment_reliability_transport_e12 import FIT_RECORDS,FIT_STARTS,PREDICTOR,normalization,geometry,history
from experiment_extended_train_error_adapters import LATE_STARTS
from experiment_fullblock_memory_risk import OUT as MEMORY,SOURCE as EXPANDED,HEAD_SOURCE,load
from experiment_point_error_scale_alignment import OUT as PREVIOUS
from fullblock_memory_risk_head import FullBlockMemoryRiskHead
from temporal_mean_memory_control import training_data
from experiment_direct_error_uncertainty import cached
from experiment_temporal_error_uncertainty import pack,normalized
from experiment_continue_rich_uncertainty import controller,MODES
from conditional_online_risk import risk_features
from calibration_aware_risk import pinball_field,unit_initial,group_scales,local_tail_loss,episode_indices
from risk_head_pilot_protocol import calibrate,pilot


OUT = ROOT/"runs/calibration_aware_risk_training_20261110"
OBJECTIVES = ("sequence_q","issued_q","issued_tail")
NAMES = ("frozen_q",*OBJECTIVES)


def new_head(scale):
    head = FullBlockMemoryRiskHead("full",scale).cuda().eval()
    head.load_state_dict(load(PREVIOUS/"quantile.pt")["model"])
    return head


def train(norm):
    base,last,h,memory,residual = training_data(norm)
    # These are actual cached native four-call point predictions, not values
    # reconstructed from either clipped features or the current future labels.
    point = torch.cat([cached("fit",start)["prediction"] for start in FIT_STARTS]
                      +[load(EXPANDED/"later_data"/(str(start)+".pt"))["prediction"] for start in LATE_STARTS])
    assert len(point)==819
    scales = group_scales(residual)
    scale = load(MEMORY/"statistics.pt")["memory_scale"]
    torch.save({key:value.cpu() for key,value in scales.items()},OUT/"loss_scales.pt")
    probe = torch.linspace(0,818,32,device="cuda").round().long()
    head = new_head(scale)
    lower,upper = head.bounds(base[probe],last[probe],h[probe],memory[probe],True)
    q = pinball_field(lower,upper,residual[probe]).mean()
    tail = local_tail_loss(pinball_field(lower,upper,residual[probe]),scales)
    q_grad = torch.autograd.grad(q,(lower,upper),retain_graph=True)
    tail_grad = torch.autograd.grad(tail,(lower,upper))
    q_norm = (q_grad[1]-q_grad[0]).norm()
    tail_norm = (tail_grad[1]-tail_grad[0]).norm()
    tail_weight = float(.1*q_norm/tail_norm)
    assert tail_weight>0 and math.isfinite(tail_weight)
    dump(OUT/"loss_balance.json",dict(tail_weight=tail_weight,probe_indices=probe.cpu().tolist(),
        initial_raw_quantile_radius_gradient_l2=float(q_norm),initial_local_tail_radius_gradient_l2=float(tail_norm),
        target_gradient_ratio=.1,prototype="Cold unitcalibration equals rawbounds; all32probesoriginalfitexamples."))
    del head,lower,upper,q,tail,q_grad,tail_grad
    schedules = {}
    for objective in OBJECTIVES:
        torch.manual_seed(42)
        head = new_head(scale)
        optimizer = torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-5)
        sampler = torch.Generator(device="cuda").manual_seed(42)
        schedule = hashlib.sha256()
        logs,begin = [],time.time()
        state = None
        for step in range(1,2001):
            window = (step-1)%9
            if window==0:
                episodes = torch.randint(91,(4,),device="cuda",generator=sampler)
                state = controller("scale_fastcal",unit_initial(base),4)
            ix = episode_indices(episodes,window)
            available = (torch.rand(4,device="cuda",generator=sampler)>=.2).float()
            schedule.update(episodes.cpu().numpy().tobytes())
            schedule.update(ix.cpu().numpy().tobytes())
            schedule.update(available.cpu().numpy().tobytes())
            assert state.revealed_windows==window
            optimizer.param_groups[0]["lr"] = 1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/2000))
            optimizer.zero_grad(set_to_none=True)
            lower,upper = head.bounds(base[ix],last[ix],h[ix],memory[ix],available)
            lo,hi = state.issue(lower,upper,risk_features(h[ix],point[ix],.5*(upper-lower)))
            # Keep only numerical issued values for future feedback. No
            # gradients run through earlier truth or the online state updates.
            state.pending = tuple(value.detach() for value in state.pending)
            with (OUT/"training_events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event="issued",objective=objective,step=step,window=window,
                    episode_ids=episodes.cpu().tolist(),fit_indices=ix.cpu().tolist(),prior_complete_windows=state.revealed_windows,time=time.time()))+"\n")
            # Offline packets already contain labels, but labels are selected
            # for loss/feedback only after current features and bounds exist.
            truth = residual[ix]
            with (OUT/"training_events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event="supervision_selected",objective=objective,step=step,time=time.time()))+"\n")
            raw = pinball_field(lower,upper,truth).mean()
            issued_field = pinball_field(lo,hi,truth)
            issued = issued_field.mean()
            local = local_tail_loss(issued_field,scales) if objective=="issued_tail" else issued.new_zeros(())
            loss = raw if objective=="sequence_q" else issued+(tail_weight*local if objective=="issued_tail" else 0)
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(head.parameters(),1))
            optimizer.step()
            with torch.no_grad():
                state.reveal(truth)
            assert state.pending is None and not state.beta.requires_grad and not state.shape.requires_grad
            with (OUT/"training_events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event="feedback_applied",objective=objective,step=step,
                                             completed_windows=state.revealed_windows,time=time.time()))+"\n")
            if step==1 or step%500==0:
                logs.append(dict(step=step,loss=float(loss),raw_quantile=float(raw),issued_quantile=float(issued),
                                 local_tail=float(local),tail_weight=tail_weight if objective=="issued_tail" else 0,
                                 elapsed=time.time()-begin))
                print("CALIBRATION_AWARE_TRAIN",objective,logs[-1],flush=True)
        schedules[objective] = schedule.hexdigest()
        torch.save(dict(model=head.state_dict(),objective=objective,seed=42,additional_steps=2000,
                        examples=819,tail_weight=tail_weight if objective=="issued_tail" else 0),OUT/(objective+".pt"))
        dump(OUT/(objective+"_training.json"),logs)
    assert len(set(schedules.values()))==1
    dump(OUT/"heads_frozen.json",dict(time=time.time(),parameters=48340,fit_pairs=819,
        episode_indices_and_availability_masks_sha256=schedules,
        independent_calibration_labels_not_used_in_fitting=True,
        artifacts={name:digest(OUT/(name+".pt")) for name in OBJECTIVES}))


class Provider:
    def __init__(self,norm):
        self.norm = norm
        self.stats = load(HEAD_SOURCE/"statistics.pt")["statistics"]
        self.scale = load(MEMORY/"statistics.pt")["memory_scale"]
        self.geo = {split:geometry(manifest()["files"][split]) for split in ("calibration","development","test")}
        self.heads = {"frozen_q":new_head(self.scale).requires_grad_(False)}
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


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    paths = [PREVIOUS/"quantile.pt",PREVIOUS/"calibration.pt",MEMORY/"statistics.pt",PREDICTOR/"augmented_diffusion/model.pt"]
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in paths}
    held = {r["file"] for split,records in manifest()["files"].items() if split!="train" for r in records}
    assert not {r["file"] for r in FIT_RECORDS}&held
    dump(OUT/"protocol.json",dict(time=time.time(),source_sha256=digest(Path(__file__)),
        helper_sha256={name:digest(ROOT/name) for name in ("calibration_aware_risk.py","risk_head_pilot_protocol.py")},
        frozen_artifacts=hashes,predictor_sha256=digest(paths[-1]),names=NAMES,objectives=OBJECTIVES,modes=MODES,
        training="819originaltrainpairs200:600,91trajectorysegmentsx9completewindows. Sameinitial48340parameterbesthead,seed42,batch4,2000windowupdates,AdamW3e-4->1e-6,clip1,20%memoryavailabilitydropout. Allthreearmsmatch episode/indices/masks exactly. Chronologicalsampler differs fromprevious randomwindowcontrol; no claimofexactprevioussampler match.",
        online_simulator="Originalscale_fastcal,unitcoldinitializationindependentofallcalibrationlabels,resetforeachsampledepisode9windows. Currentknownfeatures->issue->selectsupervision->offlineupdate->fulltruthfeedback. Stateupdatesdetached,noBPTT; gradientthroughcurrentknownconditionalshape retained. Duringofflinefitting networkweightschange betweenwindows,unlike frozen deployment; statesusevalues actuallyissuedatthoseweights.",
        objective_design="sequence_q rawpinballcontrol;issued_q actualissuedpinball=intervalscore/20;issued_tail adds topquarter normalized8x8patch/fullframe issuedscore. Patch/leadRMS from originalfitlabels only,loss-only; initial32fitprobe radiusgradientratio10% fixesweight,no scan. No formalcoverage or standardconditionalquantile claimforgroup-tail objective.",
        calibration="Fitheadswithoutreadingindependent10IDcalibrationlabels orparameters. Afterallheadsfreeze,recompute10IDdeploymentcalibration. Frozen_qmustmatchpreviousquantileinitialization exactly.",
        pilot_gate="44originaltrainIDqueries600..760after580cold. issued_q/issued_tail mustloweractualScore andimproveworstpatchcoverage by1e-6 versusmatchedsequence_q,withoutworseWidth,AURC,top10energy ormeanabsoluteRe/channelcoverageerror by1e-6. Retain allheads/controllers/primarymetricsifproceeding.",
        primary_scope="Originalfourcalls99,66,33,0,joint20to20,frozenpoint,all8metrics/96cells/tolerance1e-6,all47strongreferencepairs fromlastfullstudy. No dev/testfitting,folds,distillation,multiseeds,inferencemoduleaddition orgain tuning toknown developmentfailures.",
        literature="https://arxiv.org/abs/2608.28515 distinguishes spatialfraction versuscellwisecoverage; ourcausaltraining surrogate doesnotinheritits exchangeability/splitconformal guarantees.",
        limits="Historicaldatareused. Originalfittingpacketscontainlabels,excludedfromfeatureinputs. Frozenq'sbettermeanScore andearly2/96do notestablishfull-linegoal; strongreferencefailuresremain."))
    norm = normalization()
    train(norm)
    provider = Provider(norm)
    reference = load(PREVIOUS/"calibration.pt")["quantile"]["separable_integral"]
    initial = calibrate(provider,OUT,NAMES,"frozen_q",reference)
    values = pilot(provider,OUT,NAMES,initial)
    gate = {}
    for mode,methods in values.items():
        old = methods["sequence_q"]
        passed = []
        for name in ("issued_q","issued_tail"):
            new = methods[name]
            if (new["standardized_interval_score"]<old["standardized_interval_score"]-1e-6
                    and new["worst_patch_coverage"]>old["worst_patch_coverage"]+1e-6
                    and new["standardized_width"]<=old["standardized_width"]+1e-6
                    and new["relative_aurc"]<=old["relative_aurc"]+1e-6
                    and new["top10_energy"]>=old["top10_energy"]-1e-6
                    and new["coverage_error"]<=old["coverage_error"]+1e-6):
                passed.append(name)
        gate[mode] = dict(values=methods,passed=passed)
    dump(OUT/"gate.json",dict(results=gate,proceed_full_evaluation=any(row["passed"] for row in gate.values())))
    assert all(digest(ROOT/path)==sha for path,sha in hashes.items())
    dump(OUT/"complete.json",dict(time=time.time(),original_artifacts_unchanged=True,development_test_not_evaluated=True))
    print(json.dumps(gate),flush=True)
    print("CALIBRATION_AWARE_RISK_TRAINING_COMPLETE",flush=True)


if __name__=="__main__":
    main()
