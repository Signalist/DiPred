"""Same-network proper marginal-density objectives versus quantile training.

No extra inference modules or sampling paths. Full likelihood replaces, rather
than weakly augments, pinball loss. Both location and scale receive gradients.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from clean_reliability_common import ROOT,configure,dump,digest,manifest
from experiment_calibration_aware_risk import new_head,PREVIOUS,MEMORY,HEAD_SOURCE,load
from experiment_reliability_transport_e12 import FIT_RECORDS,PREDICTOR,normalization,geometry,history
from temporal_mean_memory_control import training_data
from experiment_direct_error_uncertainty import cached
from experiment_temporal_error_uncertainty import pack,normalized
from experiment_continue_rich_uncertainty import MODES
from calibration_aware_risk import pinball_field
from distribution_error_risk import density_loss,Q95
from risk_head_pilot_protocol import calibrate,pilot


OUT = ROOT/"runs/distribution_error_risk_20261110"
OBJECTIVES = ("density_quantile","density_gaussian","density_student4")
NAMES = ("density_reference",*OBJECTIVES)


def train(norm):
    base,last,h,memory,residual = training_data(norm)
    scale = load(MEMORY/"statistics.pt")["memory_scale"]
    probe = torch.linspace(0,818,32,device="cuda").round().long()
    schedules,balances,initial = {},{},{}
    for kind in OBJECTIVES:
        torch.manual_seed(42)
        head = new_head(scale)
        assert sum(p.numel() for p in head.parameters())==48340
        lower,upper = head.bounds(base[probe],last[probe],h[probe],memory[probe],True)
        initial[kind] = tuple(value.detach().clone() for value in (lower,upper))
        if kind=="density_quantile":
            coefficient,balance = 1.0,None
        else:
            q = pinball_field(lower,upper,residual[probe]).mean()
            nll = density_loss(lower,upper,residual[probe],kind)
            q_gradient = torch.autograd.grad(q,(lower,upper),retain_graph=True)
            nll_gradient = torch.autograd.grad(nll,(lower,upper))
            q_norm = (q_gradient[1]-q_gradient[0]).norm()
            nll_norm = (nll_gradient[1]-nll_gradient[0]).norm()
            coefficient = float(q_norm/nll_norm)
            assert math.isfinite(coefficient) and coefficient>0
            balance = dict(initial_quantile_radius_gradient_l2=float(q_norm),
                           initial_nll_radius_gradient_l2=float(nll_norm),target_initial_gradient_ratio=1.0)
        balances[kind] = dict(coefficient=coefficient,details=balance)
        sampler = torch.Generator(device="cuda").manual_seed(42)
        optimizer = torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-5)
        schedule = hashlib.sha256()
        logs,begin = [],time.time()
        for step in range(1,2001):
            ix = torch.randint(819,(4,),device="cuda",generator=sampler)
            available = (torch.rand(4,device="cuda",generator=sampler)>=.2).float()
            schedule.update(ix.cpu().numpy().tobytes())
            schedule.update(available.cpu().numpy().tobytes())
            optimizer.param_groups[0]["lr"] = 1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/2000))
            optimizer.zero_grad(set_to_none=True)
            lower,upper = head.bounds(base[ix],last[ix],h[ix],memory[ix],available)
            q = pinball_field(lower,upper,residual[ix]).mean()
            objective = q if kind=="density_quantile" else density_loss(lower,upper,residual[ix],kind)
            loss = coefficient*objective
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(head.parameters(),1))
            optimizer.step()
            if step==1 or step%500==0:
                logs.append(dict(step=step,loss=float(loss),unscaled_objective=float(objective),quantile_diagnostic=float(q),
                                 objective_coefficient=coefficient,elapsed=time.time()-begin))
                print("ERROR_DENSITY_TRAIN",kind,logs[-1],flush=True)
        schedules[kind] = schedule.hexdigest()
        torch.save(dict(model=head.state_dict(),objective=kind,objective_coefficient=coefficient,seed=42,
                        examples=819,additional_steps=2000,q95=Q95.get(kind)),OUT/(kind+".pt"))
        dump(OUT/(kind+"_training.json"),logs)
    assert len(set(schedules.values()))==1
    assert all(torch.equal(a,b) for name in OBJECTIVES[1:] for a,b in zip(initial[OBJECTIVES[0]],initial[name]))
    dump(OUT/"heads_frozen.json",dict(time=time.time(),parameters=48340,fit_pairs=819,additional_steps=2000,
        initial90percentbounds_identical=True,location_and_scale_both_trained=True,
        minibatches_and_availability_masks_sha256=schedules,coefficients=balances,q95=Q95,
        artifacts={name:digest(OUT/(name+".pt")) for name in OBJECTIVES}))


class Provider:
    def __init__(self,norm):
        self.norm = norm
        self.stats = load(HEAD_SOURCE/"statistics.pt")["statistics"]
        self.scale = load(MEMORY/"statistics.pt")["memory_scale"]
        self.geo = {split:geometry(manifest()["files"][split]) for split in ("calibration","development","test")}
        self.heads = {"density_reference":new_head(self.scale).requires_grad_(False)}
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
        helper_sha256={name:digest(ROOT/name) for name in ("distribution_error_risk.py","risk_head_pilot_protocol.py")},
        frozen_artifacts=hashes,predictor_sha256=digest(paths[-1]),names=NAMES,objectives=OBJECTIVES,modes=MODES,q95=Q95,
        rationale="Stopaddingtailpenalties aftertargetedgainsstillworsenjointmetrics. Compareproperpointwiseerror-densityfamilies versus twoendpointpinball underidenticalinferencearchitecture. This isnotweakauxiliaryGaussian-scale loss: fullnormalizedNLL replacespinball andtrainsbotherrorlocation andscale.",
        fit="Same48340parameters,bestcurrentquantileinitialweights,819originaltrainpairs200:600,seed42,batch4,2000updates,AdamW3e-4->1e-6,clip1,same20%memoryavailabilitymasksandexactrandomwindowminibatches. No addedpredictortraining.",
        families="Quantileordered90%bounds;Gaussianloc=midpoint,scale=radius/Normalq95;Student-tfixeddf4,loc=midpoint,scale=radius/Studentq95. All3initial90%boundsidentical. No degreesoffreedomscan. Midpointdescribeserrorlocation,notachangetofixedP. No jointspatiotemporal or physicalrandomness claim.",
        normalization="Entirelikelihood multipliedbyfixedpositivecoefficientsettinginitialradiusgradientL2to100%ofpinball on32evenlyspacedfitexamples. Freeze beforeupdates,no pilot/dev/testcoefficientfit. ThisdoesnotmatchallparametergradientsorFLOPs exactly.",
        calibration="Afterallheadsfreeze,recomputeindependent10IDdeploymentscales; density_reference mustexactlyreproducecurrentquantilecalibration. Existingonlinecontrollers unchanged,no extrainferencemodules orsamples.",
        pilot_gate="Original44trainID600..760after580cold. Densitycandidate mustloweractualScore by1e-6 andallother7primarymacro metrics no worse thanmatchedfreshdensity_quantile by1e-6. Groupcoverage absoluteerrors averaged perRe/channel first. Gate isonlytrainpilot screening,notall96cellcompletion.",
        invariant="Frozenoriginal4nativecalls99,66,33,0,joint20to20,actualsavedpreviousPerror,8primarymetrics/96cells/tolerance1e-6/all47strongreferences remain. No heldoutidentityfitting,folds,distillation,seedsweep,variableforecastblocks or32sampleinference.",
        limitations="Historicallyreusedtrainingpilot. A densityfamily assumptionisnotprooftruePDEerrorsfollowit. Independent90%calibrationdoesnotprovefullCDF,conditionalcoverage orwhole20framecoverage. CurrentmeanScore0.622351 andearlyreference2/96 remainuntilfullscopeverification."))
    norm = normalization()
    train(norm)
    provider = Provider(norm)
    reference = load(PREVIOUS/"calibration.pt")["quantile"]["separable_integral"]
    initial = calibrate(provider,OUT,NAMES,"density_reference",reference)
    values = pilot(provider,OUT,NAMES,initial)
    directions = {"standardized_interval_score":-1,"standardized_width":-1,"relative_aurc":-1,
                  "top10_energy":1,"per_frame_top10_energy":1,"worst_frame_coverage":1,
                  "worst_patch_coverage":1,"coverage_error":-1}
    gate = {}
    for mode,methods in values.items():
        old = methods["density_quantile"]
        passed = []
        comparisons = {}
        for name in ("density_gaussian","density_student4"):
            new = methods[name]
            regressions = [key for key,direction in directions.items() if direction*(new[key]-old[key])< -1e-6]
            comparisons[name] = dict(regressions=regressions,strict_score_gain=new["standardized_interval_score"]<old["standardized_interval_score"]-1e-6)
            if not regressions and comparisons[name]["strict_score_gain"]:
                passed.append(name)
        gate[mode] = dict(values=methods,comparisons=comparisons,passed=passed)
    dump(OUT/"gate.json",dict(results=gate,proceed_full_evaluation=any(row["passed"] for row in gate.values())))
    assert all(digest(ROOT/path)==sha for path,sha in hashes.items())
    dump(OUT/"complete.json",dict(time=time.time(),original_artifacts_unchanged=True,development_test_not_evaluated=True))
    print(json.dumps(gate),flush=True)
    print("ERROR_DENSITY_RISK_COMPLETE",flush=True)


if __name__=="__main__":
    main()
