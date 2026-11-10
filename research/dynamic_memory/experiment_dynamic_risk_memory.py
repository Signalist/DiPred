"""Learn where revealed risk features move, versus same-capacity modulation.

The existing best risk head and DiPred are frozen. All supervision is actual
frozen-predictor error on original training identities, never artificial noise.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from clean_reliability_common import ROOT, configure, dump, digest, manifest
from experiment_fullblock_memory_risk import OUT as MEMORY, HEAD_SOURCE, load
from experiment_distribution_error_risk import OUT as BEST
from experiment_reliability_transport_e12 import FIT_RECORDS, PREDICTOR, normalization, geometry, history
from experiment_extended_train_error_adapters import LATE_RECORDS
from fullblock_memory_risk_head import FullBlockMemoryRiskHead
from temporal_mean_memory_control import training_data
from dynamic_risk_memory import DynamicRiskMemoryHead, warp_risk_features
from experiment_temporal_error_uncertainty import pack, normalized
from experiment_direct_error_uncertainty import cached
from experiment_global_operator_risk import METRICS
from experiment_continue_rich_uncertainty import MODES
from risk_head_pilot_protocol import calibrate, pilot


OUT = ROOT / "runs/dynamic_risk_memory_20261110"
ARMS = ("modulate", "warp")
NAMES = ("frozen_best", *ARMS)
STEPS = 2000


def new_head(name, scale):
    original = load(BEST / "density_quantile.pt")["model"]
    if name == "frozen_best":
        head = FullBlockMemoryRiskHead("full", scale).cuda().eval().requires_grad_(False)
        head.load_state_dict(original)
        return head
    return DynamicRiskMemoryHead(scale, original, name).cuda().eval()


def train(norm):
    base, last, history_value, memory, residual = training_data(norm)
    scale = load(MEMORY / "statistics.pt")["memory_scale"]
    original = new_head("frozen_best", scale)
    with torch.no_grad():
        expected = original.bounds(base[:8], last[:8], history_value[:8], memory[:8], True)
        value = torch.randn(2, 4, 20, 32, 64, device="cuda")
        assert torch.equal(warp_risk_features(value, value.new_zeros(2, 2, 20, 32, 64)), value)
    schedules, initial_modifiers = {}, {}
    for name in ARMS:
        torch.manual_seed(42)
        head = new_head(name, scale)
        original_state = {key: value.clone() for key, value in head.state_dict().items()
                          if not key.startswith("modifier.")}
        initial_modifiers[name] = {key: value.clone() for key, value in head.modifier.state_dict().items()}
        with torch.no_grad():
            actual = head.bounds(base[:8], last[:8], history_value[:8], memory[:8], True)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        torch.save(dict(model=head.state_dict()), OUT / (name + "_initial.pt"))
        parameters = list(head.modifier.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-5)
        sampler = torch.Generator(device="cuda").manual_seed(42)
        schedule = hashlib.sha256()
        logs, begin = [], time.time()
        for step in range(1, STEPS + 1):
            indices = torch.randint(819, (4,), device="cuda", generator=sampler)
            available = (torch.rand(4, device="cuda", generator=sampler) >= .2).float()
            schedule.update(indices.cpu().numpy().tobytes())
            schedule.update(available.cpu().numpy().tobytes())
            optimizer.param_groups[0]["lr"] = 1e-6 + (3e-4 - 1e-6) * .5 * (
                1 + math.cos(math.pi * (step - 1) / STEPS)
            )
            optimizer.zero_grad(set_to_none=True)
            loss = head.loss(base[indices], last[indices], history_value[indices], memory[indices],
                             residual[indices], available)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(parameter.grad is None for key, parameter in head.named_parameters()
                       if not key.startswith("modifier."))
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters, 1))
            if step == 1:
                gradients = {key: parameter.grad.detach().cpu().clone()
                             for key, parameter in head.modifier.named_parameters()}
            optimizer.step()
            if step == 1:
                torch.save(dict(
                    model={key: value.detach().cpu() for key, value in head.state_dict().items()},
                    gradients=gradients, indices=indices.cpu(), available=available.cpu(), loss=float(loss),
                ), OUT / (name + "_first_update.pt"))
            if step == 1 or step % 500 == 0:
                logs.append(dict(step=step, loss=float(loss), elapsed=time.time() - begin))
                print("DYNAMIC_RISK_MEMORY_TRAIN", name, logs[-1], flush=True)
        assert all(torch.equal(value, head.state_dict()[key]) for key, value in original_state.items())
        schedules[name] = schedule.hexdigest()
        torch.save(dict(model=head.state_dict(), kind=name, seed=42, steps=STEPS,
                        examples=819, trainable_parameters=10430), OUT / (name + ".pt"))
        dump(OUT / (name + "_training.json"), logs)
        with torch.no_grad():
            raw = head.modifier_fields(base[:32], last[:32], history_value[:32], memory[:32], True)
            fields = (4 * raw.tanh()) if name == "warp" else (math.log(2) * raw.tanh()).exp()
            dump(OUT / (name + "_fit_field_diagnostic.json"), dict(
                min=float(fields.min()), max=float(fields.max()),
                mean=float(fields.mean()), rms=float(fields.square().mean().sqrt()),
                units="gridpixels" if name == "warp" else "multiplicative_feature_gain",
                diagnostic_only=True, no_proof_of_useful_transport=True,
            ))
    assert len(set(schedules.values())) == 1
    assert all(torch.equal(value, initial_modifiers["warp"][key])
               for key, value in initial_modifiers["modulate"].items())
    dump(OUT / "heads_frozen.json", dict(
        time=time.time(), allocated_parameters=58770, trainable_parameters=10430,
        fit_pairs=819, steps=STEPS, seed=42, initial_existing_bounds_exact=True,
        original48340weights_unchanged=True, identical_initial_modifier_parameters=True,
        minibatches_and_availability_masks_sha256=schedules,
        artifacts={name: digest(OUT / (name + ".pt")) for name in ARMS},
    ))


class Provider:
    def __init__(self, norm):
        self.norm = norm
        self.stats = load(HEAD_SOURCE / "statistics.pt")["statistics"]
        self.geo = {split: geometry(manifest()["files"][split])
                    for split in ("calibration", "development", "test")}
        scale = load(MEMORY / "statistics.pt")["memory_scale"]
        self.heads = {}
        for name in NAMES:
            head = new_head(name, scale)
            if name != "frozen_best":
                head.load_state_dict(load(OUT / (name + ".pt"))["model"])
            self.heads[name] = head.eval().requires_grad_(False)

    @torch.no_grad()
    def bounds(self, split, start):
        point, base, last = pack(split, start, self.norm, self.geo[split])
        base, last = normalized(base, last, self.stats)
        observed = (history(split, start) - self.norm["mean"]) / self.norm["std"]
        previous = None if start == 0 else cached(split, start - 20)["prediction"]
        memory = torch.zeros_like(observed) if previous is None else observed - previous
        return point, {name: head.bounds(base, last, observed, memory, previous is not None)
                       for name, head in self.heads.items()}


def gate(values):
    results = {}
    for mode, methods in values.items():
        comparisons = {}
        for reference in ("modulate", "frozen_best"):
            new, old = methods["warp"], methods[reference]
            gains = {key: direction * (new[key] - old[key]) for key, direction in METRICS.items()}
            comparisons[reference] = dict(
                gains=gains, regressions=[key for key, gain in gains.items() if gain < -1e-6],
                strict_score_gain=gains["standardized_interval_score"] > 1e-6,
            )
        results[mode] = dict(comparisons=comparisons, passed=all(
            not row["regressions"] and row["strict_score_gain"] for row in comparisons.values()
        ))
    return dict(results=results, proceed_full_evaluation=any(row["passed"] for row in results.values()))


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    files = {row["file"] for row in FIT_RECORDS + LATE_RECORDS}
    held = {row["file"] for split, records in manifest()["files"].items()
            if split != "train" for row in records}
    assert not files & held
    paths = [BEST / "density_quantile.pt", BEST / "calibration.pt", MEMORY / "statistics.pt",
             PREDICTOR / "augmented_diffusion/model.pt"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    dump(OUT / "protocol.json", dict(
        time=time.time(), source_sha256=digest(Path(__file__)),
        module_sha256=digest(ROOT / "dynamic_risk_memory.py"), frozen_artifacts=hashes,
        names=NAMES, arms=ARMS, modes=MODES, steps=STEPS, fit_pairs=819,
        allocated_parameters=58770, trainable_parameters=10430,
        question="Can forecast-conditioned spatial movement of actuallyrevealed error features improve risk beyond equal-capacity diagonal gain? Distinct from old static depthwise memory filters, fixed velocity advection, and old5frame pointCorrector.",
        architecture="Existing48340parameter best quantile risk head fullyfrozen. Inputcontext32=base14+history4+lastdenoising8+coordinates2+revealedfutureprojectedmemory4. ModifierConv3d32->12,k3,GN3,SiLU,Conv12->2,k1. Same10430trainableparams,zero last layer,initialriskbounds exactly currentbest forboth.",
        transformation="warp:two output fields,4*tanh signedgridpixel displacements perfuturelead; backwardbilinear border sampling align_cornersFalse on4future-lead projected risk-memoryfeatures. No physicalvelocity claim. modulate:same2fields give exp(log2*tanh) gains for2latentfeaturegroups; no spatialshift. Both act only on alreadyrevealederrorfeatures, neverP or currentY.",
        fitting="Onlyoriginal819trainingidentitypairs,query200..360/400..560,actualtargets220:400/420:600;seed42,batch4,2000updates,AdamW3e-4cosine1e-6,clip1,same20%availability masks and exact batches. Bothinitialmodifierweights identical; oldriskweights/normalizers neverchange. Pixelbound4 is fixed halfexisting8x8regionwidth, not tuned on known failures; gaincontrol strengths have different units, not perturbation-norm matching.",
        calibration="Afterboth headsfreeze independently recompute10calibrationIDs; frozen_best initialstates and1760pilotrows mustexactly reproduce currentquantile. No heldout or calibrationlabels train moduleparameters.",
        pilot_gate="44originaltrainIDs,580coldthen600..760warm queries(targets620:800). Underatleastonefixedcontroller warp mustlower actualScore>1e-6 and allother7primarymacro metrics no worse>1e-6 versus BOTH equalcapacitymodulate andfrozen_best. Keepallmethods; fulloriginaldevtest onlyaftergate. Nocoefficient/motionbound/gain/seedscan.",
        invariants="Native99,66,33,0, joint20to20, actualsavedpreviousP residual, feedbackonlyafterwhole20, all8metrics96cells/tolerance1e-6/all55strongreferences. No time subdivision,pointcorrection,distillation,folds or32pathinference.",
        limitation="Latent risk-feature warping is a candidate statistical motion model, not a physical errorPDE or a new jointcoverage theorem. Learned displacement alone doesnot prove transport benefit; actualrisk metrics and matchedgaincontrol decide. Allidentities historically reused.",
    ))
    norm = normalization()
    train(norm)
    provider = Provider(norm)
    reference = load(BEST / "calibration.pt")["density_quantile"]["separable_integral"]
    initial = calibrate(provider, OUT, NAMES, "frozen_best", reference)
    values = pilot(provider, OUT, NAMES, initial)
    decision = gate(values)
    dump(OUT / "gate.json", decision)
    assert all(digest(ROOT / path) == sha for path, sha in hashes.items())
    dump(OUT / "complete.json", dict(time=time.time(), original_artifacts_unchanged=True,
                                     development_test_not_evaluated=True))
    print(json.dumps(decision), flush=True)
    print("DYNAMIC_RISK_MEMORY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
