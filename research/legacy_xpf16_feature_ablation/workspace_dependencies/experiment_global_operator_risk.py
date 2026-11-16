"""Train matched-budget fresh CNN/global risk heads, holding DiPred frozen.

Only original training identities fit either risk head. Independently calibrate
after freezing, and screen on later times of training identities before any
development/test replay. Preserve the current best head as a strong reference.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from experiment_fullblock_memory_risk import OUT as MEMORY, HEAD_SOURCE, load
from experiment_distribution_error_risk import OUT as BEST
from experiment_reliability_transport_e12 import FIT_RECORDS, PREDICTOR, geometry, history, normalization
from experiment_extended_train_error_adapters import LATE_RECORDS
from temporal_mean_memory_control import training_data
from fullblock_memory_risk_head import FullBlockMemoryRiskHead
from global_operator_risk_head import GlobalOperatorRiskHead
from experiment_temporal_error_uncertainty import pack, normalized
from experiment_direct_error_uncertainty import cached
from experiment_continue_rich_uncertainty import MODES
from risk_head_pilot_protocol import calibrate, pilot


OUT = ROOT / "runs/global_operator_risk_20261110"
ARMS = ("fresh_cnn", "fresh_global")
NAMES = ("frozen_best", *ARMS)
STEPS = 4000
METRICS = {
    "standardized_interval_score": -1,
    "standardized_width": -1,
    "relative_aurc": -1,
    "top10_energy": 1,
    "per_frame_top10_energy": 1,
    "worst_frame_coverage": 1,
    "worst_patch_coverage": 1,
    "coverage_error": -1,
}


def new_head(name, scale):
    if name == "fresh_global":
        return GlobalOperatorRiskHead(scale).cuda().eval()
    if name in ("fresh_cnn", "frozen_best"):
        head = FullBlockMemoryRiskHead("full", scale).cuda().eval()
        if name == "frozen_best":
            head.load_state_dict(load(BEST / "density_quantile.pt")["model"])
        return head
    raise ValueError(name)


def train(norm):
    base, last, history_value, memory, residual = training_data(norm)
    scale = load(MEMORY / "statistics.pt")["memory_scale"]
    schedules, initial_projections, parameter_counts = {}, {}, {}
    for name in ARMS:
        torch.manual_seed(42)
        head = new_head(name, scale)
        parameter_counts[name] = sum(p.numel() for p in head.parameters())
        initial_projections[name] = {
            key: value.detach().clone() for key, value in head.state_dict().items()
            if not key.startswith("net.")
        }
        torch.save(dict(model=head.state_dict()), OUT / (name + "_initial.pt"))
        sampler = torch.Generator(device="cuda").manual_seed(42)
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-5)
        schedule = hashlib.sha256()
        logs, begin = [], time.time()
        for step in range(1, STEPS + 1):
            ix = torch.randint(819, (4,), device="cuda", generator=sampler)
            available = (torch.rand(4, device="cuda", generator=sampler) >= .2).float()
            schedule.update(ix.cpu().numpy().tobytes())
            schedule.update(available.cpu().numpy().tobytes())
            optimizer.param_groups[0]["lr"] = 1e-6 + (3e-4 - 1e-6) * .5 * (
                1 + math.cos(math.pi * (step - 1) / STEPS)
            )
            optimizer.zero_grad(set_to_none=True)
            loss = head.loss(base[ix], last[ix], history_value[ix], memory[ix], residual[ix], available)
            assert torch.isfinite(loss)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
            assert torch.isfinite(grad_norm)
            if step == 1:
                gradients = {key: value.grad.detach().cpu().clone()
                             for key, value in head.named_parameters()}
            optimizer.step()
            if step == 1:
                torch.save(dict(
                    model={key: value.detach().cpu() for key, value in head.state_dict().items()},
                    gradients=gradients, indices=ix.cpu(), available=available.cpu(),
                    loss=float(loss), unclipped_gradient_norm=float(grad_norm),
                ), OUT / (name + "_first_update.pt"))
            if step == 1 or step % 500 == 0:
                logs.append(dict(step=step, loss=float(loss), elapsed=time.time() - begin))
                print("GLOBAL_RISK_TRAIN", name, logs[-1], flush=True)
        schedules[name] = schedule.hexdigest()
        torch.save(dict(model=head.state_dict(), seed=42, steps=STEPS, examples=819,
                        parameters=parameter_counts[name]), OUT / (name + ".pt"))
        dump(OUT / (name + "_training.json"), logs)
    assert len(set(schedules.values())) == 1
    assert all(torch.equal(value, initial_projections["fresh_global"][key])
               for key, value in initial_projections["fresh_cnn"].items())
    dump(OUT / "heads_frozen.json", dict(
        time=time.time(), real_parameters=parameter_counts, fit_pairs=819, steps=STEPS,
        initial_shared_projections_and_memory_scale_exact=True,
        minibatches_and_availability_masks_sha256=schedules,
        artifacts={name: digest(OUT / (name + ".pt")) for name in ARMS},
    ))


class Provider:
    def __init__(self, norm):
        self.norm = norm
        self.stats = load(HEAD_SOURCE / "statistics.pt")["statistics"]
        scale = load(MEMORY / "statistics.pt")["memory_scale"]
        self.geo = {split: geometry(manifest()["files"][split])
                    for split in ("calibration", "development", "test")}
        self.heads = {}
        for name in NAMES:
            head = new_head(name, scale)
            if name != "frozen_best":
                head.load_state_dict(load(OUT / (name + ".pt"))["model"])
            self.heads[name] = head.requires_grad_(False)

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
        for reference in ("fresh_cnn", "frozen_best"):
            new, old = methods["fresh_global"], methods[reference]
            gains = {key: direction * (new[key] - old[key]) for key, direction in METRICS.items()}
            comparisons[reference] = dict(
                gains=gains,
                regressions=[key for key, gain in gains.items() if gain < -1e-6],
                strict_score_gain=gains["standardized_interval_score"] > 1e-6,
            )
        passed = all(not row["regressions"] and row["strict_score_gain"]
                     for row in comparisons.values())
        results[mode] = dict(comparisons=comparisons, passed=passed)
    return dict(results=results, proceed_full_evaluation=any(row["passed"] for row in results.values()))


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    fit_ids = {row["file"] for row in FIT_RECORDS + LATE_RECORDS}
    held = {row["file"] for split, rows in manifest()["files"].items()
            if split != "train" for row in rows}
    assert not fit_ids & held
    frozen = [BEST / "density_quantile.pt", BEST / "calibration.pt",
              MEMORY / "statistics.pt", PREDICTOR / "augmented_diffusion/model.pt"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in frozen}
    dump(OUT / "protocol.json", dict(
        time=time.time(), source_sha256=digest(Path(__file__)),
        module_sha256=digest(ROOT / "global_operator_risk_head.py"),
        frozen_artifacts=hashes, names=NAMES, modes=MODES, steps=STEPS,
        real_parameters=dict(fresh_cnn=48340, fresh_global=48304),
        fit_pairs=819, fit_files=sorted(fit_ids),
        fitting="Fresh seed42 cores, same randomly initialized physical/denoising/memory projections; 819 original training pairs 200:600, 4000 updates, batch4, same AdamW/cosine/clip1, exact dedicated minibatches and 20% memory masks. No pretrained CNN core transferred to the global model.",
        architecture="Two width12 global spatial Fourier blocks, signed height modes4/positive width modes7, explicit real/imag weights, replicate spatial extension4, parallel temporal3x1x1 and pointwise paths; identical complete history/physical/last48 denoising/revealed-error inputs. All20futureframes remain jointly predicted.",
        fairness="Real parameter budgets differ by36 (<0.1%); no unused padding parameters. Same updates/data, not equal FLOPs, initial outputs, receptive fields or function capacity. Existing CNN GroupNorm already couples global statistics: do not claim its strict total receptive field is7x7. Frozen best is a stronger historical-budget reference, not a matched fresh architecture control.",
        calibration="Freeze all heads, then independently calibrate10 original calibration identities; frozen_best calibration must exactly reproduce latest quantile. No calibration labels train neural parameters.",
        pilot_gate="All44 original training identities with targets600:800, after580 cold issue. Under at least one fixed parent controller global must lower actual Score by1e-6 AND all other7 primary macro criteria no worse by1e-6 versus BOTH matched fresh_cnn and frozen_best. Only then full development/test followup; do not tune modes/counts/padding on pilot or known failures.",
        invariants="Original99,66,33,0, full20to20, actual saved preceding Perror, feedback only after complete20. Original8metrics/96cells/1e-6/all55 current reference pairs retained. No folds, distillation, seed sweep or extra Diffusion calls.",
        prior_goal_turn="Progress: user-authorized publication39d7085 completed, 57remote blobs checked, 96actual portable forecasts and2CPU guards passed. Publication does not achieve all-line improvement.",
        literature=["https://github.com/neuraloperator/neuraloperator/blob/main/doc/source/theory_guide/fno.rst", "https://openreview.net/pdf?id=cGpegxy12T"],
        limitations="Global Fourier UQ is existing methodology, not novelty alone. Training pilot and evaluation identities historically reused. No conditional/joint coverage or independent confirmation claim.",
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
    print("GLOBAL_OPERATOR_RISK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
