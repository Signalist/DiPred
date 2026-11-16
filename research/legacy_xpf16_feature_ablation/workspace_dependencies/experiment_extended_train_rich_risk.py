"""Extend the risk-head training horizon without altering inference structure.

Includes an exact old-data rich-head control. New representations are fitted
only on original training IDs; the native four-call point forecaster is frozen.
"""

import json
import math
import time
from pathlib import Path

import torch

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from experiment_reliability_transport_e12 import FIT_RECORDS, FIT_STARTS, PREDICTOR, geometry, history, target, normalization
from experiment_structured_quantile_uncertainty import OUT as HEAD_SOURCE, Provider as FrozenProvider
from experiment_temporal_error_uncertainty import pack, normalized
from experiment_extended_train_error_adapters import LATE_RECORDS, LATE_STARTS
from diagnose_train_content_memory import read_block
from infer_structured_fourstep_uncertainty import StructuredFourStepPredictor
from structured_quantile_uncertainty import StructuredQuantileHead
from reliability_risk_transport import base_features
from tail_scale_risk_adapter import TailScaleRiskAdapter
from fourstep_cue_tail_adapter import CueTailAdapter
from fourstep_denoising_risk_cues import cue_inputs
from experiment_continue_rich_uncertainty import controller, MODES
from causal_shape_calibration import initial_state
from replay_conditional_risk import replay


OUT = ROOT / "runs/extended_train_rich_risk_20261109"
ADAPTERS = ("core_separate_old", "core_separate_all", "core_history_old", "core_history_all")
NAMES = ("rich", "expanded_rich", *ADAPTERS)


@torch.no_grad()
def prepare(norm):
    forecaster = StructuredFourStepPredictor(geometry(FIT_RECORDS))
    bases, lasts, histories, errors, points, sizes = [], [], [], [], [], []
    (OUT / "later_data").mkdir()
    for stage, starts, records in (("original", FIT_STARTS, FIT_RECORDS),
                                   ("later", LATE_STARTS, LATE_RECORDS)):
        geo = geometry(records)
        forecaster.geometry = geo
        for start in starts:
            if stage == "original":
                p, base, last = pack("fit", start, norm, geo)
                observed = history("fit", start)
            else:
                captured, calls = {}, []
                hooks = [forecaster.model.net.up2.register_forward_hook(
                    lambda module, args, result: captured.update(last=result.detach().clone())),
                    forecaster.model.net.register_forward_pre_hook(
                    lambda module, args: calls.append(int(args[1][0])))]
                try:
                    observed = read_block(records, start)
                    p, _, _ = forecaster.forecast(observed)
                finally:
                    for hook in hooks:
                        hook.remove()
                assert calls == [99, 66, 33, 0]
                last = captured["last"].half().float()
                base = base_features(observed, p * norm["std"] + norm["mean"], norm["mean"], norm["std"], geo)
            base, last = normalized(base, last, forecaster.stats)
            h = (observed - norm["mean"]) / norm["std"]
            with (OUT / "training_events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event="issued", stage=stage, start=start,
                    time=time.time(), batch=len(records))) + "\n")
            truth = target("fit", start) if stage == "original" else read_block(records, start + 20)
            residual = (truth - norm["mean"]) / norm["std"] - p
            if stage == "later":
                # Persist compact known inputs and actual labels so subsequent
                # risk studies need not rerun native forecasts for these IDs.
                torch.save(dict(observed=observed.cpu(), prediction=p.cpu(),
                    last=captured["last"].half().cpu(), residual=residual.cpu()),
                    OUT / "later_data" / (str(start) + ".pt"))
            with (OUT / "training_events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event="label_revealed", stage=stage,
                    start=start, time=time.time())) + "\n")
            for destination, value in ((bases, base), (lasts, last), (histories, h),
                                       (errors, residual), (points, p)):
                destination.append(value)
            sizes.append(len(records))
            print("DATA", stage, start, len(records), flush=True)
    return [torch.cat(values) for values in (bases, lasts, histories, errors, points)], sizes


def train_heads(data):
    base, last, h, residual, _ = data
    assert len(base) == 819
    original = torch.load(HEAD_SOURCE / "rich.pt", map_location="cuda", weights_only=False)["model"]
    initial = []
    for name, count in (("replay_rich", 414), ("expanded_rich", 819)):
        torch.manual_seed(42)
        head = StructuredQuantileHead("rich").cuda()
        initial.append({key: value.clone() for key, value in head.state_dict().items()})
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-5)
        logs = []
        begin = time.time()
        scale = torch.ones_like(residual[:1])
        for step in range(1, 2001):
            ix = torch.randint(count, (4,), device="cuda")
            rate = 1e-6 + (3e-4 - 1e-6) * 0.5 * (1 + math.cos(math.pi * (step - 1) / 2000))
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            loss = head.loss(base[ix], last[ix], h[ix], residual[ix], scale)
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(head.parameters(), 1))
            optimizer.step()
            if step == 1 or step % 500 == 0:
                logs.append(dict(step=step, loss=float(loss), elapsed=time.time() - begin))
                print("HEAD", name, logs[-1], flush=True)
        if name == "replay_rich":
            assert all(torch.equal(value, original[key]) for key, value in head.state_dict().items())
        torch.save(dict(model=head.state_dict(), seed=42, steps=2000, examples=count), OUT / (name + ".pt"))
        dump(OUT / (name + "_training.json"), logs)
    assert all(torch.equal(value, initial[0][key]) for key, value in initial[1].items())
    dump(OUT / "core_frozen.json", dict(time=time.time(), old_weights_exact=True,
        parameters=45060, initialization_identical=True,
        artifacts={name + ".pt": digest(OUT / (name + ".pt")) for name in ("replay_rich", "expanded_rich")}))


def train_adapters(data, sizes):
    base, last, h, residual, point = data
    rich = StructuredQuantileHead("rich").cuda().eval().requires_grad_(False)
    rich.load_state_dict(torch.load(OUT / "expanded_rich.pt", map_location="cuda", weights_only=False)["model"])
    features, lowers, uppers = [], [], []
    offset = 0
    with torch.no_grad():
        for size in sizes:
            index = slice(offset, offset + size)
            lo, hi = rich.bounds(base[index], last[index], h[index], torch.ones_like(residual[:1]))
            features.append(cue_inputs(h[index], point[index], lo, hi, None, "history_change", True))
            lowers.append(lo)
            uppers.append(hi)
            offset += size
        x, lower, upper = [torch.cat(values) for values in (features, lowers, uppers)]
    assert offset == 819
    initial = []
    for name in ADAPTERS:
        torch.manual_seed(42)
        head = CueTailAdapter() if "history" in name else TailScaleRiskAdapter(True)
        head = head.cuda()
        initial.append({key: value.clone() for key, value in head.state_dict().items()})
        count = 414 if name.endswith("_old") else 819
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-5)
        logs = []
        begin = time.time()
        for step in range(1, 2001):
            ix = torch.randint(count, (4,), device="cuda")
            rate = 1e-6 + (3e-4 - 1e-6) * 0.5 * (1 + math.cos(math.pi * (step - 1) / 2000))
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            loss = head.loss(x[ix], lower[ix], upper[ix], residual[ix])
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(head.parameters(), 1))
            optimizer.step()
            if step == 1 or step % 500 == 0:
                logs.append(dict(step=step, loss=float(loss), elapsed=time.time() - begin))
                print("ADAPTER", name, logs[-1], flush=True)
        torch.save(dict(model=head.state_dict(), seed=42, steps=2000, examples=count), OUT / (name + ".pt"))
        dump(OUT / (name + "_training.json"), logs)
    assert all(torch.equal(value, initial[0][key]) for state in initial[1:] for key, value in state.items())
    dump(OUT / "heads_frozen.json", dict(time=time.time(),
        risk_parameters=45060, adapter_parameters=13124, adapter_initialization_identical=True,
        artifacts={name + ".pt": digest(OUT / (name + ".pt")) for name in ("expanded_rich", *ADAPTERS)}))


class Provider:
    def __init__(self, norm):
        self.norm = norm
        self.frozen = FrozenProvider(norm)
        self.rich = StructuredQuantileHead("rich").cuda().eval().requires_grad_(False)
        self.rich.load_state_dict(torch.load(OUT / "expanded_rich.pt", map_location="cuda", weights_only=False)["model"])
        self.adapters = {}
        for name in ADAPTERS:
            head = CueTailAdapter() if "history" in name else TailScaleRiskAdapter(True)
            head.load_state_dict(torch.load(OUT / (name + ".pt"), map_location="cuda", weights_only=False)["model"])
            self.adapters[name] = head.cuda().eval().requires_grad_(False)

    @torch.no_grad()
    def bounds(self, split, start):
        point, raw = self.frozen.bounds(split, start)
        _, base, last = pack(split, start, self.norm, self.frozen.geo[split])
        base, last = normalized(base, last, self.frozen.stats)
        h = (history(split, start) - self.norm["mean"]) / self.norm["std"]
        lower, upper = self.rich.bounds(base, last, h, torch.ones_like(point[:1]))
        x = cue_inputs(h, point, lower, upper, None, "history_change", start > 0)
        return point, dict(rich=raw["rich"], expanded_rich=(lower, upper), **{
            name: head.bounds(x, lower, upper) for name, head in self.adapters.items()})


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    fit = {r["file"] for r in FIT_RECORDS}
    assert all(not fit.intersection(r["file"] for r in records)
               for split, records in manifest()["files"].items() if split != "train")
    frozen = {str(path.relative_to(ROOT)): digest(path) for path in (
        PREDICTOR / "augmented_diffusion/model.pt", HEAD_SOURCE / "rich.pt", HEAD_SOURCE / "statistics.pt")}
    dump(OUT / "protocol.json", dict(time=time.time(), source_sha256=digest(Path(__file__)),
        predictor_sha256=digest(PREDICTOR / "augmented_diffusion/model.pt"), frozen_artifacts=frozen,
        heads=NAMES, modes=MODES, fit_files=[r["file"] for r in FIT_RECORDS], later_files=[r["file"] for r in LATE_RECORDS],
        original_examples=414, expanded_examples=819, old_frames=[200, 400], later_frames=[400, 600],
        same_architecture="45060riskhead,13124optionaladapter. Same2000updatesperstage,seed42,batch4,AdamW3e-4cosine1e-6,actual05/95pinball. No added inference branch.",
        comparison="Exact original-rich retraining control; new-rich adapters fit either old414or expanded819samples. Combine previous old-rich adapter-data study for a2x2data-by-stage comparison, not an optimizer-step increase.",
        point="Original99,66,33,0andjoint20->20frozen. All new features from actual inference; truth only supplies offline labels.",
        calibration="Independent10trajectories,recomputed only after allnewweightsfreeze. No development/test fitting,folds,distillation or extra training seeds.",
        limits="More one-time training-data preparation; not full-pipeline equalFLOP. All evaluation data historically reused exploration; no independent confirmation or formal coverage guarantee."))
    norm = normalization()
    data, sizes = prepare(norm)
    train_heads(data)
    train_adapters(data, sizes)
    del data
    provider = Provider(norm)
    scores = {name: [] for name in NAMES}
    for start in range(0, 161, 20):
        with torch.no_grad():
            point, bounds = provider.bounds("calibration", start)
            residual = (target("calibration", start) - norm["mean"]) / norm["std"] - point
            for name, (lower, upper) in bounds.items():
                scores[name].append(((residual - 0.5 * (lower + upper)).abs()
                    / (0.5 * (upper - lower)).clamp_min(1e-6)).cpu())
    initial = {name: {"separable_integral": initial_state(torch.cat(values).cuda(), "separable_integral")}
               for name, values in scores.items()}
    torch.save({name: {mode: {k: v.cpu() for k, v in state.items()} for mode, state in modes.items()}
                for name, modes in initial.items()}, OUT / "calibration.pt")
    dump(OUT / "calibration_locked.json", dict(time=time.time(), sha256=digest(OUT / "calibration.pt")))
    replay(OUT, provider, NAMES, MODES, controller, initial)
    assert all(digest(ROOT / path) == sha for path, sha in frozen.items())
    dump(OUT / "complete.json", dict(time=time.time(), original_models_unchanged=True))


if __name__ == "__main__":
    main()
