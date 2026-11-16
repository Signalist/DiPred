"""Matched-budget error adapters using later original-training trajectories.

The point predictor and rich risk head remain frozen. Old-data controls must
reproduce the archived adapter weights exactly before any evaluation replay.
"""

import json
import math
import time
from pathlib import Path

import torch

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from experiment_reliability_transport_e12 import (
    FIT_RECORDS, FIT_STARTS, PREDICTOR, geometry, history, target, normalization,
)
from experiment_structured_quantile_uncertainty import OUT as HEAD_SOURCE, Provider as FrozenProvider
from experiment_tail_scale_risk_adapter import OUT as TAIL_SOURCE
from experiment_fourstep_cue_tails import OUT as CUE_SOURCE
from experiment_temporal_error_uncertainty import pack, normalized
from infer_structured_fourstep_uncertainty import StructuredFourStepPredictor
from tail_scale_risk_adapter import TailScaleRiskAdapter
from fourstep_cue_tail_adapter import CueTailAdapter
from fourstep_denoising_risk_cues import cue_inputs
from diagnose_train_content_memory import read_block
from experiment_continue_rich_uncertainty import controller, MODES
from causal_shape_calibration import initial_state
from replay_conditional_risk import replay


OUT = ROOT / "runs/extended_train_error_adapters_20261109"
LATE_RECORDS = [r for r in FIT_RECORDS if r["frames"] >= 600]
LATE_STARTS = tuple(range(400, 561, 20))
NAMES = ("rich", "replay_separate", "expanded_separate", "replay_history", "expanded_history")


@torch.no_grad()
def training_data(norm):
    forecaster = StructuredFourStepPredictor(geometry(FIT_RECORDS))
    all_inputs, all_lowers, all_uppers, all_errors = [], [], [], []
    for stage, starts, records in (("original", FIT_STARTS, FIT_RECORDS),
                                   ("later", LATE_STARTS, LATE_RECORDS)):
        forecaster.geometry = geometry(records)
        for start in starts:
            if stage == "original":
                point, base, last = pack("fit", start, norm, forecaster.geometry)
                base, last = normalized(base, last, forecaster.stats)
                observed = history("fit", start)
                h = (observed - norm["mean"]) / norm["std"]
                lower, upper = forecaster.head.bounds(base, last, h, torch.ones_like(point[:1]))
            else:
                observed = read_block(records, start)
                labels = []
                hook = forecaster.model.net.register_forward_pre_hook(
                    lambda module, args: labels.append(int(args[1][0])))
                try:
                    point, lower, upper = forecaster.forecast(observed)
                finally:
                    hook.remove()
                assert labels == [99, 66, 33, 0]
                h = (observed - norm["mean"]) / norm["std"]
            features = cue_inputs(h, point, lower, upper, None, "history_change", True)
            with (OUT / "training_events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(stage=stage, start=start, event="issued",
                    time=time.time(), batch=len(records))) + "\n")
            # Future truth enters only this offline supervision label, after
            # the native forecast and all adapter inputs are constructed.
            truth = target("fit", start) if stage == "original" else read_block(records, start + 20)
            residual = (truth - norm["mean"]) / norm["std"] - point
            with (OUT / "training_events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(stage=stage, start=start, event="label_revealed", time=time.time())) + "\n")
            all_inputs.append(features)
            all_lowers.append(lower)
            all_uppers.append(upper)
            all_errors.append(residual)
            print("DATA", stage, start, len(records), flush=True)
    return [torch.cat(values) for values in (all_inputs, all_lowers, all_uppers, all_errors)]


def train(norm):
    x, lower, upper, residual = training_data(norm)
    old_count = len(FIT_RECORDS) * len(FIT_STARTS)
    assert old_count == 414 and len(x) == 819
    initial = []
    controls = {}
    for name in NAMES[1:]:
        torch.manual_seed(42)
        head = (CueTailAdapter() if name.endswith("history") else TailScaleRiskAdapter(True)).cuda()
        initial.append({key: value.clone() for key, value in head.state_dict().items()})
        count = len(x) if name.startswith("expanded_") else old_count
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
            gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
            assert torch.isfinite(gradient)
            optimizer.step()
            if step == 1 or step % 500 == 0:
                logs.append(dict(step=step, loss=float(loss), elapsed=time.time() - begin))
                print("TRAIN", name, logs[-1], flush=True)
        torch.save(dict(model=head.state_dict(), seed=42, steps=2000, examples=count), OUT / (name + ".pt"))
        dump(OUT / (name + "_training.json"), logs)
        if name.startswith("replay_"):
            source = TAIL_SOURCE / "separate_tails.pt" if name.endswith("separate") else CUE_SOURCE / "asym_history.pt"
            archived = torch.load(source, map_location="cuda", weights_only=False)["model"]
            assert all(torch.equal(value, archived[key]) for key, value in head.state_dict().items()), name
            controls[name] = dict(source=str(source.relative_to(ROOT)), sha256=digest(source), weights_exact=True)
    assert all(torch.equal(value, initial[0][key]) for state in initial[1:] for key, value in state.items())
    dump(OUT / "heads_frozen.json", dict(time=time.time(), parameters_each=13124,
        initialization_identical=True, controls=controls,
        artifacts={name + ".pt": digest(OUT / (name + ".pt")) for name in NAMES[1:]}))


class Provider:
    def __init__(self, norm):
        self.norm = norm
        self.frozen = FrozenProvider(norm)
        self.heads = {}
        for name in NAMES[1:]:
            head = CueTailAdapter() if name.endswith("history") else TailScaleRiskAdapter(True)
            head.load_state_dict(torch.load(OUT / (name + ".pt"), map_location="cuda", weights_only=False)["model"])
            self.heads[name] = head.cuda().eval().requires_grad_(False)

    @torch.no_grad()
    def bounds(self, split, start):
        point, bounds = self.frozen.bounds(split, start)
        lower, upper = bounds["rich"]
        h = (history(split, start) - self.norm["mean"]) / self.norm["std"]
        x = cue_inputs(h, point, lower, upper, None, "history_change", start > 0)
        return point, dict(rich=(lower, upper), **{
            name: head.bounds(x, lower, upper) for name, head in self.heads.items()})


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    train_ids = {r["file"] for r in FIT_RECORDS}
    assert all(not train_ids.intersection(r["file"] for r in rows)
               for split, rows in manifest()["files"].items() if split != "train")
    frozen = {str(path.relative_to(ROOT)): digest(path) for path in (
        PREDICTOR / "augmented_diffusion/model.pt", HEAD_SOURCE / "rich.pt",
        HEAD_SOURCE / "statistics.pt", TAIL_SOURCE / "separate_tails.pt", CUE_SOURCE / "asym_history.pt")}
    dump(OUT / "protocol.json", dict(time=time.time(), source_sha256=digest(Path(__file__)),
        predictor_sha256=digest(PREDICTOR / "augmented_diffusion/model.pt"), frozen_artifacts=frozen,
        heads=NAMES, modes=MODES, original_fit_files=[r["file"] for r in FIT_RECORDS],
        later_fit_files=[r["file"] for r in LATE_RECORDS], original_frames=[200, 400], later_frames=[400, 600],
        old_examples=414, expanded_examples=819, parameters_each=13124, steps_each=2000,
        training="Fresh same initialization, seed42, AdamW3e-4cosine1e-6,batch4,true05/95pinball. Same update budget; more unique actual forecast errors, not more model capacity or steps.",
        controls="Original-data retraining of separate-tail and history-cue heads must exactly match frozen archives before evaluation. Same input array, separate-tail erases reservedcue slots by design.",
        point="Original4calls99,66,33,0, joint20->20; all point/richhead weights and statistics frozen. No new inference module, folds,distillation or seed sweep.",
        calibration="10independentcalibration trajectories, recomputed for every frozen head. No development/test fitting. Historical exploratory identity/time reuse, no fresh confirmation or formal coverage guarantee."))
    norm = normalization()
    train(norm)
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
    torch.save({name: {mode: {k: v.cpu() for k, v in state.items()}
                      for mode, state in modes.items()} for name, modes in initial.items()}, OUT / "calibration.pt")
    dump(OUT / "calibration_locked.json", dict(time=time.time(), sha256=digest(OUT / "calibration.pt")))
    replay(OUT, provider, NAMES, MODES, controller, initial)
    assert all(digest(ROOT / path) == sha for path, sha in frozen.items())
    dump(OUT / "complete.json", dict(time=time.time(), all_original_models_unchanged=True))


if __name__ == "__main__":
    main()
