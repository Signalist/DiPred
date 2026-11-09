"""Matched continuation training for full-block residual-risk information.

All fitting uses original train IDs; the fixed predictor never sees target
labels during inference. Calibration is recomputed after every head freezes.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from clean_reliability_common import ROOT, configure, dump, digest, manifest
from experiment_reliability_transport_e12 import FIT_RECORDS, FIT_STARTS, PREDICTOR, history, target, geometry, normalization, read_fit
from experiment_extended_train_error_adapters import LATE_RECORDS, LATE_STARTS
from experiment_extended_train_rich_risk import OUT as SOURCE
from experiment_structured_quantile_uncertainty import OUT as HEAD_SOURCE
from experiment_temporal_error_uncertainty import pack, normalized
from experiment_direct_error_uncertainty import cached
from infer_structured_fourstep_uncertainty import StructuredFourStepPredictor
from direct_error_uncertainty import forecast_features
from reliability_risk_transport import base_features
from structured_quantile_uncertainty import StructuredQuantileHead
from fullblock_memory_risk_head import MODES as MEMORY_MODES, FullBlockMemoryRiskHead
from causal_shape_calibration import initial_state


OUT = ROOT / "runs/fullblock_memory_risk_20261109"
NAMES = ("frozen_rich", "none", "last", "full")


def load(path):
    return torch.load(path, map_location="cuda", weights_only=False)


@torch.no_grad()
def prepare(norm):
    stats = load(HEAD_SOURCE / "statistics.pt")["statistics"]
    forecaster = StructuredFourStepPredictor(geometry(FIT_RECORDS))
    archived = ROOT / "runs/learned_feedback_risk_20261109/previous_fit180.pt"
    old_protocol = json.loads((archived.parent / "protocol.json").read_text())
    assert old_protocol["predictor_sha256"] == digest(PREDICTOR / "augmented_diffusion/model.pt")
    previous180 = load(archived)
    previous = {}
    for start in (180, 380):
        observed = read_fit(start)
        labels = []
        hook = forecaster.model.net.register_forward_pre_hook(lambda module, args: labels.append(int(args[1][0])))
        try:
            point, _ = forecast_features(forecaster.model, (observed - norm["mean"]) / norm["std"])
        finally:
            hook.remove()
        assert labels == [99, 66, 33, 0]
        if start == 180:
            assert torch.equal(point, previous180)
        packet = dict(prediction=point.cpu(), files=[r["file"] for r in FIT_RECORDS], native_labels=labels,
            history_sha256=hashlib.sha256(observed.cpu().numpy().tobytes()).hexdigest())
        torch.save(packet, OUT / ("previous_fit" + str(start) + ".pt"))
        previous[start] = point
        print("NATIVE_PRELUDE", start, len(point), flush=True)
    values = [[] for _ in range(6)]
    sizes, cache_hashes = [], {}
    index = {r["file"]: i for i, r in enumerate(FIT_RECORDS)}
    later_indices = [index[r["file"]] for r in LATE_RECORDS]
    for later, records, starts in ((False, FIT_RECORDS, FIT_STARTS), (True, LATE_RECORDS, LATE_STARTS)):
        geo = geometry(records)
        previous_point = previous[380][later_indices] if later else previous[180]
        for start in starts:
            if later:
                path = SOURCE / "later_data" / (str(start) + ".pt")
                saved = load(path)
                observed, point, last = saved["observed"], saved["prediction"], saved["last"].float()
                base = base_features(observed, point * norm["std"] + norm["mean"], norm["mean"], norm["std"], geo)
                cache_hashes[str(start)] = digest(path)
            else:
                point, base, last = pack("fit", start, norm, geo)
                observed = history("fit", start)
            h = (observed - norm["mean"]) / norm["std"]
            memory = h - previous_point
            base, last = normalized(base, last, stats)
            with (OUT / "training_events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event="features_ready", start=start, later=later,
                    previous_query=start - 20, time=time.time(), batch=len(records))) + "\n")
            # The reused compact later-training packets contain offline labels;
            # only observed/prediction/last enter features. No claim is made
            # that these mixed offline packets were first read without labels.
            residual = saved["residual"] if later else (target("fit", start) - norm["mean"]) / norm["std"] - point
            for destination, value in zip(values, (base, last, h, memory, residual, point)):
                destination.append(value)
            with (OUT / "training_events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event="supervision_selected", start=start, later=later, time=time.time())) + "\n")
            sizes.append(len(records))
            previous_point = point
            print("MEMORY_FIT_DATA", start, len(records), flush=True)
    data = [torch.cat(parts) for parts in values]
    assert len(data[0]) == 819
    scale = data[3].square().mean((0, 2, 3, 4), keepdim=True).sqrt().clamp_min(.01)
    torch.save(dict(memory_scale=scale.cpu()), OUT / "statistics.pt")
    dump(OUT / "training_data.json", dict(pairs=819, sizes=sizes, later_cache_hashes=cache_hashes,
        previous180_matches_archived=True, native_prelude_forecasts=92,
        previous380_created_before_loading400history=True,
        previous180_sha256=digest(OUT / "previous_fit180.pt"), previous380_sha256=digest(OUT / "previous_fit380.pt"),
        memory_scale_source="RMS of all819known previous-error inputs, original train IDs only; not current target labels.",
        offline_packet_note="Later compact fitting packets already include labels, explicitly excluded from feature inputs and used only for supervision."))
    return data, scale


def train(data, scale):
    base, last, h, memory, residual, _ = data
    original = load(SOURCE / "expanded_rich.pt")["model"]
    initial, schedules = {}, {}
    for mode in MEMORY_MODES:
        torch.manual_seed(42)
        head = FullBlockMemoryRiskHead(mode, scale).cuda()
        missing = head.load_state_dict(original, strict=False)
        assert set(missing.missing_keys) == {"memory_scale", "memory_projection.weight", "memory_projection.bias"}
        assert not missing.unexpected_keys
        assert sum(p.numel() for p in head.parameters()) == 48340
        initial[mode] = {key: value.clone() for key, value in head.state_dict().items()}
        with torch.no_grad():
            frozen = StructuredQuantileHead("rich").cuda().eval()
            frozen.load_state_dict(original)
            expected = frozen.bounds(base[:4], last[:4], h[:4], torch.ones_like(residual[:1]))
            actual = head.bounds(base[:4], last[:4], h[:4], memory[:4], True)
            assert all(torch.equal(a, b) for a, b in zip(expected, actual))
            del frozen
        # Reset the generator after identical initial-output checks so every
        # model sees precisely the same minibatches and cold-start masks.
        sampler = torch.Generator(device="cuda").manual_seed(42)
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-5)
        schedule = hashlib.sha256()
        logs, begin = [], time.time()
        for step in range(1, 2001):
            ix = torch.randint(819, (4,), device="cuda", generator=sampler)
            available = (torch.rand(4, device="cuda", generator=sampler) >= .2).float()
            schedule.update(ix.cpu().numpy().tobytes())
            schedule.update(available.cpu().numpy().tobytes())
            rate = 1e-6 + (3e-4 - 1e-6) * .5 * (1 + math.cos(math.pi * (step - 1) / 2000))
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            loss = head.loss(base[ix], last[ix], h[ix], memory[ix], residual[ix], available)
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(head.parameters(), 1))
            optimizer.step()
            if step == 1 or step % 500 == 0:
                logs.append(dict(step=step, loss=float(loss), elapsed=time.time() - begin))
                print("MEMORY_HEAD_TRAIN", mode, logs[-1], flush=True)
        schedules[mode] = schedule.hexdigest()
        torch.save(dict(model=head.state_dict(), mode=mode, seed=42, additional_steps=2000,
            examples=819, warm_start_sha256=digest(SOURCE / "expanded_rich.pt")), OUT / (mode + ".pt"))
        dump(OUT / (mode + "_training.json"), logs)
    assert len(set(schedules.values())) == 1
    assert all(torch.equal(value, initial[mode][key]) for mode in MEMORY_MODES[1:]
               for key, value in initial[MEMORY_MODES[0]].items())
    dump(OUT / "heads_frozen.json", dict(time=time.time(), parameters=48340, pretrained_core_parameters=45060,
        allocated_parameters_identical=True, initial_weights_identical=True, initial_outputs_match_frozen_rich=True,
        minibatches_and_availability_masks_sha256=schedules,
        artifacts={path.name: digest(path) for path in OUT.glob("*.pt")}))


class Provider:
    def __init__(self, norm):
        self.norm = norm
        self.stats = load(HEAD_SOURCE / "statistics.pt")["statistics"]
        self.geo = {split: geometry(manifest()["files"][split]) for split in ("calibration", "development", "test")}
        self.scale = load(OUT / "statistics.pt")["memory_scale"]
        self.frozen = StructuredQuantileHead("rich").cuda().eval().requires_grad_(False)
        self.frozen.load_state_dict(load(SOURCE / "expanded_rich.pt")["model"])
        self.heads = {}
        for mode in MEMORY_MODES:
            head = FullBlockMemoryRiskHead(mode, self.scale).cuda().eval().requires_grad_(False)
            head.load_state_dict(load(OUT / (mode + ".pt"))["model"])
            self.heads[mode] = head

    @torch.no_grad()
    def bounds(self, split, start):
        point, base, last = pack(split, start, self.norm, self.geo[split])
        base, last = normalized(base, last, self.stats)
        h = (history(split, start) - self.norm["mean"]) / self.norm["std"]
        previous = None if start == 0 else cached(split, start - 20)["prediction"]
        memory = torch.zeros_like(h) if previous is None else h - previous
        bounds = {"frozen_rich": self.frozen.bounds(base, last, h, torch.ones_like(point[:1]))}
        bounds.update({mode: head.bounds(base, last, h, memory, previous is not None) for mode, head in self.heads.items()})
        return point, bounds


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    fit_ids = {r["file"] for r in FIT_RECORDS}
    held = {r["file"] for split, rows in manifest()["files"].items() if split != "train" for r in rows}
    assert not fit_ids & held
    paths = [PREDICTOR / "augmented_diffusion/model.pt", SOURCE / "expanded_rich.pt", HEAD_SOURCE / "statistics.pt"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    dump(OUT / "protocol.json", dict(time=time.time(), source_sha256=digest(Path(__file__)),
        module_sha256=digest(ROOT / "fullblock_memory_risk_head.py"), frozen_artifacts=hashes,
        predictor_sha256=digest(paths[0]), fit_files=[r["file"] for r in FIT_RECORDS],
        later_files=[r["file"] for r in LATE_RECORDS], fit_pairs=819, fitting_extent=[200,600], modes=MEMORY_MODES,
        architecture="Existing45060parameter rich core plus zero-initialized40->80Conv2D memory projection,48340total. All20previous residual frames can condition every futurelead. No point correction or extra diffusion call.",
        information_controls="none:zero memory input;last:repeat latest observed error frame20times;full:all20error frames. Identical allocated parameters, initialization and minibatches; input information rank differs intentionally. Not an exact operator/active-function-capacity match to the historical same-lead local adapter.",
        fitting="Load frozen expanded_rich core;2000matched continuationupdates,seed42,batch4,AdamW3e-4cosine1e-6,clipgrad1. Common20%availabilitydropout represents coldstart; one fixed rate,not tuned.",
        causality="Memory=current known normalizedhistory minus actually issued preceding prediction. Validate archived180 and generate missing380from history380:400before loading400history. Current future labels excluded from feature construction.",
        calibration="All weights freeze first; independently recalibrate10original calibrationtrajectories for every rawriskhead. No old scale-forecast coefficients reused.",
        later_gate="Before development/test, evaluate600..760same originaltrainIDs after580coldprelude. At least one last/full policy under one fixed parentcontroller must improve actualScore and top10energy versus matchednone, without worse meanabsoluteRe/channelcoverageerror,tolerance1e-6. Full-history-specific benefit separately checked againstlast. Retain all heads/controllers/metrics when proceeding.",
        limits="History-reused diagnostic,equalupdates not equalFLOP. Pretrained sharedcore already fit same819pairs. Full20jointcoverage or formalCP not claimed."))
    norm = normalization()
    data, scale = prepare(norm)
    train(data, scale)
    del data
    provider = Provider(norm)
    scores = {name: [] for name in NAMES}
    with torch.no_grad():
        for start in range(0, 161, 20):
            point, bounds = provider.bounds("calibration", start)
            residual = (target("calibration", start) - norm["mean"]) / norm["std"] - point
            for name, (lo, hi) in bounds.items():
                scores[name].append(((residual - .5 * (lo + hi)).abs() / (.5 * (hi - lo)).clamp_min(1e-6)).cpu())
    initial = {name: {"separable_integral": initial_state(torch.cat(values).cuda(), "separable_integral")}
               for name, values in scores.items()}
    torch.save({name: {mode: {key: value.cpu() for key, value in state.items()} for mode, state in modes.items()}
                for name, modes in initial.items()}, OUT / "calibration.pt")
    old = load(SOURCE / "calibration.pt")["expanded_rich"]["separable_integral"]
    assert all(torch.equal(value, old[key]) for key, value in initial["frozen_rich"]["separable_integral"].items())
    dump(OUT / "calibration_locked.json", dict(time=time.time(), sha256=digest(OUT / "calibration.pt"), frozen_rich_calibration_exact=True))
    assert all(digest(ROOT / path) == sha for path, sha in hashes.items())
    dump(OUT / "complete.json", dict(time=time.time(), original_neural_models_unchanged=True, evaluation_not_yet_run=True))
    print("FULLBLOCK_MEMORY_TRAIN_CALIBRATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
