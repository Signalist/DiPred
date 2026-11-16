"""Later-training-identity pilot for error-location and tail-scale drift.

One cold prelude580->600:620 precedes warm query windows600..760. Every
known packet is saved before reading its complete target. No calibration,
development, or test identities are used to fit diagnostic regressions.
"""

import json
import time

import torch
import torch.nn.functional as F
import numpy as np

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from experiment_reliability_transport_e12 import FIT_RECORDS, PREDICTOR, geometry
from infer_structured_fourstep_uncertainty import StructuredFourStepPredictor
from experiment_extended_train_rich_risk import OUT as SOURCE
from tail_scale_risk_adapter import TailScaleRiskAdapter
from fourstep_cue_tail_adapter import CueTailAdapter
from fourstep_denoising_risk_cues import cue_inputs
from experiment_continue_rich_uncertainty import controller
from conditional_online_risk import risk_features
from diagnose_train_content_memory import read_block


OUT = ROOT / "runs/train_distribution_drift_20261109"
RECORDS = [r for r in FIT_RECORDS if r["frames"] >= 800]
STARTS = tuple(range(580, 761, 20))
PRESETS = {
    "balanced": ("core_separate_all", "scale_fastcal"),
    "mean_score": ("core_history_old", "memory_dynamic"),
}
CONTEXT_NAMES = ("log_width_gm", "log_history_variability_rms", "log_departure_rms",
                 "log_forecast_change_rms", "log_forecast_gradient_rms",
                 "forecast_mean", "history_last_mean", "midpoint_over_width_mean")


def context_summary(history, point, lower, upper):
    """Absolute known-field summaries, before per-window feature zscoring."""
    midpoint = 0.5 * (lower + upper)
    width = (0.5 * (upper - lower)).clamp_min(1e-6)

    def log_rms(value):
        return value.square().mean(tuple(range(2, value.ndim))).clamp_min(1e-12).sqrt().log()

    variability = history.std(dim=2, keepdim=True)
    departure = point - history[:, :, -1:]
    change = torch.cat([point[:, :, :1] - history[:, :, -1:], point.diff(dim=2)], 2)
    dx = F.pad(point.diff(dim=-1), (0, 1, 0, 0))
    dy = F.pad(point.diff(dim=-2), (0, 0, 0, 1))
    gradient = (dx.square() + dy.square()).sqrt()
    values = (width.log().mean((2, 3, 4)), log_rms(variability), log_rms(departure),
              log_rms(change), log_rms(gradient), point.mean((2, 3, 4)),
              history[:, :, -1].mean((2, 3)), (midpoint / width).mean((2, 3, 4)))
    return torch.stack(values, -1)


def distribution_targets(residual, midpoint, radius):
    ratio = ((residual - midpoint) / radius.clamp_min(1e-6)).flatten(2)
    q = torch.quantile(ratio, ratio.new_tensor([0.05, 0.50, 0.95]), dim=-1)
    tail = torch.quantile(ratio.abs(), 0.90, dim=-1).clamp_min(1e-6).log()
    return dict(location=0.5 * (q[0] + q[2]), mean=ratio.mean(-1), median=q[1],
                log_tail=tail, q05=q[0], q95=q[2])


@torch.no_grad()
def main():
    configure()
    OUT.mkdir(exist_ok=False)
    (OUT / "known").mkdir()
    (OUT / "labels").mkdir()
    assert len(RECORDS) == 44
    ids = {r["file"] for r in RECORDS}
    assert all(not ids.intersection(r["file"] for r in records)
               for split, records in manifest()["files"].items() if split != "train")
    frozen_paths = [PREDICTOR / "augmented_diffusion/model.pt", SOURCE / "expanded_rich.pt",
                    SOURCE / "calibration.pt", ROOT / "runs/structured_quantile_uncertainty_20261109/statistics.pt"]
    frozen_paths += [SOURCE / (name + ".pt") for name, _ in PRESETS.values()]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in frozen_paths}
    dump(OUT / "protocol.json", dict(time=time.time(), records=RECORDS, starts=STARTS,
        presets=PRESETS, context_names=CONTEXT_NAMES, frozen_artifacts=hashes,
        source_sha256=digest(ROOT / "diagnose_train_distribution_drift.py"),
        native_forecasts="Shared original4calls99,66,33,0,joint20->20. Frozen point/core/adapter weights. Cold prelude580 then warm600..760. Only originaltraining identities, all neural fitting ends600.",
        labels="Raw standardized errors about raw interval midpoint:tailcenter=(q05+q95)/2,mean,median,logq90absolute. Calibrated analogues reported separately; neither is model/physical randomness.",
        causal_cache="Save observedhistory/point/rawrichbounds in known packet before reading any currenttarget. Save residual in separate labels file after whole20reveal; no mixed future-label feature packet.",
        frozen_diagnostic="Fit innovation regression on query620,640,660,680; later query700,720,740,760 only diagnostic. Separate head/channel coefficients,ridge0.25,unpenalizedintercept,contextdelta mean/std floor0.05,clipz5. Predictpreviouslabel+innovation; caplocationinnovation+-0.5,logtailinnovation+-log2. Comparezero(location),hold,EMA0.5. No grids,folds,extra seeds or evaluation-identity fitting.",
        limits="Chronological same-training-identity pilot, not independent trajectory confirmation. All main eval data reused; this pilot does not change96primarymetrics or defaults."))
    predictor = StructuredFourStepPredictor(geometry(RECORDS))
    predictor.head.load_state_dict(torch.load(SOURCE / "expanded_rich.pt", map_location="cuda", weights_only=False)["model"])
    initial = torch.load(SOURCE / "calibration.pt", map_location="cuda", weights_only=False)
    heads, states = {}, {}
    for preset, (name, mode) in PRESETS.items():
        head = CueTailAdapter() if "history" in name else TailScaleRiskAdapter(True)
        head.load_state_dict(torch.load(SOURCE / (name + ".pt"), map_location="cuda", weights_only=False)["model"])
        heads[preset] = head.cuda().eval().requires_grad_(False)
        states[preset] = controller(mode, initial[name]["separable_integral"], len(RECORDS))
    values = {name: [] for name in PRESETS}
    for step, start in enumerate(STARTS):
        observed = read_block(RECORDS, start)
        calls = []
        hook = predictor.model.net.register_forward_pre_hook(lambda module, args: calls.append(int(args[1][0])))
        try:
            point, rich_lo, rich_hi = predictor.forecast(observed)
        finally:
            hook.remove()
        assert calls == [99, 66, 33, 0]
        h = (observed - predictor.norm["mean"]) / predictor.norm["std"]
        known = dict(observed=observed.cpu(), prediction=point.cpu(),
                     rich_lower=rich_lo.cpu(), rich_upper=rich_hi.cpu())
        torch.save(known, OUT / "known" / (str(start) + ".pt"))
        bounds, issued, contexts = {}, {}, {}
        x = cue_inputs(h, point, rich_lo, rich_hi, None, "history_change", step > 0)
        for name, head in heads.items():
            lo, hi = head.bounds(x, rich_lo, rich_hi)
            bounds[name] = (lo, hi)
            contexts[name] = context_summary(h, point, lo, hi)
            issued[name] = states[name].issue(lo, hi, risk_features(h, point, 0.5 * (hi - lo)))
        with (OUT / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event="issued", start=start, time=time.time(),
                calls=calls, previous_feedback_windows=step)) + "\n")
        truth = read_block(RECORDS, start + 20)
        residual = (truth - predictor.norm["mean"]) / predictor.norm["std"] - point
        torch.save(residual.cpu(), OUT / "labels" / (str(start) + ".pt"))
        with (OUT / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event="revealed", start=start, time=time.time())) + "\n")
        for name, (lo, hi) in bounds.items():
            raw = distribution_targets(residual, 0.5 * (lo + hi), 0.5 * (hi - lo))
            lower, upper = issued[name]
            calibrated = distribution_targets(residual, 0.5 * (lower + upper), 0.5 * (upper - lower))
            inside = ((residual >= lower) & (residual <= upper)).float().mean((2, 3, 4))
            row = {key: value.cpu().numpy() for key, value in raw.items()}
            row.update({"calibrated_" + key: value.cpu().numpy() for key, value in calibrated.items()})
            row["context"] = contexts[name].cpu().numpy()
            row["coverage"] = inside.cpu().numpy()
            values[name].append(row)
            states[name].reveal(residual)
        print("PILOT", start, len(RECORDS), flush=True)
    for name, rows in values.items():
        np.savez_compressed(OUT / (name + "_statistics.npz"),
                            **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
    assert all(digest(ROOT / path) == sha for path, sha in hashes.items())
    dump(OUT / "complete.json", dict(time=time.time(), native_forecasts=440,
        full_issue_reveal_pairs=10, all_neural_weights_unchanged=True,
        calibration_development_test_identities_read=False))


if __name__ == "__main__":
    main()
