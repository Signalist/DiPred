"""Train-only prequential diagnostic of content-memory trust, without fitting.

Frozen native four-call forecasts are issued before reading each target. The
later training time segment is diagnostic data, not independent confirmation.
Patch labels concern dispersion around the raw interval midpoint, not point MSE.
"""

import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr

from clean_reliability_common import ROOT, DATA, configure, digest, dump, manifest
from experiment_reliability_transport_e12 import FIT_RECORDS, geometry
from infer_structured_fourstep_uncertainty import StructuredFourStepPredictor
from tail_scale_risk_adapter import TailScaleRiskAdapter, tail_inputs
from conditional_online_risk import risk_features
from content_matched_risk_memory import patch_keys, patch_log_scores
from spatial_content_risk_memory import SpatialContentRiskMemory


OUT = ROOT / "runs/train_content_memory_diagnostic_20261109"
SOURCE = ROOT / "runs/tail_scale_risk_adapter_20261109"
STARTS = tuple(range(400, 561, 20))


def read_block(records, start):
    rows = []
    for record in records:
        with h5py.File(DATA / record["file"], "r") as stream:
            raw = np.stack([stream[c][start:start + 20, ::2, ::2]
                            for c in ("u", "v")]).astype(np.float32)
        assert raw.shape == (2, 20, 32, 64) and np.isfinite(raw).all()
        rows.append(torch.from_numpy(raw))
    return torch.stack(rows).cuda()


def lookup(query, memory):
    """Independent diagnostic lookup, returning dispersion among neighbors."""
    batch, channels, tokens, dimensions = query.shape
    keys = torch.cat([item[0] for item in memory], dim=2)
    values = torch.stack([item[1] for item in memory], dim=2).flatten(2)
    q = query.flatten(0, 1)
    k = keys.flatten(0, 1)
    distances = (q.square().sum(-1, keepdim=True)
                 + k.square().sum(-1)[:, None]
                 - 2 * torch.bmm(q, k.transpose(1, 2))).clamp_min(0) / dimensions
    distance, indices = distances.topk(16, largest=False)
    labels = values.flatten(0, 1).gather(1, indices.flatten(1)).reshape_as(distance)
    weights = (-distance).softmax(-1)
    mean = (weights * labels).sum(-1)
    variance = (weights * (labels - mean[..., None]).square()).sum(-1)
    shape = (batch, channels, tokens)
    return (mean.reshape(shape), distance.mean(-1).reshape(shape),
            variance.reshape(shape))


def spatial_center(value):
    shaped = value.reshape(*value.shape[:2], 20, 32)
    return (shaped - shaped.mean(-1, keepdim=True)).reshape_as(value)


def summarize(arrays):
    truth = arrays["truth"]
    predicted = arrays["retrieved"]
    position = arrays["position"]
    error = (predicted - truth) ** 2
    benefit = truth ** 2 - error
    result = {
        "mse_zero": float(np.mean(truth ** 2)),
        "mse_content": float(np.mean(error)),
        "mse_position": float(np.mean((position - truth) ** 2)),
        "content_better_than_zero_fraction": float(np.mean(benefit > 0)),
        "note": "Descriptive patch statistics, not independent-pixel inference or interval-score gains.",
        "trust": {},
    }
    for name in ("distance", "neighbor_variance"):
        feature = arrays[name]
        # Correlations are reduced per trajectory/channel, avoiding a pooled
        # significance test that incorrectly treats patches as independent.
        correlations = []
        bins = [[] for _ in range(5)]
        for b in range(feature.shape[1]):
            for c in range(2):
                x = feature[:, b, c].ravel()
                y = error[:, b, c].ravel()
                correlations.append(float(spearmanr(x, y).correlation))
                order = np.argsort(x, kind="stable")
                for i, selected in enumerate(np.array_split(order, 5)):
                    bins[i].append(float(benefit[:, b, c].ravel()[selected].mean()))
        result["trust"][name] = {
            "trajectory_channel_mean_spearman_with_squared_error": float(np.mean(correlations)),
            "per_trajectory_channel_spearman": correlations,
            "mean_benefit_quintiles_low_to_high": np.mean(bins, axis=1).tolist(),
        }
    return result


@torch.no_grad()
def main():
    configure()
    # A failed preflight may leave an empty directory; never overwrite a run.
    if OUT.exists():
        assert not any(OUT.iterdir()), "Refuse to overwrite diagnostic artifacts."
    else:
        OUT.mkdir()
    records = [r for r in FIT_RECORDS if r["frames"] >= 600]
    excluded = [r for r in FIT_RECORDS if r["frames"] < 600]
    splits = manifest()["files"]
    names = {r["file"] for r in records}
    assert all(not names.intersection(r["file"] for r in splits[split])
               for split in ("calibration", "development", "test"))
    weights = [SOURCE / "separate_tails.pt",
               ROOT / "runs/structured_quantile_uncertainty_20261109/rich.pt",
               ROOT / "runs/physical20_augmented_dynamics_20261109/augmented_diffusion/model.pt"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in weights}
    dump(OUT / "protocol.json", dict(
        files=[r["file"] for r in records], excluded=excluded, starts=STARTS,
        target_frames=[420, 600], fitting=False, model_hashes=hashes,
        source_sha256=digest(Path(__file__)),
        purpose="Check whether pre-target retrieval distance and neighbor disagreement predict spatial log-risk retrieval error; no gating rule selected on dev/test.",
        limits="Same training identities, later times; not independent confirmation. Native4calls, joint20; no folds or sampling ensemble."))
    predictor = StructuredFourStepPredictor(geometry(records))
    adapter = TailScaleRiskAdapter(True).cuda().eval().requires_grad_(False)
    adapter.load_state_dict(torch.load(weights[0], map_location="cuda", weights_only=False)["model"])
    initial = torch.load(SOURCE / "calibration.pt", map_location="cuda", weights_only=False)
    controller = SpatialContentRiskMemory("memory_content_spatial",
        initial["separate_tails"]["separable_integral"], len(records))
    rows = {name: [] for name in ("truth", "retrieved", "position", "distance", "neighbor_variance")}
    events = []
    for start in STARTS:
        observed = read_block(records, start)
        labels = []
        handle = predictor.model.net.register_forward_pre_hook(
            lambda module, args: labels.append(int(args[1][0])))
        point, lower, upper = predictor.forecast(observed)
        handle.remove()
        assert labels == [99, 66, 33, 0], labels
        h = (observed - predictor.norm["mean"]) / predictor.norm["std"]
        lower, upper = adapter.bounds(tail_inputs(h, point, lower, upper, True), lower, upper)
        features = risk_features(h, point, 0.5 * (upper - lower))
        controller.issue(lower, upper, features)
        keys, radius, _ = controller.memory_pending
        query = patch_keys(features)
        assert torch.equal(query, keys)
        diagnostics = None
        if controller.memory:
            retrieved, distance, variance = lookup(query, controller.memory)
            position = torch.stack([item[1] for item in controller.memory], dim=2).mean(2)
            diagnostics = (spatial_center(retrieved), spatial_center(position), distance, variance)
        events.append(dict(start=start, event="issued", native_calls=labels,
                           memory_windows=len(controller.memory)))
        # The target is first read only after issuing the entire forecast and
        # computing every diagnostic feature available to a future trust gate.
        truth = read_block(records, start + 20)
        residual = (truth - predictor.norm["mean"]) / predictor.norm["std"] - point
        midpoint = 0.5 * (lower + upper)
        target = spatial_center(patch_log_scores((residual - midpoint).abs() / radius.clamp_min(1e-6)))
        if diagnostics is not None:
            retrieved, position, distance, variance = diagnostics
            for name, value in zip(rows, (target, retrieved, position, distance, variance)):
                rows[name].append(value.cpu().numpy())
        controller.reveal(residual)
        events.append(dict(start=start, event="revealed", memory_windows=len(controller.memory)))
        print("DONE", start, len(records), flush=True)
    arrays = {name: np.stack(values) for name, values in rows.items()}
    dump(OUT / "events.json", events)
    np.savez_compressed(OUT / "patch_diagnostics.npz", **arrays)
    dump(OUT / "summary.json", summarize(arrays))
    assert all(digest(ROOT / path) == sha for path, sha in hashes.items())
    dump(OUT / "complete.json", dict(time=time.time(), weights_unchanged=True,
        complete_issue_reveal_pairs=len(STARTS), forecast_examples=len(STARTS) * len(records)))


if __name__ == "__main__":
    main()
