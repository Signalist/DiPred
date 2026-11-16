"""Frozen Direct20 risk heads with causal, structured interval calibration.

Regions and forecast leads group calibration scores only; the predictor still
generates all twenty future frames jointly. No development/test hyperparameter fit.
"""

import json
from pathlib import Path
import time

import torch

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from experiment_reliability_transport_e12 import (
    OUT as PREVIOUS, PREDICTOR, RiskProvider, geometry, history, target, issued,
)
from evaluate_reliability_transport_e12 import interval_metrics, summarize
from reliability_risk_transport import CausalRiskMemory


OUT = ROOT / "runs/reliability_structured_e3_20261109"
METHODS = ("global_fixed", "global_online", "global95_online",
           "lead_online", "patch_online", "lead_patch_online")


def grouped_quantile(ratio, grouping, probability=.9, pooled=False):
    """Quantiles within a trajectory, or pooled across calibration trajectories.

    Return broadcastable B,C,T,H,W scale arrays. Each tile is eight by eight.
    Pixel scores are correlated; these empirical quantiles have no CP guarantee.
    """
    b, c, t, h, w = ratio.shape
    tiles = ratio.reshape(b, c, t, 4, 8, 8, 8)
    if grouping == "global":
        values = ratio.permute(1, 0, 2, 3, 4).reshape(c, -1) if pooled else ratio.flatten(2)
        q = torch.quantile(values, probability, dim=-1)
        return q.reshape(1 if pooled else b, c, 1, 1, 1)
    if grouping == "lead":
        values = ratio.permute(1, 2, 0, 3, 4).reshape(c, t, -1) if pooled else ratio.flatten(3)
        q = torch.quantile(values, probability, dim=-1)
        return q.reshape(1 if pooled else b, c, t, 1, 1)
    if grouping == "patch":
        values = tiles.permute(1, 3, 5, 0, 2, 4, 6).reshape(c, 4, 8, -1) if pooled else tiles.permute(0, 1, 3, 5, 2, 4, 6).reshape(b, c, 4, 8, -1)
        q = torch.quantile(values, probability, dim=-1).reshape(1 if pooled else b, c, 1, 4, 8)
    else:
        assert grouping == "lead_patch"
        values = tiles.permute(1, 2, 3, 5, 0, 4, 6).reshape(c, t, 4, 8, -1) if pooled else tiles.permute(0, 1, 2, 3, 5, 4, 6).flatten(5)
        q = torch.quantile(values, probability, dim=-1).reshape(1 if pooled else b, c, t, 4, 8)
    return q.repeat_interleave(8, dim=3).repeat_interleave(8, dim=4)


def estimates(ratio, pooled=False):
    global_q = grouped_quantile(ratio, "global", pooled=pooled).clamp(1e-6, 1e3)
    values = dict(global_fixed=global_q, global_online=global_q,
                  global95_online=grouped_quantile(ratio, "global", .95, pooled).clamp(1e-6, 1e3))
    for grouping in ("lead", "patch", "lead_patch"):
        local = grouped_quantile(ratio, grouping, pooled=pooled).clamp(1e-6, 1e3)
        # Fixed equal log-weight shrinkage; not selected on development/test.
        values[grouping + "_online"] = (local * global_q).sqrt()
    return values


def event(**value):
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time=time.time(), **value)) + "\n")


@torch.no_grad()
def main():
    configure()
    OUT.mkdir(exist_ok=False)
    frozen = json.loads((PREVIOUS / "controls_frozen.json").read_text())
    assert all(digest(PREVIOUS / name) == sha for name, sha in frozen["artifacts"].items())
    predictor_hashes = {kind: digest(PREDICTOR / ("augmented_" + kind) / "model.pt") for kind in ("diffusion", "cnn")}
    dump(OUT / "protocol.json", dict(source_sha256=digest(Path(__file__)),
         methods=METHODS, target=.9, wider_global_control_target=.95,
         grouping="20 leads;32 fixed8x8 tiles; joint lead x tile",
         shrink_local_log_weight=.5, online_log_step=.5, clamp=[1e-6, 1e3],
         issue="All radii frozen before current target loads; update only after full20 reveal.",
         calibration="10 reserved trajectories,9 windows; pooled empirical quantiles, no formal guarantee.",
         heads="Frozen E1/E2 heads; no new training or development/test selection.",
         predictor_sha256=predictor_hashes))
    rows = []
    for kind in ("diffusion", "cnn"):
        provider = RiskProvider(kind)
        names = ("head_static", "head_advected", "spread_raw") if kind == "diffusion" else ("head_static", "head_advected")
        records = manifest()["files"]["calibration"]
        geo = geometry(records)
        memory = CausalRiskMemory(provider.norm["std"])
        ratios = {name: [] for name in names}
        for start in range(0, 161, 20):
            h, p = history("calibration", start), issued(kind, "calibration", start)
            maps, available = memory.issue(h, p["mean"], geo)
            profiles = provider.profiles(h, p, maps, available, geo)
            y = target("calibration", start)
            error = ((y - p["mean"]) / provider.norm["std"]).abs()
            for name in names:
                ratios[name].append((error / profiles[name]).cpu())
            memory.reveal(y)
        initial = {name: estimates(torch.cat(ratios[name]).cuda(), pooled=True) for name in names}
        torch.save({name: {method: q.cpu() for method, q in values.items()} for name, values in initial.items()}, OUT / (kind + "_calibration.pt"))
        del ratios
        for split in ("development", "test"):
            records = manifest()["files"][split]
            geo = geometry(records)
            memory = CausalRiskMemory(provider.norm["std"])
            state = {name: {method: q.expand(len(records), *q.shape[1:]).clone() for method, q in values.items()} for name, values in initial.items()}
            for start in range(0, 761, 20):
                h, p = history(split, start), issued(kind, split, start)
                maps, available = memory.issue(h, p["mean"], geo)
                profiles = provider.profiles(h, p, maps, available, geo)
                # Materialize every issued interval before reading current truth.
                radii = {name: {method: profiles[name] * q for method, q in values.items()} for name, values in state.items()}
                event(event="issued", kind=kind, split=split, start=start, feedback=memory.revealed_windows)
                y = target(split, start)
                event(event="revealed", kind=kind, split=split, start=start)
                error = ((y - p["mean"]) / provider.norm["std"]).abs()
                mse = (y - p["mean"]).square().mean((2, 3, 4))
                for name in names:
                    for method in METHODS:
                        metrics = interval_metrics(radii[name][method], error.new_ones(len(records), 2), error)
                        for i, record in enumerate(records):
                            for c, channel in enumerate(("u", "v")):
                                rows.append(dict(kind=kind, split=split, file=record["file"], group=record["group"], start=start,
                                     period="confirmation" if start >= 580 else "main", channel=channel,
                                     map=name, variant=method, mse=float(mse[i, c]),
                                     **{key: float(value[i, c]) for key, value in metrics.items()}))
                    update = estimates(error / profiles[name])
                    for method in METHODS:
                        if method != "global_fixed":
                            state[name][method] = (state[name][method] * update[method]).sqrt()
                memory.reveal(y)
                if start in (180, 380, 580, 760):
                    print(kind, split, start, flush=True)
    dump(OUT / "per_case.json", rows)
    dump(OUT / "summary.json", summarize(rows))
    assert all(digest(PREVIOUS / name) == sha for name, sha in frozen["artifacts"].items())
    assert all(digest(PREDICTOR / ("augmented_" + kind) / "model.pt") == sha for kind, sha in predictor_hashes.items())
    dump(OUT / "complete.json", dict(rows=len(rows), frozen_artifacts_unchanged=True, time=time.time()))


if __name__ == "__main__":
    main()
