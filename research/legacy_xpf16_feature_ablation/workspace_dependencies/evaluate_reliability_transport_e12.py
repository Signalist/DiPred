"""Causal E1/E2 evaluation: ranking, intervals, and complete-block diagnostics."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from experiment_reliability_transport_e12 import (OUT, KINDS, PREDICTOR,
    RiskProvider, geometry, history, target, issued, event)
from reliability_risk_transport import CausalRiskMemory


def allocation(score, fraction, high):
    """Fractional ties make the constant-map null exactly an uninformative map."""
    quantile = torch.quantile(score, 1 - fraction if high else fraction, dim=-1, keepdim=True)
    selected = score > quantile if high else score < quantile
    ties = score == quantile
    amount = ((fraction * score.shape[-1] - selected.sum(-1, keepdim=True))
              / ties.sum(-1, keepdim=True).clamp_min(1)).clamp(0, 1)
    return selected.to(score.dtype) + ties * amount


def ranking(profile, error):
    flat = profile.flatten(2)
    energy = error.square().flatten(2)
    total = energy.sum(-1).clamp_min(1e-20)
    top = allocation(flat, .1, True)
    captured = (top * energy).sum(-1) / total
    truth_top = allocation(energy, .1, True)
    recall = (top * truth_top).sum(-1) / (.1 * energy.shape[-1])
    fractions = (.1, .25, .5, .75, 1.)
    retained = [(allocation(flat, fraction, False) * energy).sum(-1) / (fraction * total)
                for fraction in fractions]
    area = sum(.5 * (retained[i] + retained[i + 1]) * (fractions[i + 1] - fractions[i])
               for i in range(4)) / .9
    spatial_score = profile.flatten(3)
    spatial_energy = error.square().flatten(3)
    spatial_top = allocation(spatial_score, .1, True)
    frame_capture = ((spatial_top * spatial_energy).sum(-1)
                     / spatial_energy.sum(-1).clamp_min(1e-20)).mean(-1)
    predicted_lead = profile.mean((3, 4))
    actual_lead = error.square().mean((3, 4))
    predicted_lead = predicted_lead - predicted_lead.mean(-1, keepdim=True)
    actual_lead = actual_lead - actual_lead.mean(-1, keepdim=True)
    lead_correlation = (predicted_lead * actual_lead).sum(-1) / (
        predicted_lead.square().sum(-1) * actual_lead.square().sum(-1)).sqrt().clamp_min(1e-12)
    return dict(top10_energy=captured, top10_peak_recall=recall,
                per_frame_top10_energy=frame_capture, relative_aurc=area,
                lead_risk_error_correlation=lead_correlation,
                **{f"retained_risk{int(fraction * 100)}": value for fraction, value in zip(fractions, retained)})


def interval_metrics(profile, multiplier, error):
    radius = profile * multiplier[:, :, None, None, None]
    inside = error <= radius
    coverage = inside.float().mean((2, 3, 4))
    frame_coverage = inside.float().mean((3, 4))
    # Fixed8x8-grid regions; minimum after averaging over physical forecast time.
    patch_coverage = inside.reshape(len(inside), 2, 20, 4, 8, 8, 8).float().mean((2, 4, 6))
    joint = inside.flatten(1).all(-1)[:, None].expand(-1, 2)
    return dict(coverage=coverage, standardized_width=2 * radius.mean((2, 3, 4)),
                standardized_interval_score=(2 * radius + 20 * (error - radius).clamp_min(0)).mean((2, 3, 4)),
                window95=(coverage >= .95).float(), window90=(coverage >= .9).float(),
                window_all=inside.flatten(2).all(-1).float(), whole_block_joint_covered=joint.float(),
                worst_frame_coverage=frame_coverage.min(-1).values,
                worst_patch_coverage=patch_coverage.flatten(2).min(-1).values)


def summarize(rows):
    buckets = {}
    keys = ("kind", "split", "period", "group", "channel", "map", "variant")
    metrics = [key for key, value in rows[0].items() if isinstance(value, float)]
    for row in rows:
        for period in (row["period"], "warm" if row["start"] >= 20 else "cold"):
            key = tuple(period if name == "period" else row[name] for name in keys)
            buckets.setdefault(key, []).append(row)
    grouped = []
    for key, values in buckets.items():
        grouped.append(dict(zip(keys, key), **{metric: float(np.mean([row[metric] for row in values])) for metric in metrics}))
    macro = {}
    for row in grouped:
        key = tuple(row[name] for name in keys if name != "group")
        macro.setdefault(key, []).append(row)
    for key, values in macro.items():
        group_keys = tuple(name for name in keys if name != "group")
        result = dict(zip(group_keys, key), group="macro",
                      **{metric: float(np.mean([r[metric] for r in values])) for metric in metrics})
        result["worst_group_coverage"] = min(row["coverage"] for row in values)
        grouped.append(result)
    return grouped


@torch.no_grad()
def evaluate():
    configure()
    locked = json.loads((OUT / "calibration_locked.json").read_text())
    frozen = json.loads((OUT / "controls_frozen.json").read_text())
    assert all(digest(OUT / name) == sha for name, sha in frozen["artifacts"].items())
    dump(OUT / "evaluation_protocol.json", dict(time=time.time(), source_sha256=digest(Path(__file__)),
         target=.9, variants=["fixed", "online_geometric"], spatial_region=[8, 8],
         risk_ranking="Top10% error energy, top10% true-peak recall, per-frame spatial capture; ties fractionally allocated.",
         selective_risk="MSE retained at10/25/50/75/100% coverage divided by full MSE; AURC normalized over[.1,1].",
         block_diagnostics="Frequency of windows with>=95%point coverage and complete2channel20frame joint coverage; no guarantee.",
         primary="Warm windows exclude first cold-start window; also report main/confirmation/cold. No target-dependent map selection."))
    rows, leads = [], []
    for kind in KINDS:
        provider = RiskProvider(kind)
        factors = locked["factors"][kind]
        for split in ("development", "test"):
            records = manifest()["files"][split]
            geo = geometry(records)
            memory = CausalRiskMemory(provider.norm["std"])
            scales = {name: torch.tensor(value, device="cuda")[None].expand(len(records), 2).clone()
                      for name, value in factors.items()}
            for start in range(0, 761, 20):
                h, p = history(split, start), issued(kind, split, start)
                memories, available = memory.issue(h, p["mean"], geo)
                profiles = provider.profiles(h, p, memories, available, geo)
                before = {name: value.clone() for name, value in scales.items()}
                if split == "test" and start == 760:
                    (OUT / "snapshots").mkdir(exist_ok=True)
                    torch.save(dict(mean=p["mean"].cpu(),
                        profiles={name: value.cpu() for name, value in profiles.items()},
                        scales={name: value.cpu() for name, value in before.items()}),
                        OUT / "snapshots" / f"{kind}_{start}.pt")
                event(event="risk_intervals_issued", kind=kind, split=split, start=start,
                      previous_feedback_windows=memory.revealed_windows)
                y = target(split, start)
                event(event="risk_target_revealed", kind=kind, split=split, start=start)
                error = ((y - p["mean"]) / provider.norm["std"]).abs()
                raw_mse = (y - p["mean"]).square().mean((2, 3, 4))
                for name, profile in profiles.items():
                    location = ranking(profile, error)
                    for variant in ("fixed", "online_geometric"):
                        multiplier = before[name] if variant == "online_geometric" else profile.new_tensor(factors[name])[None].expand(len(records), 2)
                        metrics = interval_metrics(profile, multiplier, error)
                        by_lead = (error <= profile * multiplier[:, :, None, None, None]).float().mean((3, 4))
                        for i, record in enumerate(records):
                            for c, channel in enumerate(("u", "v")):
                                rows.append(dict(kind=kind, split=split, file=record["file"], group=record["group"],
                                     start=start, period="confirmation" if start >= 580 else "main", channel=channel,
                                     map=name, variant=variant, mse=float(raw_mse[i, c]),
                                     **{key: float(value[i, c]) for key, value in {**location, **metrics}.items()}))
                        # Aggregate leads by Re first, then average Re groups.
                        for c, channel in enumerate(("u", "v")):
                            mean = torch.stack([by_lead[[i for i, r in enumerate(records) if r["group"] == group], c].mean(0)
                                                for group in manifest()["groups"][split]]).mean(0)
                            for lead in range(20):
                                leads.append(dict(kind=kind, split=split, start=start, map=name, variant=variant,
                                                  channel=channel, lead=lead + 1, coverage=float(mean[lead])))
                    q = torch.quantile((error / profile).flatten(2), .9, dim=2)
                    scales[name] = (before[name] * q.clamp(1e-6, 1e3)).sqrt()
                memory.reveal(y)
                if start in (180, 380, 580, 760):
                    event(event="risk_replay_progress", kind=kind, split=split, start=start)
                    print("EVALUATE", kind, split, start, flush=True)
    dump(OUT / "per_case.json", rows)
    dump(OUT / "per_lead_coverage.json", leads)
    dump(OUT / "summary.json", summarize(rows))
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert all(digest(PREDICTOR / arm / "model.pt") == sha for arm, sha in protocol["predictor_sha256"].items())
    assert all(digest(OUT / name) == sha for name, sha in frozen["artifacts"].items())
    dump(OUT / "complete.json", dict(time=time.time(), point_predictors_unchanged=True,
         risk_controls_unchanged=True, rows=len(rows), lead_rows=len(leads)))


if __name__ == "__main__":
    evaluate()
