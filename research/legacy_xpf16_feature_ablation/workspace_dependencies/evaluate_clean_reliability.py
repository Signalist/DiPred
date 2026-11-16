"""Causal replay of frozen risk maps with matched point predictions.

Truth-matched widths below are explicitly oracle diagnostics. Deployed interval
scales are fitted only on calibration trajectories or previous revealed windows.
"""

import json
import os
import time

import torch

from clean_reliability_common import (ROOT, OUT, SOURCE, KINDS, MAPS, configure,
    digest, dump, manifest, starts, load_field, load_issued, deterministic_maps,
    RiskMapNet, risk_input, normalize_map, period, macro, log_event)


class MapProvider:
    def __init__(self, kind):
        self.kind = kind
        self.norm = torch.load(SOURCE / "normalization.pt", map_location="cuda", weights_only=False)
        self.template = torch.load(OUT / "controls" / f"{kind}_template.pt", map_location="cuda", weights_only=False)
        self.statistics = json.loads((OUT / "controls/training_means.json").read_text())[kind]
        self.heads = {}
        for name in ("learned_risk", "learned_risk_spread"):
            head = RiskMapNet().cuda().eval().requires_grad_(False)
            ck = torch.load(OUT / "controls" / f"{kind}_{name}.pt", map_location="cuda", weights_only=False)
            assert ck["split"] == "train" and ck["manifest_sha256"] == digest(SOURCE / "manifest.json")
            head.load_state_dict(ck["model"])
            self.heads[name] = head

    @torch.no_grad()
    def maps(self, history, issued, previous_error):
        mu = issued["mean"]
        raw = deterministic_maps(history, mu)
        raw["train_template"] = self.template[None].expand_as(mu)
        raw["feedback_error"] = raw["train_template"] if previous_error is None else previous_error
        for source, name in (("std8", "diffusion_std8"), ("std32", "diffusion_std32"), ("quantile32", "diffusion_quantile32")):
            raw[name] = issued[source]
        for name, head in self.heads.items():
            inputs = risk_input(history, mu, issued["std32"], self.norm, name == "learned_risk_spread")
            raw[name] = head(inputs).reshape_as(mu).clamp(-9, 5).exp() * self.norm["std"]
        assert set(raw) == set(MAPS)
        return {name: normalize_map(value, torch.tensor(self.statistics[name], device="cuda")) for name, value in raw.items()}


def calibrate(kind, provider):
    scores = {name: [] for name in MAPS}
    previous = None
    for start in starts("calibration"):
        issued = load_issued(kind, "calibration", start)
        history = load_field("history", "calibration", start)
        profiles = provider.maps(history, issued, previous)
        # No target is passed into the map construction.
        error = (load_field("target", "calibration", start) - issued["mean"]).abs()
        for name, radius in profiles.items():
            scores[name].append((error / radius).permute(1, 0, 2, 3, 4).reshape(2, -1).cpu())
        previous = error
    factors = {}
    for name in MAPS:
        values = torch.cat(scores[name], dim=1).cuda()
        factor = torch.quantile(values, .9, dim=1)
        assert torch.isfinite(factor).all() and (factor > 0).all()
        factors[name] = factor.cpu().tolist()
    return factors


def localization_metrics(profile, error):
    flat = profile.flatten(2)
    energy = error.square().flatten(2)
    quantile = torch.quantile(flat, torch.tensor([.5, .9], device="cuda"), dim=2)
    n = flat.shape[2]
    # Fractional allocation of ties avoids spatial-order bias for constant maps.
    high = flat > quantile[1, :, :, None]
    equal = flat == quantile[1, :, :, None]
    fraction = ((.1 * n - high.sum(2)) / equal.sum(2).clamp_min(1)).clamp(0, 1)
    captured = ((energy * high).sum(2) + fraction * (energy * equal).sum(2)) / energy.sum(2).clamp_min(1e-20)
    low = flat < quantile[0, :, :, None]
    equal = flat == quantile[0, :, :, None]
    fraction = ((.5 * n - low.sum(2)) / equal.sum(2).clamp_min(1)).clamp(0, 1)
    retained = ((energy * low).sum(2) + fraction * (energy * equal).sum(2)) / (.5 * energy.sum(2)).clamp_min(1e-20)
    ratios = (error / profile).flatten(2)
    oracle_scale = torch.quantile(ratios, .9, dim=2)
    oracle_width = 2 * oracle_scale * flat.mean(2)
    oracle_coverage = (ratios <= oracle_scale[:, :, None]).float().mean(2)
    # Tied ratios can conservatively cover more than90%; never force artificial misses.
    assert oracle_coverage.min() >= .9 - 1 / n - 1e-6
    return dict(top10_energy=captured, retained_half_relative_mse=retained,
                oracle_width90=oracle_width, oracle_coverage90=oracle_coverage)


def interval_metrics(radius, error):
    excess = (error - radius).clamp_min(0)
    return dict(coverage=(error <= radius).float().mean((2, 3, 4)),
                width=2 * radius.mean((2, 3, 4)),
                interval_score=(2 * radius + 20 * excess).mean((2, 3, 4)))


@torch.no_grad()
def evaluate_kind(kind, provider, factors):
    rows = []
    for split in ("development", "test"):
        previous = None
        scale = {name: torch.tensor(factors[name], device="cuda")[None].expand(len(manifest()["files"][split]), -1).clone() for name in MAPS}
        for window, start in enumerate(starts(split), 1):
            issued = load_issued(kind, split, start)
            history = load_field("history", split, start)
            profiles = provider.maps(history, issued, previous)
            # Freeze current scales before this target is revealed to the replay.
            scales_before = {name: value.clone() for name, value in scale.items()}
            log_event(event="interval_issued", kind=kind, split=split, start=start,
                      previous_feedback_start=None if window == 1 else start - 20)
            signed_error = load_field("target", split, start) - issued["mean"]
            error = signed_error.abs()
            mse = error.square().mean((2, 3, 4))
            squared_mean_bias = signed_error.mean((2, 3, 4)).square()
            log_event(event="interval_target_revealed", kind=kind, split=split, start=start)
            for name, profile in profiles.items():
                location = localization_metrics(profile, error)
                for variant in ("fixed", "online"):
                    multiplier = torch.tensor(factors[name], device="cuda")[None] if variant == "fixed" else scales_before[name]
                    radius = profile * multiplier[:, :, None, None, None]
                    metrics = interval_metrics(radius, error)
                    std = provider.norm["std"].flatten()
                    for i, rec in enumerate(manifest()["files"][split]):
                        for c, channel in enumerate(("u", "v")):
                            rows.append(dict(method=kind, split=split, file=rec["file"], group=rec["group"],
                                             window=window, start=start, period=period(start), map=name, variant=variant,
                                             mse=float(mse[i, c]),
                                             mean_bias_energy_fraction=float(squared_mean_bias[i, c] / mse[i, c].clamp_min(1e-20)),
                                             channel=channel, coverage=float(metrics["coverage"][i, c]),
                                             width=float(metrics["width"][i, c]), interval_score=float(metrics["interval_score"][i, c]),
                                             width_normalized=float(metrics["width"][i, c] / std[c]),
                                             interval_score_normalized=float(metrics["interval_score"][i, c] / std[c]),
                                             oracle_width90=float(location["oracle_width90"][i, c]),
                                             oracle_coverage90=float(location["oracle_coverage90"][i, c]),
                                             oracle_width90_normalized=float(location["oracle_width90"][i, c] / std[c]),
                                             top10_energy=float(location["top10_energy"][i, c]),
                                             retained_half_relative_mse=float(location["retained_half_relative_mse"][i, c])))
                # Updates happen after both fixed and online intervals are scored.
                q = torch.quantile((error / profile).flatten(2), .9, dim=2).clamp(1e-6, 1e3)
                scale[name] = (scales_before[name] * q).sqrt()
            previous = error
            if start % 100 == 0:
                print("REPLAY", kind, split, start, flush=True)
            dump(OUT / "status.json", dict(stage="evaluate", kind=kind, split=split, start=start, pid=os.getpid()))
    return rows


def aggregate(rows):
    summaries = []
    for split in ("development", "test"):
        for kind in KINDS:
            for block in ("all", "early", "middle", "late"):
                for channel in ("u", "v"):
                    for name in MAPS:
                        for variant in ("fixed", "online"):
                            selected = [r for r in rows if r["split"] == split and r["method"] == kind and r["channel"] == channel
                                        and r["map"] == name and r["variant"] == variant and r["window"] >= 2
                                        and (block == "all" or r["period"] == block)]
                            values = {key: macro(selected, key) for key in ("coverage", "width_normalized", "interval_score_normalized",
                                       "oracle_width90_normalized", "top10_energy", "retained_half_relative_mse")}
                            names = sorted({r["file"] for r in selected})
                            case_gaps = []
                            for file in names:
                                sub = [r for r in selected if r["file"] == file]
                                case_gaps.append(dict(group=sub[0]["group"], gap=abs(sum(r["coverage"] for r in sub) / len(sub) - .9)))
                            values["trajectory_abs_coverage_gap"] = macro(case_gaps, "gap")
                            summaries.append(dict(split=split, method=kind, period=block, channel=channel, map=name, variant=variant, **values))
    return summaries


def main():
    configure()
    assert os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "0"
    frozen = json.loads((OUT / "controls_frozen.json").read_text())
    assert all(digest(OUT / p) == sha for p, sha in frozen["artifacts"].items())
    dump(OUT / "evaluation_locked.json", dict(code_sha256={name: digest(ROOT / name) for name in
        ("evaluate_clean_reliability.py", "clean_reliability_common.py")}, controls_frozen_sha256=digest(OUT / "controls_frozen.json")))
    factors = {}
    for kind in KINDS:
        provider = MapProvider(kind)
        factors[kind] = calibrate(kind, provider)
        del provider
    dump(OUT / "calibration_locked.json", factors)
    log_event(event="calibration_locked")
    rows = []
    for kind in KINDS:
        provider = MapProvider(kind)
        rows.extend(evaluate_kind(kind, provider, factors[kind]))
        del provider
        torch.cuda.empty_cache()
    dump(OUT / "per_case.json", rows)
    dump(OUT / "summary.json", aggregate(rows))
    dump(OUT / "status.json", dict(stage="complete", pid=os.getpid()))


if __name__ == "__main__":
    main()
