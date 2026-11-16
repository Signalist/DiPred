"""Train-only fixed-budget noise/parameterization controls, then frozen replay.

Run after source env.sh with TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0.
All three endpoints are frozen before any calibration/development/test access.
"""

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from clean_reliability_common import ROOT, SOURCE, configure, digest, dump, load_field, manifest
from confirm_clean_reliability_cache import field_file
from experiment_clean_physical_direct20 import new_model as original_model
from physical20_noise_refinement import ARMS, NoiseRefinement20
from retrain_clean_direct20 import train_steps


OUT = ROOT / "runs/physical20_noise_refinement_20261109"
BASE = ROOT / "runs/clean_physical_direct20_20261109"


def event(**values):
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time=time.time(), **values)) + "\n")


def new_model(arm):
    torch.manual_seed(42)
    return NoiseRefinement20(arm).cuda()


def preflight():
    reference = original_model("state_diffusion")
    reference_weights = {k: v.detach().clone() for k, v in reference.net.state_dict().items()}
    sigma = (1 - reference.abar).sqrt().cpu().tolist()
    del reference
    checks = []
    for arm in ARMS:
        model = new_model(arm)
        assert all(torch.equal(v, reference_weights[k]) for k, v in model.net.state_dict().items())
        assert model.abar[-1] == 0 and torch.all(model.abar[:-1] > 0)
        h = torch.randn(4, 2, 20, 32, 64, device="cuda")
        y, eps = torch.randn_like(h), torch.randn_like(h)
        t = torch.arange(4, device="cuda")
        a = model.abar[t, None, None, None, None]
        x = a.sqrt() * y + (1 - a).sqrt() * eps
        v = model.net(x, t * 33, h)
        clean = a.sqrt() * x - (1 - a).sqrt() * v
        target = a.sqrt() * eps - (1 - a).sqrt() * y
        lhs = (v - target).square().mean()
        rhs = ((clean - y).square() / (1 - a)).mean()
        assert torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-6)
        called = []
        hook = model.net.register_forward_pre_hook(lambda module, args: called.append(list(args[0].shape)))
        with torch.no_grad():
            output = model.rollout(h, 500000, sign=0)
        hook.remove()
        assert called == [[4, 2, 20, 32, 64]] * 4
        assert output.shape == h.shape and torch.isfinite(output).all()
        loss = model.loss(h, y)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        checks.append(dict(arm=arm, sigma=(1 - model.abar).sqrt().cpu().tolist(),
                           loss_equivalence_error=float((lhs - rhs).abs()),
                           network_input_shapes=called, parameters=sum(p.numel() for p in model.parameters())))
        del model
    dump(OUT / "preflight.json", dict(passed=True, identical_initial_net=True,
                                     baseline_sigma=sigma, checks=checks))


def train():
    event(event="read_train_only", files=[r["file"] for r in manifest()["files"]["train"]], frames=[0, 200])
    raw = torch.cat([load_field("history", "train", s) for s in range(0, 161, 20)]
                    + [load_field("target", "train", 160)], dim=2)
    assert raw.shape == (47, 2, 200, 32, 64)
    for i, record in enumerate(manifest()["files"]["train"]):
        assert hashlib.sha256(raw[i].cpu().numpy().tobytes()).hexdigest() == record["sha256_first200"]
    norm = torch.load(SOURCE / "normalization.pt", map_location="cuda", weights_only=False)
    fields = (raw - norm["mean"]) / norm["std"]
    histories = torch.cat([fields[:, :, s:s + 20] for s in range(0, 161, 5)])
    futures = torch.cat([fields[:, :, s + 20:s + 40] for s in range(0, 161, 5)])
    assert len(histories) == 1551
    del raw, fields
    for arm in ARMS:
        folder = OUT / arm
        folder.mkdir()
        model = new_model(arm)

        def objective():
            ix = torch.randint(len(histories), (8,), device="cuda")
            return model.loss(histories[ix], futures[ix]), {}

        train_steps(model, objective, 6000, folder, arm, OUT)
        torch.save(dict(model=model.state_dict(), arm=arm, seed=42, steps=6000,
                        fit_split="train", manifest_sha256=digest(SOURCE / "manifest.json")), folder / "model.pt")
        event(event="endpoint_frozen", arm=arm, sha256=digest(folder / "model.pt"))
        del model
        torch.cuda.empty_cache()
    dump(OUT / "frozen.json", dict(time=time.time(), artifacts={arm: digest(OUT / arm / "model.pt") for arm in ARMS}))


@torch.no_grad()
def issue():
    assert (OUT / "frozen.json").exists()
    norm = torch.load(SOURCE / "normalization.pt", map_location="cuda", weights_only=False)
    # Sample one model at a time to keep memory bounded. No targets are opened.
    for arm in ARMS:
        model = new_model(arm).eval().requires_grad_(False)
        model.load_state_dict(torch.load(OUT / arm / "model.pt", map_location="cuda", weights_only=False)["model"])
        folder = OUT / "issued" / arm
        folder.mkdir(parents=True)
        for split in ("calibration", "development", "test"):
            for start in range(0, 161 if split == "calibration" else 761, 20):
                observed = torch.load(field_file("history", split, start), map_location="cuda", weights_only=False)
                history = (observed - norm["mean"]) / norm["std"]
                zero = model.rollout(history, 500000, sign=0) * norm["std"] + norm["mean"]
                samples = torch.stack([model.rollout(history, seed) * norm["std"] + norm["mean"]
                                       for seed in range(500000, 500032)])
                assert torch.isfinite(samples).all() and torch.isfinite(zero).all()
                # Sample detail is retained separately from mean-field detail.
                difference = .5 * (samples.diff(dim=-1).square().mean((3, 4, 5))
                                    + samples.diff(dim=-2).square().mean((3, 4, 5)))
                torch.save(dict(zero=zero.cpu(), mean=samples.mean(0).cpu(), std32=samples.std(0).cpu(),
                                sample_difference_energy=difference.mean(0).cpu()), folder / f"{split}_{start}.pt")
                event(event="issued", arm=arm, split=split, start=start)
                del samples
            print("ISSUED", arm, split, flush=True)
        del model
    dump(OUT / "sampling_complete.json", dict(time=time.time(), windows_per_arm=87, paths=32))


def gradient_energy(x):
    return .5 * (x.diff(dim=-1).square().mean((2, 3, 4)) + x.diff(dim=-2).square().mean((2, 3, 4)))


def spectral(x):
    # Window nonperiodic boundaries identically for truth and all forecasts.
    ny, nx = x.shape[-2:]
    window = torch.hann_window(ny, periodic=False, device=x.device)[:, None] * torch.hann_window(nx, periodic=False, device=x.device)[None, :]
    transformed = torch.fft.rfft2((x - x.mean((-2, -1), keepdim=True)) * window, norm="ortho")
    fy = torch.fft.fftfreq(ny, device=x.device)[:, None]
    fx = torch.fft.rfftfreq(nx, device=x.device)[None, :]
    mask = (fy.square() + fx.square()).sqrt() >= .25
    return transformed[..., mask]


def summarize(rows):
    result = []
    for split in ("development", "test"):
        for period in ("main", "confirmation"):
            for arm in sorted({r["arm"] for r in rows}):
                for channel in ("u", "v", "both"):
                    for group in [*manifest()["groups"][split], "macro"]:
                        selected = [r for r in rows if r["split"] == split and r["period"] == period and r["arm"] == arm
                                    and (channel == "both" or r["channel"] == channel)
                                    and (group == "macro" or r["group"] == group)]
                        keys = [k for k, v in selected[0].items() if isinstance(v, float)]
                        groups = {r["group"] for r in selected}
                        values = {k: float(np.mean([np.mean([r[k] for r in selected if r["group"] == g]) for g in groups])) for k in keys}
                        values["difference_energy_ratio"] = values["difference_energy"] / values["truth_difference_energy"]
                        values["high_power_ratio"] = values["high_power"] / values["truth_high_power"]
                        result.append(dict(split=split, period=period, group=group, arm=arm, channel=channel, **values))
    return result


@torch.no_grad()
def evaluate():
    assert (OUT / "sampling_complete.json").exists()
    norm = torch.load(SOURCE / "normalization.pt", map_location="cuda", weights_only=False)
    ratios = {arm: [[], []] for arm in ARMS}
    for start in range(0, 161, 20):
        target = torch.load(field_file("target", "calibration", start), map_location="cuda", weights_only=False)
        for arm in ARMS:
            issued = torch.load(OUT / "issued" / arm / f"calibration_{start}.pt", map_location="cuda", weights_only=False)
            value = (target - issued["mean"]).abs() / issued["std32"].clamp_min(norm["std"] * 1e-4)
            for c in range(2):
                ratios[arm][c].append(value[:, c].flatten().cpu())
    factors = {arm: [float(torch.quantile(torch.cat(values).cuda(), .9)) for values in channels]
               for arm, channels in ratios.items()}
    dump(OUT / "calibration_locked.json", factors)
    rows = []
    for split in ("development", "test"):
        scales = {}
        for start in range(0, 761, 20):
            event(event="replay_predictions", split=split, start=start)
            old = torch.load(BASE / "issued" / f"{split}_{start}.pt", map_location="cuda", weights_only=False)
            new = {arm: torch.load(OUT / "issued" / arm / f"{split}_{start}.pt", map_location="cuda", weights_only=False) for arm in ARMS}
            target = torch.load(field_file("target", split, start), map_location="cuda", weights_only=False)
            event(event="reveal_target", split=split, start=start)
            spectrum_y = spectral(target)
            truth_gradient = gradient_energy(target)
            points = dict(baseline_zero=old["state_diffusion"]["zero"], baseline_mean32=old["state_diffusion"]["mean"],
                          state_cnn=old["state_cnn"]["mean"], delta_cnn=old["delta_cnn"]["mean"])
            for arm, issued in new.items():
                points[arm + "_zero"] = issued["zero"]
                points[arm + "_mean32"] = issued["mean"]
            for name, point in points.items():
                error = (point - target).square().mean((2, 3, 4))
                gradient = gradient_energy(point)
                spectrum = spectral(point)
                values = dict(mse=error, standardized_mse=error / norm["std"].flatten()[None].square(),
                              difference_energy=gradient, truth_difference_energy=truth_gradient,
                              high_power=spectrum.abs().square().mean((2, 3)),
                              truth_high_power=spectrum_y.abs().square().mean((2, 3)),
                              high_error=(spectrum - spectrum_y).abs().square().mean((2, 3)))
                if name.endswith("_mean32") and name.removesuffix("_mean32") in ARMS:
                    arm = name.removesuffix("_mean32")
                    issued = new[arm]
                    fixed = target.new_tensor(factors[arm]).reshape(1, 2, 1, 1, 1)
                    if arm not in scales:
                        scales[arm] = fixed.expand(len(target), 2, 1, 1, 1).clone()
                    sd = issued["std32"].clamp_min(norm["std"] * 1e-4)
                    absolute = (point - target).abs()
                    radius = sd * scales[arm]
                    values.update(fixed_coverage=(absolute <= fixed * sd).float().mean((2, 3, 4)),
                                  online_coverage=(absolute <= radius).float().mean((2, 3, 4)),
                                  interval_score_standardized=((2 * radius + 20 * (absolute - radius).clamp_min(0)) / norm["std"]).mean((2, 3, 4)),
                                  sample_difference_energy=issued["sample_difference_energy"])
                    q = torch.quantile((absolute / sd).flatten(2), .9, dim=2).reshape(len(target), 2, 1, 1, 1)
                    scales[arm] = (scales[arm] * q.clamp(1e-6, 1e3)).sqrt()
                for i, record in enumerate(manifest()["files"][split]):
                    for c, channel in enumerate(("u", "v")):
                        rows.append(dict(split=split, period="confirmation" if start >= 580 else "main", start=start,
                                         file=record["file"], group=record["group"], channel=channel, arm=name,
                                         **{k: float(v[i, c]) for k, v in values.items()}))
    dump(OUT / "per_case.json", rows)
    dump(OUT / "summary.json", summarize(rows))
    print("EVALUATED", len(rows), "rows", flush=True)


def main():
    configure()
    assert os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "0"
    OUT.mkdir(exist_ok=False)
    dump(OUT / "protocol.json", dict(
        time=time.time(), arms=ARMS, seed=42, batch=8, updates=6000, fit_pairs=1551,
        fit_frames=[0, 200], fit_split="train", fit_trajectories=47,
        architecture="Unchanged physical U-Net, 1597672 parameters; all20 frames at all4 reverse levels.",
        hypothesis="Existing lowest VP sigma~0.39 may obscure weak detail; test fixed sigma .01,.0464,.2154,1 with native v skip.",
        controls="Historical clean/cosine baseline; new v/cosine, v/low-noise, clean/low-noise factorial arms.",
        initialization="Identical seed42 unfitted network weights and training RNG as historical physical baseline.",
        fitting="Only audited train tensors and train-only normalization. No fitted predictor loads, crossfolds, distillation or extra seeds.",
        selection="All fixed6000 endpoints frozen before any held-out field access; evaluate all arms, no development selection.",
        sampling="32 fixed paths500000:500032 plus zero initial noise; no targets read until every forecast saved.",
        metrics="Raw/standardized MSE, grid-difference power, Hann-windowed frequencies>=.25cycles/grid power AND phase-sensitive error, marginal intervals.",
        limitations="Historically reused data; exploratory single seed. This adapts known low-noise refinement ideas, not a novelty claim.",
        references=["https://arxiv.org/abs/2308.05732", "https://arxiv.org/abs/2506.10711"],
        manifest_sha256=digest(SOURCE / "manifest.json"), normalization_sha256=digest(SOURCE / "normalization.pt"),
        code_sha256={name: digest(ROOT / name) for name in (Path(__file__).name, "physical20_noise_refinement.py")},
    ))
    preflight()
    train()
    issue()
    evaluate()
    frozen = json.loads((OUT / "frozen.json").read_text())
    assert all(digest(OUT / arm / "model.pt") == sha for arm, sha in frozen["artifacts"].items())
    dump(OUT / "complete.json", dict(time=time.time(), passed=True))


if __name__ == "__main__":
    main()
