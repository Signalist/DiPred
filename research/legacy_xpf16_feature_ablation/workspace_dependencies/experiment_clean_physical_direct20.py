"""Clean-split, uncompressed physical-field direct20 forecasting controls.

All denoising steps jointly operate on B,2,20,32,64. State and last-observed
increment targets use identical networks, initialization, training pairs and
budgets. Each gets a fresh deterministic CNN control with the same architecture.
No codec, teacher prediction, online correction network or old fitted weights
are loaded by the new predictors.
"""

import hashlib
import json
import os
import time

import numpy as np
import torch

from clean_reliability_common import ROOT, SOURCE, configure, digest, dump, manifest, load_field
from confirm_clean_reliability_cache import field_file
from joint20_physical_meanrisk import Joint20PhysicalMeanRisk
from retrain_clean_direct20 import train_steps


OUT = ROOT / "runs/clean_physical_direct20_20261109"
ARMS = ("state_diffusion", "delta_diffusion", "state_cnn", "delta_cnn")


def source_hashes():
    files = [SOURCE / "manifest.json", SOURCE / "normalization.pt"]
    for kind in ("ae", "vae"):
        files += [SOURCE / kind / "representation.pt", SOURCE / kind / "diffusion.pt",
                  ROOT / "runs/clean_direct20_cnn_controls_20261108" / kind / "model.pt",
                  ROOT / "runs/clean_direct20_delta_20261108" / kind / "diffusion/model.pt"]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def event(**values):
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time=time.time(), **values)) + "\n")


def new_model(arm):
    # Construct the same diffusion object even for a CNN, so every arm's core
    # and input/output layers receive exactly the same seed42 initialization.
    torch.manual_seed(42)
    model = Joint20PhysicalMeanRisk().cuda()
    return model if arm.endswith("diffusion") else model.net


def preflight():
    torch.manual_seed(42)
    h = torch.randn(2, 2, 20, 32, 64, device="cuda")
    y = torch.randn_like(h)
    hashes = {}
    for arm in ARMS:
        model = new_model(arm)
        net = model.net if arm.endswith("diffusion") else model
        out = net(torch.zeros_like(h), torch.full((2,), 99, device="cuda", dtype=torch.long), h)
        assert out.shape == h.shape
        assert torch.count_nonzero(out) == 0
        raw = b"".join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())
        hashes[arm] = hashlib.sha256(raw).hexdigest()
        if arm.endswith("diffusion"):
            loss = model.loss(h, y)
            assert torch.isfinite(loss)
            loss.backward()
            model.zero_grad(set_to_none=True)
            sample = model.rollout(h, 500000)
            assert sample.shape == y.shape and torch.isfinite(sample).all()
        del model
    assert len(set(hashes.values())) == 1
    # Inspect exact module boundary shapes to rule out hidden10+10 generation.
    dump(OUT / "architecture_audit.json", dict(passed=True, initialization_sha256=hashes,
         state_shape=[2, 2, 20, 32, 64], first_input_channels=80, output_channels=40,
         reverse_levels=4, temporal_subdivision=False,
         parameters=sum(p.numel() for p in new_model("state_cnn").parameters())))


def train():
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
            h, y = histories[ix], futures[ix]
            target = y - h[:, :, -1:] if arm.startswith("delta") else y
            if arm.endswith("diffusion"):
                return model.loss(h, target), {}
            pred = model(torch.zeros_like(h), torch.full((8,), 99, device="cuda", dtype=torch.long), h)
            # The CNN learns the same state or increment target, with the
            # original per-channel standardized physical MSE objective.
            return (pred - target).square().mean(), {}

        train_steps(model, objective, 6000, folder, arm, OUT)
        torch.save(dict(model=model.state_dict(), fit_split="train", seed=42, steps=6000,
                        manifest_sha256=digest(SOURCE / "manifest.json")), folder / "model.pt")
        del model
        torch.cuda.empty_cache()
    dump(OUT / "frozen.json", dict(artifacts={str(p.relative_to(OUT)): digest(p) for p in OUT.glob("*/model.pt")},
         fit_files=[r["file"] for r in manifest()["files"]["train"]], fit_frame_range=[0, 200],
         event="All four new predictors frozen before calibration/development/test field access."))


@torch.no_grad()
def issue():
    norm = torch.load(SOURCE / "normalization.pt", map_location="cuda", weights_only=False)
    models = {}
    for arm in ARMS:
        m = new_model(arm).eval().requires_grad_(False)
        m.load_state_dict(torch.load(OUT / arm / "model.pt", map_location="cuda", weights_only=False)["model"])
        models[arm] = m
    for split in ("calibration", "development", "test"):
        for start in range(0, 161 if split == "calibration" else 761, 20):
            observed = torch.load(field_file("history", split, start), map_location="cuda", weights_only=False)
            history = (observed - norm["mean"]) / norm["std"]
            issued = {}
            for arm, model in models.items():
                anchor = history[:, :, -1:] if arm.startswith("delta") else 0.
                if arm.endswith("diffusion"):
                    samples = torch.stack([(anchor + model.rollout(history, seed)) * norm["std"] + norm["mean"]
                                           for seed in range(500000, 500032)])
                    zero = (anchor + model.rollout(history, 500000, sign=0)) * norm["std"] + norm["mean"]
                    assert torch.isfinite(samples).all() and torch.isfinite(zero).all()
                    issued[arm] = dict(mean=samples.mean(0).cpu(), std32=samples.std(0).cpu(), zero=zero.cpu())
                    del samples
                else:
                    pred = model(torch.zeros_like(history), torch.full((len(history),), 99, device="cuda", dtype=torch.long), history)
                    pred = (anchor + pred) * norm["std"] + norm["mean"]
                    assert pred.shape == observed.shape and torch.isfinite(pred).all()
                    issued[arm] = dict(mean=pred.cpu())
            torch.save(issued, OUT / "issued" / f"{split}_{start}.pt")
            event(event="issued", split=split, start=start)
            # No target field is loaded in this sampling program. Metrics are a
            # separate offline replay, using only the saved forecasts above.
            dump(OUT / "status.json", dict(stage="sampling", split=split, start=start, pid=os.getpid()))
        print("PHYSICAL ISSUED", split, flush=True)
    dump(OUT / "sampling_complete.json", dict(windows=9 + 39 + 39, time=time.time()))


def summary(rows):
    result = []
    for split in ("development", "test"):
        for phase in ("main", "confirmation", "all"):
            for group in [*manifest()["groups"][split], "macro"]:
                for arm in (*ARMS, "state_zero", "delta_zero", "persistence"):
                    for channel in ("u", "v", "both"):
                        selected = [r for r in rows if r["split"] == split and r["arm"] == arm
                                    and (phase == "all" or r["period"] == phase)
                                    and (group == "macro" or r["group"] == group)
                                    and (channel == "both" or r["channel"] == channel)]
                        def mean(key):
                            groups = sorted({r["group"] for r in selected})
                            return float(np.mean([np.mean([r[key] for r in selected if r["group"] == g]) for g in groups]))
                        keys = [k for k in selected[0] if isinstance(selected[0][k], float)]
                        values = {k: mean(k) for k in keys}
                        values["difference_energy_ratio"] = values["difference_energy"] / values["truth_difference_energy"]
                        values["spatial_energy_ratio"] = values["spatial_energy"] / values["truth_spatial_energy"]
                        result.append(dict(split=split, period=phase, group=group, arm=arm, channel=channel, **values))
    return result


@torch.no_grad()
def evaluate():
    # GPU is used for reductions/quantiles only; there is no forecast recompute.
    norm = torch.load(SOURCE / "normalization.pt", map_location="cuda", weights_only=False)
    ratios = {arm: [[], []] for arm in ("state_diffusion", "delta_diffusion")}
    for start in range(0, 161, 20):
        pred = torch.load(OUT / "issued" / f"calibration_{start}.pt", map_location="cuda", weights_only=False)
        y = torch.load(field_file("target", "calibration", start), map_location="cuda", weights_only=False)
        for arm in ratios:
            ratio = (y - pred[arm]["mean"]).abs() / pred[arm]["std32"].clamp_min(norm["std"] * 1e-4)
            for c in range(2):
                ratios[arm][c].append(ratio[:, c].flatten().cpu())
    factors = {arm: [float(torch.quantile(torch.cat(ratios[arm][c]).cuda(), .9)) for c in range(2)] for arm in ratios}
    dump(OUT / "calibration_locked.json", factors)
    rows = []
    for split in ("development", "test"):
        scales = {}
        for start in range(0, 761, 20):
            history = torch.load(field_file("history", split, start), map_location="cuda", weights_only=False)
            issued = torch.load(OUT / "issued" / f"{split}_{start}.pt", map_location="cuda", weights_only=False)
            event(event="replay_issued", split=split, start=start)
            target = torch.load(field_file("target", split, start), map_location="cuda", weights_only=False)
            event(event="replay_target_revealed", split=split, start=start)
            points = {arm: issued[arm]["mean"] for arm in ARMS}
            points.update(state_zero=issued["state_diffusion"]["zero"], delta_zero=issued["delta_diffusion"]["zero"],
                          persistence=history[:, :, -1:].expand_as(target))
            truth_space = (target - target.mean((-2, -1), keepdim=True)).square().mean((2, 3, 4))
            def difference_energy(x):
                return .5 * ((x[..., 1:] - x[..., :-1]).square().mean((2, 3, 4))
                             + (x[..., 1:, :] - x[..., :-1, :]).square().mean((2, 3, 4)))
            truth_gradient = difference_energy(target)
            for arm, mu in points.items():
                error = (mu - target).abs()
                mse = error.square().mean((2, 3, 4))
                spatial = (mu - mu.mean((-2, -1), keepdim=True)).square().mean((2, 3, 4))
                gradient = difference_energy(mu)
                if arm in factors:
                    fixed = torch.tensor(factors[arm], device="cuda").reshape(1, 2, 1, 1, 1)
                    if arm not in scales:
                        scales[arm] = fixed.expand(len(target), 2, 1, 1, 1).clone()
                    sd = issued[arm]["std32"].clamp_min(norm["std"] * 1e-4)
                    radius = scales[arm] * sd
                    coverage = (error <= radius).float().mean((2, 3, 4))
                    fixed_coverage = (error <= fixed * sd).float().mean((2, 3, 4))
                    score = (2 * radius + 20 * (error - radius).clamp_min(0)).mean((2, 3, 4))
                    q = torch.quantile((error / sd).flatten(2), .9, dim=2).reshape(len(target), 2, 1, 1, 1)
                    scales[arm] = (scales[arm] * q.clamp(1e-6, 1e3)).sqrt()
                for i, record in enumerate(manifest()["files"][split]):
                    for c, channel in enumerate(("u", "v")):
                        row = dict(split=split, file=record["file"], group=record["group"], start=start,
                                   period="confirmation" if start >= 580 else "main", channel=channel, arm=arm,
                                   mse=float(mse[i, c]), standardized_mse=float(mse[i, c] / norm["std"].flatten()[c].square()),
                                   spatial_energy=float(spatial[i, c]), truth_spatial_energy=float(truth_space[i, c]),
                                   difference_energy=float(gradient[i, c]), truth_difference_energy=float(truth_gradient[i, c]))
                        if arm in factors:
                            row.update(online_coverage=float(coverage[i, c]), fixed_coverage=float(fixed_coverage[i, c]),
                                       online_interval_score_standardized=float(score[i, c] / norm["std"].flatten()[c]))
                        rows.append(row)
    dump(OUT / "per_case.json", rows)
    dump(OUT / "summary.json", summary(rows))


def main():
    configure()
    assert os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "0"
    OUT.mkdir(exist_ok=False)
    (OUT / "issued").mkdir()
    before = source_hashes()
    code_files = ("experiment_clean_physical_direct20.py", "joint20_physical_meanrisk.py", "joint20_data_head_diffusion.py",
                  "joint20_state_grid_diffusion.py", "fullblock_diffusion_parameterization.py", "retrain_clean_direct20.py")
    dump(OUT / "protocol.json", dict(seed=42, arms=ARMS, steps=6000, batch=8, fit_pairs=1551,
         task="20 observed physical frames ->20 jointly generated physical future frames; no10+10 or5frame recursion.",
         fitting="Only47train trajectories first200frames; reuse audited train-only mean/std; no codec, pretrained predictor or teacher labels.",
         hypothesis="Remove learned state compression and compare physical last-state increments against physical state prediction.",
         coordinates="State:y_normalized. Delta:y_normalized-last_observed_normalized, with the same original field scale. No increment-RMS fitting.",
         objectives="Diffusion:unchanged native clean-state error/(1-alpha_bar),4noise levels. CNN:standardized physical MSE on same target coordinates.",
         initialization="All4 identical net weights after seed42 fresh initialization; no loading old fitted model weights.",
         selection="Fixed6000 endpoints; no development/test selection. All4 frozen before held-out field access.",
         sampling="32paths500000:500032,4DDIM steps, entire20frame tensor; additionally one zero-initial-noise path.",
         intervals="Per-channel q90 calibrated only on2calibration Re groups first200; raw sigma floor trainstd*1e-4. Online sqrt(scale*q90) update after entire current20truth, affecting next window.",
         compute="Physical network runs at32x64 vs codec latents16x32; same width, different input/output channels and larger spatial compute. Not an equal-FLOP representation comparison.",
         scope="Historically reused groups, exploratory. New train excludes all protected dev/test/calibration groups. No cross-validation, distillation or multiple training seeds.",
         source_sha256=before, code_sha256={f: digest(ROOT / f) for f in code_files}))
    preflight()
    train()
    issue()
    evaluate()
    frozen = json.loads((OUT / "frozen.json").read_text())["artifacts"]
    assert all(digest(OUT / name) == sha for name, sha in frozen.items())
    assert source_hashes() == before
    dump(OUT / "audit.json", dict(passed=True, source_unchanged=True, new_weights_unchanged_after_freeze=True))
    dump(OUT / "status.json", dict(stage="complete"))
    print("PHYSICAL COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    main()
