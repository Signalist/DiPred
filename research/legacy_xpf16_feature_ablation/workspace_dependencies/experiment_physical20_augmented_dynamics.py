"""Isolated train-only conditioning ablation for full20 physical diffusion.

All fixed6000 endpoints are frozen before any held-out history is read.
No old fitted predictor is loaded into training. Original results remain intact.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from clean_reliability_common import ROOT,SOURCE,configure,digest,dump,load_field,manifest
from confirm_clean_reliability_cache import field_file
from experiment_clean_physical_direct20 import new_model as original_model
from experiment_physical20_noise_refinement import gradient_energy,spectral,summarize
from physical20_augmented_dynamics import AugmentedDynamics20
from retrain_clean_direct20 import train_steps

OUT = ROOT / "runs/physical20_augmented_dynamics_20261109"
BASE = ROOT / "runs/clean_physical_direct20_20261109"
ARMS = ("augmented_diffusion",)
TRAIN_ARMS = (*ARMS, "augmented_cnn")


def event(**values):
    with (OUT/"events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time=time.time(),**values))+"\n")


def new_model(arm, scale=None):
    if scale is None:
        scale = torch.load(OUT / "dynamics_scale.pt", map_location="cuda", weights_only=False)["scale"]
    torch.manual_seed(42)
    return AugmentedDynamics20(scale).cuda()


def preflight(scale):
    reference = original_model("state_diffusion")
    model = new_model("augmented_diffusion", scale)
    for name, value in reference.net.state_dict().items():
        candidate = model.net.state_dict()[name]
        if name in ("first.a.weight", "first.skip.weight"):
            assert torch.equal(candidate[:, :80], value)
            assert torch.count_nonzero(candidate[:, 80:]) == 0
        else:
            assert torch.equal(candidate, value)
    # Check function preservation with a deliberately nonzero output head so
    # the test cannot pass merely because both original heads output zeros.
    generator = torch.Generator(device="cuda").manual_seed(71)
    with torch.no_grad():
        reference.net.out.weight.normal_(std=.02, generator=generator)
        reference.net.out.bias.normal_(std=.02, generator=generator)
        model.net.out.load_state_dict(reference.net.out.state_dict())
        h = torch.randn(2,2,20,32,64,device="cuda",generator=generator)
        x = torch.randn(h.shape,device="cuda",generator=generator)
        t = torch.tensor([0,99],device="cuda")
        difference = float((model.net(x,t,h)-reference.net(x,t,h)).abs().max())
        assert difference < 1e-5
    del reference, model
    checks = []
    for arm in TRAIN_ARMS:
        model = new_model(arm, scale)
        calls = []
        hook = model.net.register_forward_pre_hook(lambda module,args:calls.append(list(args[0].shape)))
        with torch.no_grad():
            pred = model.rollout(h,500000,sign=0)
        hook.remove()
        assert calls == [[2,2,20,32,64]]*4
        assert torch.isfinite(pred).all()
        loss = model.loss(h,torch.randn_like(h))
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        checks.append(dict(arm=arm,network_shapes=calls,parameters=sum(p.numel() for p in model.parameters())))
    dump(OUT / "preflight.json",dict(passed=True,original_weights_preserved=True,
         nonzero_head_initial_function_max_error=difference,checks=checks))


def train():
    event(event="read_train_only", files=[r["file"] for r in manifest()["files"]["train"]], frames=[0, 200])
    raw = torch.cat([load_field("history", "train", s) for s in range(0, 161, 20)]
                    + [load_field("target", "train", 160)], dim=2)
    assert raw.shape == (47, 2, 200, 32, 64)
    for i, record in enumerate(manifest()["files"]["train"]):
        assert hashlib.sha256(raw[i].cpu().numpy().tobytes()).hexdigest() == record["sha256_first200"]
    norm = torch.load(SOURCE / "normalization.pt", map_location="cuda", weights_only=False)
    fields = (raw - norm["mean"]) / norm["std"]
    fields_scale = fields.diff(dim=2).square().mean((0,2,3,4),keepdim=True).sqrt().clamp_min(1e-4)
    histories = torch.cat([fields[:, :, s:s + 20] for s in range(0, 161, 5)])
    futures = torch.cat([fields[:, :, s + 20:s + 40] for s in range(0, 161, 5)])
    assert len(histories) == 1551
    del raw, fields
    scale = fields_scale
    torch.save(dict(scale=scale.cpu(), fit_split="train", frames=[0,200]), OUT / "dynamics_scale.pt")
    dump(OUT / "dynamics_scale.json", dict(scale=scale.flatten().cpu().tolist(), fit_split="train", frames=[0,200]))
    preflight(scale)
    for arm in TRAIN_ARMS:
        folder = OUT / arm
        folder.mkdir()
        model = new_model(arm, scale)

        def objective():
            ix = torch.randint(len(histories), (8,), device="cuda")
            h, y = histories[ix], futures[ix]
            if arm == "augmented_cnn":
                pred = model.net(torch.zeros_like(h), torch.full((8,),99,device=h.device,dtype=torch.long), h)
                return (pred-y).square().mean(), {}
            return model.loss(h,y), {}

        train_steps(model, objective, 6000, folder, arm, OUT)
        torch.save(dict(model=model.state_dict(), arm=arm, seed=42, steps=6000,
                        fit_split="train", manifest_sha256=digest(SOURCE / "manifest.json")), folder / "model.pt")
        event(event="endpoint_frozen", arm=arm, sha256=digest(folder / "model.pt"))
        del model
        torch.cuda.empty_cache()
    dump(OUT / "frozen.json", dict(time=time.time(), artifacts={arm: digest(OUT / arm / "model.pt") for arm in TRAIN_ARMS}))


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
    model = new_model("augmented_cnn").eval().requires_grad_(False)
    model.load_state_dict(torch.load(OUT / "augmented_cnn/model.pt",map_location="cuda",weights_only=False)["model"])
    folder = OUT / "issued/augmented_cnn"
    folder.mkdir()
    for split in ("development","test"):
        for start in range(0,761,20):
            observed=torch.load(field_file("history",split,start),map_location="cuda",weights_only=False)
            history=(observed-norm["mean"])/norm["std"]
            pred=model.net(torch.zeros_like(history),torch.full((len(history),),99,device="cuda",dtype=torch.long),history)
            pred=pred*norm["std"]+norm["mean"]
            assert pred.shape==observed.shape and torch.isfinite(pred).all()
            torch.save(dict(zero=pred.cpu()),folder/f"{split}_{start}.pt")
            event(event="issued",arm="augmented_cnn",split=split,start=start)
    del model
    dump(OUT / "sampling_complete.json", dict(time=time.time(), windows_per_arm=87, paths=32))


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
    lead_rows = []
    for split in ("development", "test"):
        scales = {}
        for start in range(0, 761, 20):
            event(event="replay_predictions", split=split, start=start)
            old = torch.load(BASE / "issued" / f"{split}_{start}.pt", map_location="cuda", weights_only=False)
            new = {arm: torch.load(OUT / "issued" / arm / f"{split}_{start}.pt", map_location="cuda", weights_only=False) for arm in ARMS}
            cnn = torch.load(OUT / "issued/augmented_cnn" / f"{split}_{start}.pt", map_location="cuda", weights_only=False)["zero"]
            target = torch.load(field_file("target", split, start), map_location="cuda", weights_only=False)
            event(event="reveal_target", split=split, start=start)
            spectrum_y = spectral(target)
            truth_gradient = gradient_energy(target)
            points = dict(baseline_zero=old["state_diffusion"]["zero"], baseline_mean32=old["state_diffusion"]["mean"],
                          state_cnn=old["state_cnn"]["mean"], delta_cnn=old["delta_cnn"]["mean"])
            points["augmented_cnn"] = cnn
            for arm, issued in new.items():
                points[arm + "_zero"] = issued["zero"]
                points[arm + "_mean32"] = issued["mean"]
            for name, point in points.items():
                squared = (point-target).square()
                error = squared.mean((2,3,4))
                by_lead = squared.mean((3,4))
                gradient = gradient_energy(point)
                spectrum = spectral(point)
                values = dict(mse=error, standardized_mse=error / norm["std"].flatten()[None].square(),
                              difference_energy=gradient, truth_difference_energy=truth_gradient,
                              high_power=spectrum.abs().square().mean((2, 3)),
                              truth_high_power=spectrum_y.abs().square().mean((2, 3)),
                              high_error=(spectrum - spectrum_y).abs().square().mean((2, 3)))
                if name.endswith("_mean32") and name[:-len("_mean32")] in ARMS:
                    arm = name[:-len("_mean32")]
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
                        for lead in range(20):
                            lead_rows.append(dict(split=split,period="confirmation" if start>=580 else "main",start=start,
                                file=record["file"],group=record["group"],channel=channel,arm=name,lead=lead+1,
                                mse=float(by_lead[i,c,lead])))
                        rows.append(dict(split=split, period="confirmation" if start >= 580 else "main", start=start,
                                         file=record["file"], group=record["group"], channel=channel, arm=name,
                                         **{k: float(v[i, c]) for k, v in values.items()}))
    dump(OUT / "per_lead.json", lead_rows)
    dump(OUT / "per_case.json", rows)
    dump(OUT / "summary.json", summarize(rows))
    print("EVALUATED", len(rows), "rows", flush=True)



def main():
    configure()
    assert os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")=="0"
    OUT.mkdir(exist_ok=False)
    dump(OUT/"protocol.json",dict(
        time=time.time(),arms=TRAIN_ARMS,seed=42,updates=6000,batch=8,fit_pairs=1551,fit_trajectories=47,fit_frames=[0,200],
        hypothesis="Retain raw history and append normalized differences with zero-added input weights, preserving the original unfitted function.",
        conditioning="Original40history+40noisyfuture channels, plus40 observed-difference channels (first frame zero), train-only RMS scale.",
        target="Unchanged full physical future state; all20 frames jointly at each of4 cosine DDIM steps.",
        controls="Matched augmented-condition one-call CNN; frozen original full-history CNN and Diffusion. Historical replacement-condition results kept separately.",
        initialization="Seed42 unfitted original weights retained; extra first-block columns zero.1616872params vs1597672 original (+1.2%). No fitted predictor or representation module.",
        fitting="Only47train trajectories first200frames; scale and physical mean/std train only. No development/test normalization or fitting.",
        selection="Fixed6000 endpoints; both frozen before any held-out history read. Report all arms; no dev selection, crossfolds, distillation or extra training seeds.",
        sampling="Diffusion32paths500000:500032 plus zero initial noise; all predictions saved before target access.",
        metrics="Per-window and per-lead raw/standardized MSE, frequency power and phase-sensitive error, marginal90% intervals.",
        scope="Historically reused data; exploratory. Reparameterizing history adds no information or independent novelty claim.",
        manifest_sha256=digest(SOURCE/"manifest.json"),normalization_sha256=digest(SOURCE/"normalization.pt"),
        code_sha256={name:digest(ROOT/name) for name in [Path(__file__).name,"physical20_augmented_dynamics.py"]}))
    train()
    issue()
    evaluate()
    frozen=json.loads((OUT/"frozen.json").read_text())
    assert all(digest(OUT/arm/"model.pt")==sha for arm,sha in frozen["artifacts"].items())
    dump(OUT/"complete.json",dict(time=time.time(),passed=True))


if __name__=="__main__":
    main()
