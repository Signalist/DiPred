"""Train direct-20 AE/VAE/RAE diffusions with isolated fit/calibration data.

All fitted artifacts belong to this run. Legacy manifests supply only protected
group IDs, never fields, normalization, weights, latent statistics, or labels.
Development and test groups remain excluded from every gradient update.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import torch

from joint20_data_head_diffusion import Joint20DataHeadDiffusion
from local_grid_representation import LocalGridRepresentation


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "OLD/REALPDE/realpdebench-competition/data/train_real"
KINDS = ("ae", "vae", "rae")
FRAME_END = 200
EVAL_STARTS = (0, 40, 80, 120, 160)


def dump(path, value):
    # Replace complete files atomically so interrupted runs leave readable status.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def prepare(out, args):
    old = json.loads((ROOT / "runs/shortblock_relative_objective/manifest.json").read_text())
    protected = set(old["groups"]["validation"] + old["groups"]["test"])
    inventory = []
    duplicates = []
    seen = {}
    for path in sorted(DATA.glob("*.h5")):
        with h5py.File(path, "r") as source:
            assert min(len(source[c]) for c in ("u", "v")) >= FRAME_END
            # A data-integrity read only: these bytes never enter fitting.
            values = np.stack([source[c][:FRAME_END, ::2, ::2] for c in ("u", "v")]).astype(np.float32)
            assert np.isfinite(values).all()
            sha = hashlib.sha256(values.tobytes()).hexdigest()
            if sha in seen:
                duplicates.append({"excluded": path.name, "retained": seen[sha], "sha256_first200": sha})
                continue
            seen[sha] = path.name
            inventory.append(dict(file=path.name, group=path.stem.split("_")[0],
                                  re=float(source["re"][()]), aoa=float(source["aoa"][()]),
                                  frames=len(source["u"]), sha256_first200=sha))
    rng = np.random.default_rng(42)
    candidate_groups = sorted({r["group"] for r in inventory} - protected, key=int)
    candidate_groups = rng.permutation(candidate_groups).tolist()
    holdout_groups = rng.permutation(sorted(protected, key=int)).tolist()
    assert len(candidate_groups) == 12 and len(holdout_groups) == 6
    groups = {"train": candidate_groups[2:], "calibration": candidate_groups[:2],
              "development": holdout_groups[:3], "test": holdout_groups[3:]}
    files = {split: [r for r in inventory if r["group"] in gs] for split, gs in groups.items()}
    for split in groups:
        assert not (set(groups[split]) & set().union(*(set(gs) for k, gs in groups.items() if k != split)))
    assert not set(groups["train"]) & protected
    assert not set(groups["calibration"]) & protected
    manifest = dict(groups=groups, files=files, duplicates=duplicates,
                    protected_legacy_development_test_groups=sorted(protected, key=int),
                    split_seed=42, fit_frames=[0, FRAME_END], history=20, horizon=20,
                    train_window_starts=list(range(0, 161, 5)), evaluation_window_starts=EVAL_STARTS,
                    statistics="Physical and latent normalization are fitted on new training fields only.",
                    scope="Group-disjoint refit. All data have a research history; reshuffling does not make a fresh external test.")
    dump(out / "manifest.json", manifest)
    dump(out / "protocol.json", dict(
        seed=42, kinds=KINDS, initialization="All new random weights; no legacy checkpoint or normalization loads.",
        representation_steps=args.representation_steps, masked_pretraining_steps=args.pretrain_steps,
        diffusion_steps=args.diffusion_steps, batch=8, vae_kl=1e-4,
        representation="Same local-grid architecture; RAE encoder mask-pretrained only on new train, then frozen; decoder reinitialized.",
        diffusion="Joint20DataHeadDiffusion, 4 reverse levels, complete 4x20x16x32 latent state at every level, no temporal split.",
        objective="Native clean-state squared error/(1-alpha_bar); fixed endpoint; no point-risk continuation in this clean baseline.",
        selection="Fixed 6000-step defaults, no development-driven checkpoint selection or hyperparameter search.",
        development="Diagnostic only, never used for fitting, normalization, pretraining, scale calibration, or checkpoint selection.",
        calibration="Dedicated unseen Re groups; only interval scales fitted after all predictors frozen.",
        test="Metrics computed only after all 3 codec/predictor pairs and calibration factors are frozen.",
        paths=args.samples, sample_seeds=list(range(500000, 500000 + args.samples)),
        limits=["Single training seed; RAE has extra masked pretraining budget.",
                "Different Reynolds groups imply distribution shift; no conformal coverage guarantee.",
                "All split data were used in previous research; clean refit, not a new external confirmation.",
                "Default budgets fixed before reading current run development metrics; changed budgets are recorded."],
        source_sha256={f: digest(ROOT / f) for f in ("retrain_clean_direct20.py", "joint20_data_head_diffusion.py", "joint20_state_grid_diffusion.py", "local_grid_representation.py")},
    ))
    print("SPLIT", {s: {"groups": groups[s], "files": len(files[s])} for s in files}, flush=True)
    return manifest


def read_fields(manifest, split, out, purpose):
    if purpose in ("fit", "normalization"):
        assert split == "train", "Fitting may only read the new training split."
    with (out / "data_access.jsonl").open("a") as audit:
        audit.write(json.dumps(dict(time=time.time(), split=split, purpose=purpose,
                                    files=[r["file"] for r in manifest["files"][split]],
                                    frame_range=[0, FRAME_END])) + "\n")
    rows = []
    for record in manifest["files"][split]:
        with h5py.File(DATA / record["file"], "r") as source:
            raw = np.stack([source[c][:FRAME_END, ::2, ::2] for c in ("u", "v")]).astype(np.float32)
        assert hashlib.sha256(raw.tobytes()).hexdigest() == record["sha256_first200"]
        rows.append(torch.from_numpy(raw))
    return torch.stack(rows).cuda()


def train_steps(model, objective, steps, folder, stage, out):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-5)
    logs = []
    begin = time.time()
    for step in range(1, steps + 1):
        learning_rate = 1e-6 + (3e-4 - 1e-6) * .5 * (1 + math.cos(math.pi * (step - 1) / steps))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        loss, details = objective()
        assert torch.isfinite(loss)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(params, 1.)
        assert torch.isfinite(gradient)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == steps:
            row = dict(step=step, loss=float(loss), gradient_norm=float(gradient),
                       elapsed=time.time() - begin, **details)
            logs.append(row)
            dump(folder / "training.json", logs)
            dump(out / "status.json", dict(stage=stage, step=step, total=steps, pid=os.getpid()))
            print(stage, row, flush=True)


@torch.no_grad()
def encode(codec, blocks):
    return torch.cat([codec.encode(blocks[i:i + 4])[0] for i in range(0, len(blocks), 4)])


def train(manifest, out, args):
    raw = read_fields(manifest, "train", out, "fit")
    # Training statistics include training targets; no held-out values are used.
    mean = raw.mean((0, 2, 3, 4), keepdim=True)
    std = raw.std((0, 2, 3, 4), keepdim=True).clamp_min(1e-6)
    fields = (raw - mean) / std
    torch.save(dict(mean=mean.cpu(), std=std.cpu(), fit_files=[r["file"] for r in manifest["files"]["train"]],
                    manifest_sha256=digest(out / "manifest.json")), out / "normalization.pt")
    blocks = torch.cat([fields[:, :, s:s + 20] for s in range(0, FRAME_END, 20)])
    histories = torch.cat([fields[:, :, s:s + 20] for s in range(0, 161, 5)])
    futures = torch.cat([fields[:, :, s + 20:s + 40] for s in range(0, 161, 5)])
    del raw, fields
    pretrained = None
    for kind in KINDS:
        folder = out / kind
        folder.mkdir()
        if kind == "rae":
            masked_folder = folder / "masked"
            masked_folder.mkdir()
            torch.manual_seed(42)
            masked = LocalGridRepresentation("ae").cuda()

            def masked_loss():
                x = blocks[torch.randint(len(blocks), (8,), device="cuda")]
                return masked.masked_loss(x), {}

            train_steps(masked, masked_loss, args.pretrain_steps, masked_folder, "rae_mask_pretrain", out)
            pretrained = copy.deepcopy(masked.encoder.state_dict())
            torch.save(dict(encoder=pretrained, fit_split="train", manifest_sha256=digest(out / "manifest.json")), masked_folder / "encoder.pt")
            del masked
        torch.manual_seed(42)
        codec = LocalGridRepresentation(kind, pretrained).cuda()
        frozen_encoder = copy.deepcopy(codec.encoder.state_dict()) if kind == "rae" else None

        def representation_loss():
            x = blocks[torch.randint(len(blocks), (8,), device="cuda")]
            z, kl = codec.encode(x, sample=kind == "vae")
            mse = (codec.decode(z) - x).square().mean()
            return mse + (1e-4 * kl if kind == "vae" else 0), {"reconstruction_mse": float(mse), "kl": float(kl)}

        rec_folder = folder / "reconstruction"
        rec_folder.mkdir()
        train_steps(codec, representation_loss, args.representation_steps, rec_folder, kind + "_reconstruction", out)
        if frozen_encoder is not None:
            assert all(torch.equal(t, codec.encoder.state_dict()[k]) for k, t in frozen_encoder.items())
        codec.zero_grad(set_to_none=True)
        codec.eval().requires_grad_(False)
        with torch.no_grad():
            codes = encode(codec, blocks)
            center = codes.mean((0, 2, 3, 4), keepdim=True)
            scale = codes.std((0, 2, 3, 4), keepdim=True).clamp_min(1e-4)
            zh = (encode(codec, histories) - center) / scale
            zy = (encode(codec, futures) - center) / scale
        provenance = dict(kind=kind, manifest_sha256=digest(out / "manifest.json"), fit_split="train")
        torch.save(dict(model=codec.state_dict(), latent_center=center, latent_scale=scale, **provenance), folder / "representation.pt")
        torch.manual_seed(42)
        model = Joint20DataHeadDiffusion().cuda()

        def diffusion_loss():
            ix = torch.randint(len(zh), (8,), device="cuda")
            return model.loss(zh[ix], zy[ix]), {}

        diff_folder = folder / "diffusion"
        diff_folder.mkdir()
        train_steps(model, diffusion_loss, args.diffusion_steps, diff_folder, kind + "_diffusion", out)
        assert all(p.grad is None for p in codec.parameters())
        torch.save(dict(model=model.state_dict(), **provenance), folder / "diffusion.pt")
        del model, codec, zh, zy, codes
        torch.cuda.empty_cache()
    artifacts = {str(path.relative_to(out)): digest(path) for path in out.glob("*/*.pt")}
    artifacts["normalization.pt"] = digest(out / "normalization.pt")
    dump(out / "frozen.json", dict(artifacts=artifacts, manifest_sha256=digest(out / "manifest.json"),
                                   stage="All fitted weights frozen before calibration/development/test metric access."))


@torch.no_grad()
def predict(model, codec, ck, history, mean, std, samples):
    z = (codec.encode((history - mean) / std)[0] - ck["latent_center"]) / ck["latent_scale"]
    return torch.stack([
        codec.decode(model.rollout(z, seed) * ck["latent_scale"] + ck["latent_center"]) * std + mean
        for seed in range(500000, 500000 + samples)
    ])


def interval_statistics(samples, truth):
    center = samples.mean(0)
    lower, upper = torch.quantile(samples, torch.tensor([.05, .95], device="cuda"), dim=0)
    low = (center - lower).clamp_min(1e-9)
    high = (upper - center).clamp_min(1e-9)
    ratio = (truth - center).abs() / torch.where(truth >= center, high, low)
    return center, low, high, ratio


def macro(rows, key):
    groups = sorted({r["group"] for r in rows})
    return float(np.mean([np.mean([r[key] for r in rows if r["group"] == group]) for group in groups]))


def evaluate(manifest, out, args):
    frozen = json.loads((out / "frozen.json").read_text())
    assert digest(out / "manifest.json") == frozen["manifest_sha256"]
    assert all(digest(out / path) == sha for path, sha in frozen["artifacts"].items())
    norm = torch.load(out / "normalization.pt", map_location="cuda", weights_only=False)
    mean, std = norm["mean"], norm["std"]
    rows, factors = [], {}
    # Evaluate calibration separately; test fields are not loaded before scale lock.
    for split in ("calibration", "development", "test"):
        if split == "test":
            dump(out / "calibration_locked.json", factors)
        raw = read_fields(manifest, split, out, "interval_calibration" if split == "calibration" else "evaluation")
        for kind in KINDS:
            ck = torch.load(out / kind / "representation.pt", map_location="cuda", weights_only=False)
            state = ck["model"]
            pretrained = {k[len("encoder."):]: t for k, t in state.items() if k.startswith("encoder.")} if kind == "rae" else None
            codec = LocalGridRepresentation(kind, pretrained).cuda().eval()
            codec.load_state_dict(state)
            model = Joint20DataHeadDiffusion().cuda().eval()
            model.load_state_dict(torch.load(out / kind / "diffusion.pt", map_location="cuda", weights_only=False)["model"])
            needed = []
            for start in EVAL_STARTS:
                with torch.no_grad():
                    samples = predict(model, codec, ck, raw[:, :, start:start + 20], mean, std, args.samples)
                    truth = raw[:, :, start + 20:start + 40]
                    center, low, high, ratio = interval_statistics(samples, truth)
                    if split == "calibration":
                        flat = ratio.flatten(2)
                        needed.append(torch.stack([torch.quantile(flat, .9, dim=2), torch.quantile(flat, .95, dim=2), flat.amax(dim=2)]))
                        continue
                    reconstruction_pred = codec.decode(codec.encode((truth - mean) / std)[0]) * std + mean
                    rec_error = (reconstruction_pred - truth).square().mean((2, 3, 4))
                    mse = (center - truth).square().mean((2, 3, 4))
                    normalized = ((center - truth) / std).square().mean((2, 3, 4))
                    ordered = samples.sort(0).values
                    weights = (2 * torch.arange(1, args.samples + 1, device="cuda") - args.samples - 1).reshape(args.samples, 1, 1, 1, 1, 1)
                    crps = ((samples - truth[None]).abs().mean(0) - (ordered * weights).sum(0) / (args.samples * (args.samples - 1))).mean((2, 3, 4))
                    variants = {"raw": [1., 1.], **factors[kind]}
                    for variant, fac in variants.items():
                        scale = torch.tensor(fac, device="cuda")[None, :, None, None, None]
                        coverage = ((truth >= center - scale * low) & (truth <= center + scale * high)).float().mean((2, 3, 4))
                        width = (scale * (low + high)).mean((2, 3, 4))
                        for i, record in enumerate(manifest["files"][split]):
                            for c, channel in enumerate(("u", "v")):
                                rows.append(dict(method=kind, split=split, file=record["file"], group=record["group"], start=start,
                                                 channel=channel, variant=variant, mse=float(mse[i, c]), standardized_mse=float(normalized[i, c]),
                                                 raw_fair_crps=float(crps[i, c]), reconstruction_mse=float(rec_error[i, c]), coverage=float(coverage[i, c]),
                                                 width=float(width[i, c]), at_least_95=float(coverage[i, c] >= .95), all=float(coverage[i, c] == 1)))
            if split == "calibration":
                # Independent calibration trajectories supply one worst-window score each.
                scores = torch.stack(needed).amax(dim=0)
                n = scores.shape[1]
                rank = math.ceil((n + 1) * .9)
                # With fewer than 9 trajectories, the conservative finite rank is infinite.
                # We explicitly choose the empirical maximum instead; no finite-sample guarantee.
                rank = min(n, rank)
                selected = scores.sort(dim=1).values[:, rank - 1, :]
                assert torch.isfinite(selected).all() and (selected > 0).all()
                factors[kind] = {name: selected[i].cpu().tolist() for i, name in enumerate(("window90", "window95", "window100"))}
            del codec, model
            torch.cuda.empty_cache()
            print("EVALUATION", split, kind, flush=True)
        del raw
    summary = {}
    for split in ("development", "test"):
        summary[split] = {}
        for kind in KINDS:
            summary[split][kind] = {}
            for channel in ("u", "v"):
                summary[split][kind][channel] = {}
                for variant in ("raw", "window90", "window95", "window100"):
                    selected = [r for r in rows if r["split"] == split and r["method"] == kind and r["channel"] == channel and r["variant"] == variant]
                    summary[split][kind][channel][variant] = {k: macro(selected, k) for k in ("mse", "standardized_mse", "raw_fair_crps", "reconstruction_mse", "coverage", "width", "at_least_95", "all")}
    dump(out / "per_case.json", rows)
    dump(out / "results.json", dict(summary=summary, calibration_factors=factors,
                                    interpretation="Raw CRPS refers to unscaled ensemble only; scale variants share identical point forecast. Per-channel coverage, not joint u/v guarantee. All splits are Reynolds-disjoint for this refit; historical reuse remains."))
    assert all(digest(out / path) == sha for path, sha in frozen["artifacts"].items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/clean_direct20_20261107")
    parser.add_argument("--representation-steps", type=int, default=6000)
    parser.add_argument("--pretrain-steps", type=int, default=6000)
    parser.add_argument("--diffusion-steps", type=int, default=6000)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)
    try:
        assert os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "0"
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        manifest = prepare(out, args)
        train(manifest, out, args)
        evaluate(manifest, out, args)
        dump(out / "status.json", dict(stage="complete", pid=os.getpid()))
    except Exception as error:
        dump(out / "status.json", dict(stage="failed", error=repr(error), pid=os.getpid()))
        raise


if __name__ == "__main__":
    main()
