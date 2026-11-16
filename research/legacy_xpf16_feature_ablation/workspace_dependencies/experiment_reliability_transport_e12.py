"""E1/E2: frozen Direct20 predictions, risk heads, and causal error transport.

Risk fitting uses only original training trajectories at times200:400, outside
the predictor's fitted first200 frames. Calibration trajectories remain separate.
"""

import json
import math
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from clean_reliability_common import DATA, ROOT, SOURCE, configure, digest, dump, manifest
from confirm_clean_reliability_cache import field_file
from experiment_physical20_augmented_dynamics import OUT as PREDICTOR, new_model
from reliability_risk_transport import (CausalRiskMemory, SpatiotemporalRiskNet,
    assemble_features, base_features, preflight)


OUT = ROOT / "runs/reliability_transport_e12_20261109"
KINDS = ("diffusion", "cnn")
MODES = dict(diffusion=("base", "spread", "static", "advected", "reverse"),
             cnn=("base", "static", "advected", "reverse"))
FIT_STARTS = tuple(range(200, 361, 20))
FIT_RECORDS = [r for r in manifest()["files"]["train"] if r["frames"] >= 400]


def event(**values):
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time=time.time(), **values)) + "\n")


def normalization():
    return torch.load(SOURCE / "normalization.pt", map_location="cuda", weights_only=False)


def geometry(records):
    values = []
    for record in records:
        with h5py.File(DATA / record["file"], "r") as source:
            x, y = source["x"][::2, ::2], source["y"][::2, ::2]
            dt = np.diff(source["t"][:21]).astype(np.float64)
            dx, dy = float(np.median(np.diff(x, axis=1))), float(np.median(np.diff(y, axis=0)))
            assert np.allclose(np.diff(x, axis=1), dx, rtol=1e-4, atol=1e-9)
            assert np.allclose(np.diff(y, axis=0), dy, rtol=1e-4, atol=1e-9)
            assert np.allclose(dt, np.median(dt), rtol=1e-3, atol=1e-7)
            values.append([dx, dy, float(np.median(dt))])
    return torch.tensor(values, device="cuda", dtype=torch.float32)


def read_fit(start):
    rows = []
    for record in FIT_RECORDS:
        with h5py.File(DATA / record["file"], "r") as source:
            value = np.stack([source[c][start:start + 20, ::2, ::2] for c in ("u", "v")]).astype(np.float32)
        assert value.shape == (2, 20, 32, 64) and np.isfinite(value).all()
        rows.append(torch.from_numpy(value))
    return torch.stack(rows).cuda()


def history(split, start):
    if split == "fit":
        return read_fit(start)
    return torch.load(field_file("history", split, start), map_location="cuda", weights_only=False)


def target(split, start):
    if split == "fit":
        return read_fit(start + 20)
    return torch.load(field_file("target", split, start), map_location="cuda", weights_only=False)


def issued(kind, split, start):
    if split == "fit":
        return torch.load(OUT / "fit_issued" / f"{kind}_{start}.pt", map_location="cuda", weights_only=False)
    if kind == "diffusion":
        return torch.load(PREDICTOR / "issued/augmented_diffusion" / f"{split}_{start}.pt", map_location="cuda", weights_only=False)
    path = OUT / "calibration_cnn" / f"{start}.pt" if split == "calibration" else PREDICTOR / "issued/augmented_cnn" / f"{split}_{start}.pt"
    raw = torch.load(path, map_location="cuda", weights_only=False)
    return dict(mean=raw["zero"], std32=torch.zeros_like(raw["zero"]))


@torch.no_grad()
def cache_forecasts():
    norm = normalization()
    (OUT / "fit_issued").mkdir()
    (OUT / "calibration_cnn").mkdir()
    for kind in KINDS:
        arm = "augmented_diffusion" if kind == "diffusion" else "augmented_cnn"
        model = new_model(arm).eval().requires_grad_(False)
        model.load_state_dict(torch.load(PREDICTOR / arm / "model.pt", map_location="cuda", weights_only=False)["model"])
        for start in FIT_STARTS:
            observed = history("fit", start)
            h = (observed - norm["mean"]) / norm["std"]
            if kind == "diffusion":
                paths = torch.stack([model.rollout(h, seed) * norm["std"] + norm["mean"] for seed in range(500000, 500032)])
                value = dict(mean=paths.mean(0).cpu(), std32=paths.std(0).cpu())
                del paths
            else:
                p = model.net(torch.zeros_like(h), torch.full((len(h),), 99, device="cuda", dtype=torch.long), h)
                p = p * norm["std"] + norm["mean"]
                value = dict(mean=p.cpu(), std32=torch.zeros_like(p).cpu())
            torch.save(value, OUT / "fit_issued" / f"{kind}_{start}.pt")
            event(event="fit_forecast_issued", kind=kind, start=start)
        if kind == "cnn":
            for start in range(0, 161, 20):
                observed = history("calibration", start)
                h = (observed - norm["mean"]) / norm["std"]
                p = model.net(torch.zeros_like(h), torch.full((len(h),), 99, device="cuda", dtype=torch.long), h)
                torch.save(dict(zero=(p * norm["std"] + norm["mean"]).cpu()), OUT / "calibration_cnn" / f"{start}.pt")
        del model
    dump(OUT / "cache_complete.json", dict(time=time.time(), fit_records=len(FIT_RECORDS), windows=9, paths=32))


def normalize_inputs(value, mode, kind, stats):
    inputs = ((value - stats["mean"]) / stats["std"]).clamp(-10, 10)
    if mode in ("base", "spread"):
        inputs[:, 14:17] = 0
    if mode == "base" or kind == "cnn":
        inputs[:, 17:19] = 0
    return inputs


def simple_profiles(base, spread, memories, available, template, kind):
    maps = dict(constant=torch.ones_like(spread), history_variability=base[:, 4:6],
                departure=base[:, 6:8].abs(), gradient=base[:, 8:10],
                template=template[None].expand_as(spread))
    for mode in ("static", "advected", "reverse"):
        maps["memory_" + mode] = memories[mode] if available else maps["template"]
    if kind == "diffusion":
        maps["spread_raw"] = spread
        maps["spread_power"] = spread
    return maps


@torch.no_grad()
def fit_pack(kind):
    norm = normalization()
    geo = geometry(FIT_RECORDS)
    memory = CausalRiskMemory(norm["std"])
    bases, spreads, errors, available_rows = [], [], [], []
    memory_rows = {mode: [] for mode in ("static", "advected", "reverse")}
    inputs = {mode: [] for mode in MODES[kind]}
    for start in FIT_STARTS:
        h = history("fit", start)
        p = issued(kind, "fit", start)
        memories, available = memory.issue(h, p["mean"], geo)
        base = base_features(h, p["mean"], norm["mean"], norm["std"], geo)
        spread = p["std32"] / norm["std"]
        for mode in MODES[kind]:
            inputs[mode].append(assemble_features(base, memories, available, spread, mode, kind == "diffusion"))
        event(event="fit_risk_features_issued", kind=kind, start=start, previous_revealed_windows=memory.revealed_windows)
        y = target("fit", start)
        event(event="fit_risk_target_revealed", kind=kind, start=start)
        errors.append(((y - p["mean"]) / norm["std"]).abs())
        memory.reveal(y)
        bases.append(base)
        spreads.append(spread)
        available_rows.append(available)
        for mode in memory_rows:
            memory_rows[mode].append(memories[mode])
    return dict(inputs={mode: torch.cat(values) for mode, values in inputs.items()},
                base=torch.cat(bases), spread=torch.cat(spreads), error=torch.cat(errors),
                memories={mode: torch.cat(values) for mode, values in memory_rows.items()},
                available=available_rows)


def train_risk_heads():
    for kind in KINDS:
        folder = OUT / "controls" / kind
        folder.mkdir(parents=True)
        pack = fit_pack(kind)
        reference = pack["inputs"]["advected"]
        stats = dict(mean=reference.mean((0, 2, 3, 4), keepdim=True),
                     std=reference.std((0, 2, 3, 4), keepdim=True).clamp_min(.05))
        template = pack["error"].square().mean(0).sqrt()
        torch.save(dict(mean=stats["mean"].cpu(), std=stats["std"].cpu(), template=template.cpu()), folder / "statistics.pt")
        label = (pack["error"] + 1e-4).log()
        learned = {}
        for mode in MODES[kind]:
            value = normalize_inputs(pack["inputs"][mode], mode, kind, stats)
            torch.manual_seed(42)
            head = SpatiotemporalRiskNet().cuda()
            optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-5)
            logs = []
            begin = time.time()
            for step in range(1, 2001):
                ix = torch.randint(len(value), (4,), device="cuda")
                lr = 1e-6 + (3e-4 - 1e-6) * .5 * (1 + math.cos(math.pi * (step - 1) / 2000))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss = (head(value[ix]) - label[ix]).square().mean()
                assert torch.isfinite(loss)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                assert torch.isfinite(gradient)
                optimizer.step()
                if step == 1 or step % 500 == 0:
                    logs.append(dict(step=step, loss=float(loss), elapsed=time.time() - begin))
                    dump(folder / f"{mode}_training.json", logs)
                    print("RISK", kind, mode, logs[-1], flush=True)
            head.eval().requires_grad_(False)
            with torch.no_grad():
                learned["head_" + mode] = torch.cat([head(value[i:i + 4]).exp() for i in range(0, len(value), 4)])
            torch.save(dict(model=head.state_dict(), fit_files=[r["file"] for r in FIT_RECORDS],
                            fit_frames=[200, 400], seed=42, steps=2000), folder / f"{mode}.pt")
            del head, value, optimizer
        with torch.no_grad():
            maps = simple_profiles(pack["base"], pack["spread"], pack["memories"], True, template, kind)
            # Replace first-window memory by the same train-fitted cold template.
            for mode in ("static", "advected", "reverse"):
                maps["memory_" + mode] = maps["memory_" + mode].clone()
                maps["memory_" + mode][:len(FIT_RECORDS)] = template
            maps.update(learned)
            y = label.permute(1, 0, 2, 3, 4).flatten(1).double()
            betas = {}
            for name, profile in maps.items():
                x = profile.clamp_min(1e-4).log().permute(1, 0, 2, 3, 4).flatten(1).double()
                variance = x.var(dim=1, unbiased=False)
                covariance = ((x - x.mean(1, keepdim=True)) * (y - y.mean(1, keepdim=True))).mean(1)
                beta = torch.where(variance > 1e-12, covariance / variance.clamp_min(1e-12), 0).clamp(0, 2)
                if name == "spread_raw":
                    beta = torch.ones_like(beta)
                betas[name] = beta.cpu().tolist()
            dump(folder / "powers.json", betas)
        del pack, learned, maps, label
        torch.cuda.empty_cache()
    dump(OUT / "controls_frozen.json", dict(time=time.time(), artifacts={str(p.relative_to(OUT)): digest(p)
         for p in (OUT / "controls").rglob("*") if p.is_file()}))


class RiskProvider:
    def __init__(self, kind):
        self.kind = kind
        self.norm = normalization()
        folder = OUT / "controls" / kind
        self.stats = torch.load(folder / "statistics.pt", map_location="cuda", weights_only=False)
        self.powers = json.loads((folder / "powers.json").read_text())
        self.heads = {}
        for mode in MODES[kind]:
            head = SpatiotemporalRiskNet().cuda().eval().requires_grad_(False)
            head.load_state_dict(torch.load(folder / f"{mode}.pt", map_location="cuda", weights_only=False)["model"])
            self.heads[mode] = head

    @torch.no_grad()
    def profiles(self, h, p, memories, available, geo):
        base = base_features(h, p["mean"], self.norm["mean"], self.norm["std"], geo)
        spread = p["std32"] / self.norm["std"]
        maps = simple_profiles(base, spread, memories, available, self.stats["template"], self.kind)
        for mode, head in self.heads.items():
            inputs = assemble_features(base, memories, available, spread, mode, self.kind == "diffusion")
            inputs = normalize_inputs(inputs, mode, self.kind, self.stats)
            maps["head_" + mode] = head(inputs).exp()
        return {name: value.clamp_min(1e-4).pow(value.new_tensor(self.powers[name]).reshape(1, 2, 1, 1, 1))
                for name, value in maps.items()}


@torch.no_grad()
def calibrate():
    factors = {}
    geo = geometry(manifest()["files"]["calibration"])
    for kind in KINDS:
        provider = RiskProvider(kind)
        memory = CausalRiskMemory(provider.norm["std"])
        scores = {name: [] for name in provider.powers}
        for start in range(0, 161, 20):
            h, p = history("calibration", start), issued(kind, "calibration", start)
            memories, available = memory.issue(h, p["mean"], geo)
            profiles = provider.profiles(h, p, memories, available, geo)
            event(event="calibration_intervals_issued", kind=kind, start=start)
            y = target("calibration", start)
            error = ((y - p["mean"]) / provider.norm["std"]).abs()
            for name, profile in profiles.items():
                scores[name].append((error / profile).permute(1, 0, 2, 3, 4).reshape(2, -1).cpu())
            memory.reveal(y)
        factors[kind] = {name: torch.quantile(torch.cat(values, dim=1).cuda(), .9, dim=1).cpu().tolist()
                         for name, values in scores.items()}
    dump(OUT / "calibration_locked.json", dict(factors=factors, time=time.time(),
         interpretation="Pooled empirical marginal calibration; correlated pixels/windows are not independent conformal units."))


def initialize():
    OUT.mkdir(exist_ok=False)
    dump(OUT / "protocol.json", dict(time=time.time(), predictors="Frozen augmented Direct20 Diffusion32mean and matched CNN; no point changes.",
         risk_fit_files=[r["file"] for r in FIT_RECORDS], risk_fit_frames=[200, 400], predictor_fit_frames=[0, 200],
         short_file_excluded=[r["file"] for r in manifest()["files"]["train"] if r["frames"] < 400],
         calibration_files=[r["file"] for r in manifest()["files"]["calibration"]], calibration_frames=[0, 200],
         risk_head="19input channels, shared3D conv16/16/2;2000fixed updates,batch4,seed42; log absolute normalized error targets.",
         modes=MODES, transport="Absolute last-revealed error;20 semi-Lagrangian steps, predicted midpoint velocity, signed grid spacings and stored dt; static and reverse-direction controls.",
         maps="Constant, history variability, departure, gradient, template, raw/power spread, pure memory and matched learned heads; train-only power shapes.",
         evaluation="Unchanged development/test39windows; interval issue before target; only complete-window feedback; primary memory analysis excludes cold first window.",
         calibration="Independent trajectories, per-channel pooled90% quantile; compare fixed/identical geometric online update. Not a conformal guarantee.",
         metrics="Risk ranking, selective risk, per-frame localization, width, interval score, coverage, worst patch/frame, window95 and joint block coverage; group-equal summaries.",
         selection="No validation/test head or checkpoint selection, crossfolds, distillation, extra predictor training or training seeds.",
         limitations=["All data historically reused; exploratory.", "Risk-fit times are out-of-predictor-fit, but share its training trajectories/Re groups.",
                     "Only10calibration files in2Re clusters; cannot assume pixels or windows independent.",
                     "Advection assumes stored velocity and coordinate/time units compatible; error is not necessarily a passive scalar.",
                     "No ACI/CRC guarantee is claimed for geometric scale adaptation."],
         references=["https://arxiv.org/abs/2208.02814", "https://arxiv.org/abs/2106.00170", "https://arxiv.org/abs/2608.28515", "https://proceedings.mlr.press/v235/zhou24l.html"],
         predictor_sha256={arm: digest(PREDICTOR / arm / "model.pt") for arm in ("augmented_diffusion", "augmented_cnn")},
         code_sha256={name: digest(ROOT / name) for name in (Path(__file__).name, "reliability_risk_transport.py")}))
    dump(OUT / "preflight.json", preflight())


def main():
    configure()
    assert os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "0"
    initialize()
    cache_forecasts()
    train_risk_heads()
    calibrate()
    # Replay is separate so all fitted artifacts are frozen before evaluation.
    print("FITTING AND CALIBRATION COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    main()
