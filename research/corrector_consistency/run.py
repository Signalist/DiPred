"""Fixed shared warm-start training, selection, and deterministic residual scoring."""
import argparse
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch

from research.corrector.run import dump, sha, table
from research.corrector_window_study.data import tensor_sha

from .data import OUT, Split, sources
from .metrics import aggregate, case_metrics, comparisons, local_gradient
from .model import ARMS, WEIGHT, PointCorrector, loss
from .verify import verify

SEED = 42
BATCH = 8
STEPS = (0, 250, 500, 750, 1000)
ARMS_WITH_NATIVE = ("Native", *ARMS)


def make_model(device):
    return PointCorrector().to(device=device, memory_format=torch.channels_last)


def normalized(split, mean, units):
    return (split.x - mean) / units, (split.p - mean) / units


@torch.inference_mode()
def development_mse(model, data, inputs, scale):
    model.eval()
    values = []
    for x, p, native, target in zip(inputs[0].split(BATCH), inputs[1].split(BATCH),
                                    data.p.split(BATCH), data.y.split(BATCH)):
        q = native.double() + (model(x, p) * scale).double()
        values.extend((q - target).square().flatten(1).mean(1).tolist())
    # The partition loader enforces the same two windows on every trajectory.
    return float(np.mean(values))


def fit(train, dev, mean, units, scale, schedule, out):
    inputs = normalized(train, mean, units)
    held_out = normalized(dev, mean, units)
    r_true = (train.y - train.p) / scale
    model = make_model(train.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    log = []
    folder = out / "checkpoints"
    folder.mkdir()

    def save(arm, step, aggregate_loss=None):
        mse = development_mse(model, dev, held_out, scale)
        name = f"{arm}_{step}.pt"
        path = folder / name
        torch.save(dict(arm=arm, step=step, model=model.state_dict(),
                        optimizer=optimizer.state_dict()), path)
        row = dict(arm=arm, step=step, development_mse=mse,
                   previous_250_updates_mean_training_loss=aggregate_loss,
                   checkpoint=f"checkpoints/{name}", sha256=sha(path))
        log.append(row)
        table(out / "training.csv", log)
        print("CHECKPOINT", arm, step, f"development_mse={mse:.12g}", flush=True)
        return row

    def update(arm, first, stop):
        model.train()
        values = []
        for i in schedule[first:stop]:
            optimizer.zero_grad(set_to_none=True)
            p = model(inputs[0][i], inputs[1][i])
            v = loss(p, r_true[i], arm)
            v.backward()
            optimizer.step()
            values.append(v.detach())
        average = torch.stack(values).mean()
        assert torch.isfinite(average)
        return float(average)

    zero = save("shared", 0)
    shared = save("shared", 250, update("A", 0, 250))
    selected, finals = {}, {}
    for arm in ARMS:
        state = torch.load(out / shared["checkpoint"], map_location=train.x.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        assert all(int(v["step"]) == 250 for v in optimizer.state.values())
        candidates = [zero, shared]
        for step in STEPS[2:]:
            candidates.append(save(arm, step, update(arm, step-250, step)))
        selected[arm] = min(candidates, key=lambda r: (r["development_mse"], r["step"]))
        finals[arm] = candidates[-1]
    assert len(log) == 23
    dump(out / "selections.json", dict(selected=selected, final=finals, shared=[zero, shared]))
    return selected, finals


@torch.inference_mode()
def audit_checkpoints(test, mean, units, scale, thresholds, selected, finals, out):
    windows, trajectory, summary = [], [], []
    inputs = normalized(test, mean, units)
    model = make_model(test.x.device).eval().requires_grad_(False)
    for phase, checkpoints in (("selected", selected), ("step1000", finals)):
        for arm in ARMS_WITH_NATIVE:
            use_model = arm != "Native"
            if use_model:
                record = checkpoints[arm]
                assert sha(out / record["checkpoint"]) == record["sha256"]
                ck = torch.load(out / record["checkpoint"], map_location=test.x.device, weights_only=False)
                assert ck["step"] == record["step"]
                model.load_state_dict(ck["model"])
            rows = []
            correction = (model(*inputs) * scale if use_model else torch.zeros_like(test.p))
            assert torch.isfinite(correction).all()
            for i in range(0, len(test.cases), BATCH):
                n = slice(i, i+BATCH)
                diagnostics = case_metrics(test.p[n], test.y[n], correction[n], units, scale, thresholds)
                for c, v in zip(test.cases[n], diagnostics):
                    rows.append(dict(phase=phase, arm=arm, trajectory=c["file"],
                                     re=c["re"], start=c["start"], **v))
            trs, avg = aggregate(rows, phase, arm)
            trajectory.extend(trs)
            summary.append(avg)
            windows.extend(rows)

    all_leads = []
    native = {(r["phase"], r["trajectory"], r["start"]): r for r in windows if r["arm"] == "Native"}
    baseline = {(r["phase"], r["trajectory"], r["start"]): r for r in windows if r["arm"] == "A"}
    for row in windows:
        key = row["phase"], row["trajectory"], row["start"]
        row["improve_native_pct"] = 100 * (1 - row["mse"] / native[key]["mse"])
        row["improve_A_pct"] = 100 * (1 - row["mse"] / baseline[key]["mse"])
        for k in range(1, 21):
            all_leads.append(dict(phase=row["phase"], arm=row["arm"], trajectory=row["trajectory"],
                                  start=row["start"], lead=k, mse=row[f"lead_{k:02d}"]))
    for group in (trajectory, summary):
        keyfields = ("phase", "trajectory") if group is trajectory else ("phase",)
        ref = {tuple(r[k] for k in keyfields)+(r["arm"],): r["mse"] for r in group}
        for row in group:
            key = tuple(row[k] for k in keyfields)
            for arm in ("Native", "A"):
                row[f"improve_{arm}_pct"] = 100 * (1 - row["mse"] / ref[key+(arm,)])
    table(out / "windows.csv", windows)
    table(out / "trajectories.csv", trajectory)
    table(out / "leads.csv", all_leads)
    table(out / "scores.csv", summary)
    table(out / "contrasts.csv", comparisons(trajectory))
    return dict(maximum_mse_decomposition_error=max(r["identity_error"] for r in windows),
                test_windows=len(test.cases), test_trajectories=len({r["trajectory"] for r in windows}),
                finite_results=all(np.isfinite(v) for r in windows for v in r.values()
                                   if isinstance(v, (int, float))))


@torch.inference_mode()
def save_plot_case(dev, mean, units, scale, finals, out):
    """Keep one predeclared development case showing the step-1000 corrections."""
    native = dev.p[:1]
    target = dev.y[0] - native[0]
    fields = dict(target=target.cpu().numpy(), Native=torch.zeros_like(target).cpu().numpy())
    x, p = (dev.x[:1]-mean)/units, (native-mean)/units
    model = make_model(dev.x.device).eval().requires_grad_(False)
    for arm, choice in finals.items():
        checkpoint = out / choice["checkpoint"]
        assert sha(checkpoint) == choice["sha256"]
        state = torch.load(checkpoint, map_location=dev.x.device, weights_only=False)
        assert state["step"] == 1000
        model.load_state_dict(state["model"])
        fields[arm] = (model(x, p)[0] * scale).cpu().numpy()
    np.savez_compressed(out / "plot_case.npz", **fields)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    assert not (out / "checkpoints").exists(), "Choose a fresh experiment output path"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print("COMMAND", shlex.join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]), flush=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    manifest, provenance = sources()
    train = Split(manifest, "train", device)
    dev = Split(manifest, "development", device)
    dump(out / "sources.json", provenance)
    mean = train.x.double().mean((0, 2, 3, 4)).float().view(1, 2, 1, 1, 1)
    units = train.x.double().std((0, 2, 3, 4), unbiased=False).float().view_as(mean)
    scale = (train.y.double() - train.p.double()).square().mean().sqrt().float()
    residual = (train.y - train.p) / scale
    thresholds = dict(gradient_q90=float(torch.quantile(local_gradient(residual).flatten(), .9)),
                      amplitude_q90=float(torch.quantile(
                          torch.linalg.vector_norm(residual, dim=1).flatten(), .9)))
    checks = verify(dev, mean, units, scale)
    print("PREFLIGHT PASS", checks, flush=True)
    torch.manual_seed(SEED)
    schedule = torch.randint(len(train.cases), (1000, BATCH),
                             generator=torch.Generator().manual_seed(SEED))
    params = sum(p.numel() for p in make_model(device).parameters())
    torch.manual_seed(SEED)
    case = dict(trajectory=dev.cases[0]["file"], start=400, lead=20, grid=[16, 32])
    assert dev.cases[0]["start"] == 400
    dump(out / "config.json", dict(seed=SEED, batch=BATCH, steps=STEPS, adam_lr=.001,
         parameters=params, arms=ARMS, constraint_weight=WEIGHT, plot_case=case,
         schedule_sha256=tensor_sha(schedule), training_rms=float(scale),
         channel_mean=mean.flatten().tolist(), channel_scale=units.flatten().tolist(),
         thresholds=thresholds, torch=torch.__version__, python=sys.version.split()[0],
         cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
         fit_windows=len(train.cases), development_windows=len(dev.cases), precision="float32"))
    print("FIT", len(train.cases), len(dev.cases), params, thresholds, flush=True)
    selected, finals = fit(train, dev, mean, units, scale, schedule.to(device), out)
    print("FROZEN", {arm: row["step"] for arm, row in selected.items()}, flush=True)
    test = Split(manifest, "test", device)
    checks.update(audit_checkpoints(test, mean, units, scale, thresholds, selected, finals, out))
    save_plot_case(dev, mean, units, scale, finals, out)
    _, after = sources()
    assert provenance == after
    checks.update(frozen_weights_unchanged=True, wall_seconds=time.monotonic()-started,
                  optimizer_updates_executed=250+7*750, peak_gpu_bytes=torch.cuda.max_memory_allocated())
    dump(out / "verification.json", checks)
    print("PASS", checks, flush=True)


if __name__ == "__main__":
    main()
