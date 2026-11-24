"""Lock and evaluate independent real trajectories with the frozen S2 and recent models."""

import argparse
import csv
import hashlib
from pathlib import Path
from statistics import mean

import h5py
import numpy as np
import torch

from reliability.backbone import configure
from research.corrector.run import dump, sha, table
from research.corrector_window_study.data import LOCAL, ROOT, STARTS, WORKSPACE, StudyData, read_json
from research.residual_dynamics.run import load_backbone

from .calibration import apply_scale, scales, static_basis
from .conditional import basis, release_features
from .mechanism import METRICS, evaluate

STUDY = ROOT / "results/corrector_distribution_study"
PREVIOUS = ROOT / "results/corrector_window_study"
METHODS = ("native", "S2", "recent")
PHYSICAL = ("u", "v")


def frozen_inputs():
    f = read_json(STUDY / "FROZEN.json")
    assert f["selected_simple"] == "S2"
    assert f["windows"] == list(STARTS) and f["members"] == list(range(500000, 500032))
    for name, digest in f["artifacts"].items():
        assert sha(STUDY / name) == digest
    for name, digest in f["code"].items():
        assert sha(Path(__file__).with_name(name)) == digest
    config = PREVIOUS / "config.json"
    sources = read_json(STUDY / "SOURCE_HASHES.json")
    assert sha(config) == sources[str(config.relative_to(ROOT))]
    return f, read_json(config)["normalization"]["scale"]


def fields(handle, start, stop):
    return np.stack([handle[channel][start:stop, ::2, ::2] for channel in PHYSICAL]).astype("float32")


def inventory(records):
    """Check physical layout and identities without inspecting forecast outcomes."""
    known = read_json(ROOT / "metadata/manifest.json")
    known_rows = [r for split in known["files"].values() for r in split]
    seen_names = {r["file"] for r in known_rows} | {r["excluded"] for r in known["duplicates"]}
    seen_fields = {r["sha256_first200"] for r in known_rows}
    seen_files = set()
    locked = []
    assert 1 <= len(records) <= 12, "Supply one to twelve independent trajectories"
    for r in records:
        path = Path(r["path"]).resolve(strict=True)
        for key in ("trajectory_id", "provenance"):
            assert isinstance(r[key], str) and r[key].strip(), f"Missing {key}: {path}"
        assert r["unseen_by_backbone_and_method_selection"] is True, f"Missing attestation: {path}"
        assert path.name not in seen_names, f"Previously used filename: {path.name}"
        digest = sha(path)
        assert digest not in seen_files, f"Repeated file: {path}"
        with h5py.File(path, "r") as handle:
            assert all(handle[c].ndim == 3 for c in PHYSICAL)
            shape = handle["u"].shape
            assert shape == handle["v"].shape and shape[1:] == (64, 128)
            assert shape[0] >= max(STARTS) + 40, f"Too few frames: {path}"
            prefix = fields(handle, 0, 200)
            assert np.isfinite(prefix).all()
            identity = hashlib.sha256(prefix.tobytes()).hexdigest()
        assert identity not in seen_fields, f"Prior or repeated physical trajectory: {path}"
        seen_fields.add(identity)
        seen_files.add(digest)
        seen_names.add(path.name)
        locked.append(dict(trajectory_id=r["trajectory_id"], path=str(path),
                           provenance=r["provenance"], sha256=digest, sha256_first200=identity,
                           frames=shape[0], unseen_by_backbone_and_method_selection=True))
    assert len({r["trajectory_id"] for r in locked}) == len(locked)
    return sorted(locked, key=lambda row: row["trajectory_id"])


def plan(path, out):
    f, _ = frozen_inputs()
    records = inventory(read_json(path))
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "LOCKED.json", dict(frozen_sha256=sha(STUDY / "FROZEN.json"),
         script_sha256=sha(Path(__file__)), training_or_tuning=False, cases=records,
         windows=f["windows"], member_seeds=f["members"], primary="recent minus S2 standardized CRPS",
         interval="95% paired trajectory percentile bootstrap, 10000 draws, seed 42"))
    print(f"Locked {len(records)} trajectories; no forecasts or scores computed.")


def calibrated(native, history, scale, f):
    # Recent uses only the first feature. The other slots are ignored by its
    # existing fitted basis, so no prior forecast or target enters the calculation.
    activity = (history.diff(dim=2) / scale).square().mean((2, 3, 4)).sqrt()
    release_only = torch.stack((activity, torch.zeros_like(activity), torch.zeros_like(activity)), -1)
    static = static_basis("S2", len(native), native.device)
    dynamic = basis("recent", release_only, f["conditional"]["feature_limits"])
    s2 = scales(static, f["static"]["S2"]["coefficients"])
    recent = scales(dynamic, f["conditional"]["models"]["recent"]["coefficients"])
    calibrated_s2 = apply_scale(native, s2)
    calibrated_recent = apply_scale(native, recent)
    order = native.argsort(dim=1)
    for paths in (calibrated_s2, calibrated_recent):
        torch.testing.assert_close(paths.mean(1).double(), native.mean(1).double(), atol=3e-7, rtol=0)
        assert (paths.gather(1, order).diff(dim=1) >= -1e-7).all()
    return dict(native=native, S2=calibrated_s2, recent=calibrated_recent), release_only


def read_block(records, start, stop):
    blocks = []
    for row in records:
        with h5py.File(row["path"], "r") as handle:
            block = fields(handle, start, stop)
        assert block.shape == (2, stop-start, 32, 64) and np.isfinite(block).all()
        blocks.append(block)
    return torch.from_numpy(np.stack(blocks)).cuda()


def summarize(rows, records, out):
    by_method = {method: [] for method in METHODS}
    for r in records:
        for method in METHODS:
            values = [row for row in rows if row["trajectory"] == r["trajectory_id"]
                      and row["method"] == method]
            assert sorted(row["start"] for row in values) == list(STARTS)
            by_method[method].append(dict(trajectory=r["trajectory_id"], method=method,
                **{metric: mean(row[metric] for row in values) for metric in METRICS}))
    trajectories = [row for group in by_method.values() for row in group]
    table(out / "windows.csv", rows)
    table(out / "trajectories.csv", trajectories)
    table(out / "scores.csv", [dict(method=method, trajectories=len(group),
          **{metric: mean(row[metric] for row in group) for metric in METRICS})
          for method, group in by_method.items()])
    delta = np.subtract([r["crps"] for r in by_method["recent"]],
                        [r["crps"] for r in by_method["S2"]])
    interval = None
    if len(delta) > 1:
        draws = np.random.default_rng(42).integers(len(delta), size=(10000, len(delta)))
        interval = np.quantile(delta[draws].mean(1), [0.025, 0.975]).tolist()
    return dict(primary="recent minus S2 standardized CRPS", difference=float(delta.mean()),
                interval95=interval, improved=int((delta < 0).sum()), trajectories=len(delta))


@torch.inference_mode()
def score(out):
    f, units = frozen_inputs()
    locked = read_json(out / "LOCKED.json")
    assert locked["frozen_sha256"] == sha(STUDY / "FROZEN.json")
    assert locked["script_sha256"] == sha(Path(__file__))
    assert locked["windows"] == f["windows"] and locked["member_seeds"] == f["members"]
    assert not (out / "RESULT.json").exists(), "Confirmation already scored"
    records = locked["cases"]
    for r in records:
        assert sha(Path(r["path"])) == r["sha256"], f"File changed after lock: {r['path']}"
    configure()
    args = argparse.Namespace(workspace=WORKSPACE)
    model, normalization, identity = load_backbone(args)
    assert identity == read_json(PREVIOUS / "cache_manifest.json")["native_checkpoint"]
    scale = torch.tensor(units, device="cuda").view(1, 2, 1, 1, 1)
    rows = []
    for start in STARTS:
        history = read_block(records, start, start+20)
        h = (history - normalization["mean"]) / normalization["std"]
        native = torch.stack([model.rollout(h, seed, sign=1) * normalization["std"]
                              + normalization["mean"] for seed in f["members"]], dim=1)
        predictions, _ = calibrated(native, history, scale, f)
        target = read_block(records, start+20, start+40)
        for method, paths in predictions.items():
            for i, r in enumerate(records):
                scores = evaluate(paths[i:i+1], target[i:i+1], scale)
                rows.append(dict(trajectory=r["trajectory_id"], start=start, method=method,
                                 **{k: scores[k][0] for k in METRICS}))
    dump(out / "RESULT.json", summarize(rows, records, out))


@torch.inference_mode()
def verify():
    f, units = frozen_inputs()
    configure()
    data = StudyData(LOCAL, read_json(PREVIOUS / "cache_manifest.json"), "development", "cuda")
    scale = torch.tensor(units, device="cuda").view(1, 2, 1, 1, 1)
    release = data.release([0, 1])
    forecasts, features = calibrated(release[2], release[0], scale, f)
    expected = release_features(release, scale)
    assert torch.equal(expected[:, :, 0], features[:, :, 0])
    recent = f["conditional"]
    alpha = scales(basis("recent", expected, recent["feature_limits"]),
                   recent["models"]["recent"]["coefficients"])
    assert torch.equal(forecasts["recent"], apply_scale(release[2], alpha))
    scale_s2 = scales(static_basis("S2", 2, "cuda"), f["static"]["S2"]["coefficients"])
    assert torch.equal(forecasts["S2"], apply_scale(release[2], scale_s2))
    stored = {}
    for name in ("development_cases.csv", "development_exploration_cases.csv"):
        with (STUDY / name).open(newline="") as handle:
            stored.update({(r["method"], r["trajectory"], int(r["start"])): r
                           for r in csv.DictReader(handle) if r["method"] in METHODS})
    maximum = 0.
    for method, paths in forecasts.items():
        actual = evaluate(paths, data.target([0, 1]), scale)
        for i, case in enumerate(data.cases[:2]):
            expected_row = stored[method, case["file"], case["start"]]
            maximum = max(maximum, *(abs(actual[k][i] - float(expected_row[k])) for k in METRICS))
    assert maximum < 2e-7, maximum
    print("PASS: frozen hashes, bitwise matched development predictions, preserved means and ranks, "
          f"historical score max absolute difference={maximum:.3g}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock")
    lock.add_argument("inventory", type=Path)
    lock.add_argument("out", type=Path)
    commands.add_parser("verify")
    run = commands.add_parser("score")
    run.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "lock":
        plan(args.inventory, args.out)
    elif args.command == "score":
        score(args.out)
    else:
        verify()


if __name__ == "__main__":
    main()
