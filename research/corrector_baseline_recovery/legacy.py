"""Replay the pinned old deterministic batch-one baseline and checkpoint."""
from collections import defaultdict
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from reliability.backbone import configure
from research.corrector.run import dump, sha, table
from research.corrector_window_study.data import ROOT, WORKSPACE, read_json, tensor_sha
from research.residual_dynamics.run import load_backbone

from . import crossfit_corrector
from .crossfit_corrector import BlockResidualCorrector

OUT = ROOT / "results/corrector_baseline_recovery"
RUNS = WORKSPACE / "runs"
OLD = RUNS / "direct_correction_controls_20261117"
OLD_XY = RUNS / "error_structure_20261117"
OLD_P = RUNS / "active_observation_20261117"
PINNED = {
    "commit": "333934b60b56f6faf97aedd3bd7d1d0ca79c2583",
    "selection.json": "43a4fc7b263937971e0f6ea51346e9f2adad742b9b90fe8e0ccdfbcda541b564",
    "controls.py": "0156cf4d18bbe136fe69e1d713fee7876deed3dc3dc05f829c23797d702e2233",
    "PROTOCOL.md": "ed2c860a87e8edd2a0af0413e9b7cdbdef88a428bf8931fc3e4a18f39b8a160a",
    "REPORT.md": "325a40365d2ce919af4cf0b97fae6a2b768884389b00cfbd46b3a40ce8e88051",
    "crossfit_corrector.py": "e9378147c0a4b749ee996fe6cc64afe73ba24b3078216ac1b1768db2fea95a7f",
}


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def old_sources():
    sources = {
        "selection.json": OLD / "selection.json",
        "controls.py": WORKSPACE / "direct_correction_controls_20261117/controls.py",
        "PROTOCOL.md": WORKSPACE / "direct_correction_controls_20261117/PROTOCOL.md",
        "REPORT.md": OLD / "REPORT.md",
        "crossfit_corrector.py": Path(crossfit_corrector.__file__),
    }
    for name, path in sources.items():
        assert sha(path) == PINNED[name], path
    selected = read_json(sources["selection.json"])
    assert sha(OLD / "rescrct_block.pt") == selected["model_hashes"]["rescrct_block"]
    return selected


@torch.inference_mode()
def replay(out):
    selected = old_sources()
    checkpoint = OLD / "rescrct_block.pt"
    weights = load(checkpoint)
    cfg = weights["summary"]
    assert json.loads(json.dumps(cfg)) == selected["models"]["rescrct_block"] and cfg["gain"] == .5
    model, norm, backbone = load_backbone(SimpleNamespace(workspace=WORKSPACE))
    head = BlockResidualCorrector().cuda().eval().requires_grad_(False)
    head.load_state_dict(weights["model"])
    units = norm["std"][0].double().cpu().numpy()
    hashes = {str(checkpoint.relative_to(WORKSPACE)): sha(checkpoint)}
    rows, summaries = [], []
    for split, count in (("calibration", 90), ("development", 108), ("late", 120), ("future", 72)):
        xy_path, p_path = OLD_XY / f"{split}.pt", OLD_P / f"{split}.pt"
        issued_path = OLD / split / "issued.pt"
        xy, actual, issued = (load(path) for path in (xy_path, p_path, issued_path))
        for path in (xy_path, p_path, issued_path):
            hashes[str(path.relative_to(WORKSPACE))] = sha(path)
        assert xy["meta"] == actual["meta"] and len(xy["meta"]) == count
        split_tracks = {r["file"] for r in xy["meta"]}
        assert not split_tracks & set(cfg["train_files"])
        assert (split_tracks == set(cfg["calibration_files"]) if split == "calibration"
                else not split_tracks & set(cfg["calibration_files"]))
        by_track = defaultdict(list)
        max_point, max_issued = 0., 0.
        for i, case in enumerate(xy["meta"]):
            x = xy["x"][i:i+1].cuda()
            p = model.rollout(x, 500000, sign=0)
            raw = head(x, p)
            q = p + cfg["gain"] * (raw - p)
            max_point = max(max_point, float((p.cpu()[0] - actual["p"][i]).abs().max()))
            max_issued = max(max_issued, float((q.cpu()[0] - issued["rescrct_block"][i]).abs().max()))
            y = xy["y"][i].double().numpy()
            mses = [float(np.mean(((z.cpu()[0].double().numpy() - y) * units)**2))
                    for z in (p, q, raw)]
            rows.append(dict(split=split, **case, native_mse=mses[0], selected_mse=mses[1], raw_mse=mses[2]))
            by_track[case["file"]].append(mses)
        means = np.stack([np.mean(v, axis=0) for v in by_track.values()])
        native, calibrated, raw = map(float, means.mean(0))
        summary_file = OLD / split / "summary.csv"
        with summary_file.open() as file:
            expected = {r["method"]: r for r in csv.DictReader(file)}
        error = max(abs(native - float(expected["base"]["mse"])),
                    abs(calibrated - float(expected["rescrct_block"]["mse"])))
        assert max_point == max_issued == 0. and error < 1e-9, (split, max_point, max_issued, error)
        summaries.append(dict(split=split, windows=count, trajectories=len(by_track),
                              native_mse=native, selected_mse=calibrated, raw_mse=raw,
                              selected_gain_pct=100*(1-calibrated/native),
                              improved_trajectories=int((means[:, 1] < means[:, 0]).sum()),
                              prediction_max_error=max_issued, published_mse_max_error=error))
        print(summaries[-1], flush=True)
    train_path = OLD_XY / "train.pt"
    actual_train_path = OLD_P / "train.pt"
    train, actual_train = load(train_path), load(actual_train_path)
    assert train["meta"] == actual_train["meta"] and len(train["meta"]) == 819
    assert {r["file"] for r in train["meta"]} == set(cfg["train_files"])
    for path in (train_path, actual_train_path):
        hashes[str(path.relative_to(WORKSPACE))] = sha(path)
    dump(out / "legacy_windows_819.json", train["meta"])
    dump(out / "legacy_replay.json", dict(pinned=PINNED, source_sha256=hashes,
                                         gain=cfg["gain"], backbone=backbone,
                                         normalization={k: v.flatten().tolist() for k, v in norm.items()},
                                         train_history_sha256=tensor_sha(train["x"]), scores=summaries))
    table(out / "legacy_replay_windows.csv", rows)


if __name__ == "__main__":
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    replay(OUT)
