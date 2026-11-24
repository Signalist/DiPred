"""Verified member means and trajectory-local labels."""
from pathlib import Path

import torch

from research.corrector.run import sha
from research.corrector_consistency.data import sources as consistency_sources
from research.corrector_window_study.data import LOCAL, ROOT, StudyData, read_json, tensor_sha

OUT = ROOT / "results/corrector_predictability"
CACHE = ROOT / ".cache/corrector_predictability"
OLD = ROOT / "results/corrector_consistency"
SMALL = (400, 420)
EXPANDED = (*SMALL, 440, 460)


def sources():
    manifest, lineage = consistency_sources()
    checkpoints = {r["arm"]: r for r in read_json(OLD / "selections.json")["final"].values()
                   if r["arm"] in "ABG"}
    assert set(checkpoints) == {"A", "B", "G"}
    for row in checkpoints.values():
        assert row["step"] == 1000 and sha(OLD / row["checkpoint"]) == row["sha256"]
    lineage["frozen_correctors"] = checkpoints
    lineage["previous_consistency_config_sha256"] = sha(OLD / "config.json")
    lineage["expanded_native"] = {str(k): manifest["splits"]["train"]["native"][str(k)]
                                  for k in EXPANDED}
    return manifest, lineage


class Split:
    def __init__(self, manifest, name, starts, device="cuda"):
        assert name == "train" or tuple(starts) == SMALL
        spec = manifest["splits"][name]
        obs = StudyData.load(LOCAL / f"{name}_observed.pt", spec["observed_sha256"], "cpu")
        labels = StudyData.load(LOCAL / f"{name}_labels.pt", spec["labels_sha256"], "cpu")
        self.cases, xs, ps, ys = [], [], [], []
        for start in starts:
            row = spec["native"][str(start)]
            assert row["identity"] == dict(split=name, start=start, files=spec["files"],
                                            seeds=manifest["seeds"], frozen=manifest["native_checkpoint"])
            keys, cases = zip(*[(i, c) for i, c in enumerate(spec["cases"]) if c["start"] == start])
            indices = [c["index"] for c in cases]
            assert indices == sorted(set(indices))
            for c in cases:
                assert c["file"] == spec["files"][c["index"]]
                assert c["history"] == [start, start+20] and c["target"] == [start+20, start+40]
            k = start - 380
            history = obs[:, :, k:k+20].contiguous()
            truth = labels[list(keys)].contiguous()
            assert tensor_sha(history) == row["history_sha256"]
            if "truth_sha256" in row:
                assert tensor_sha(truth) == row["truth_sha256"]
            assert torch.equal(truth, obs[indices, :, k+20:k+40])
            members = StudyData.load(LOCAL / f"{name}_{start}.pt", row["sha256"], device)
            assert members.shape == (len(spec["files"]), 32, 2, 20, 32, 64)
            assert torch.isfinite(members).all()
            self.cases.extend(cases)
            xs.append(history[indices].to(device))
            ps.append(members.mean(1)[indices])
            ys.append(truth.to(device))
        self.x, self.p, self.y = (torch.cat(v) for v in (xs, ps, ys))
        assert self.x.shape == self.p.shape == self.y.shape
        assert len(self.cases) == (183 if tuple(starts) == EXPANDED else 92 if name == "train" else 24)

    def small(self):
        result = Split.__new__(Split)
        n = 92
        result.cases, result.x, result.p, result.y = self.cases[:n], self.x[:n], self.p[:n], self.y[:n]
        assert {c["start"] for c in result.cases} == set(SMALL)
        return result


def normalization(device):
    config = read_json(OLD / "config.json")
    mean, units = (torch.tensor(config[k], device=device).view(1, 2, 1, 1, 1)
                   for k in ("channel_mean", "channel_scale"))
    return mean, units, config["training_rms"], config["thresholds"]
