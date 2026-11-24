"""Verified original member means and old trajectory partitions."""
from itertools import combinations
from pathlib import Path

import torch

from research.corrector.run import sha
from research.corrector_window_study.data import LOCAL, RESULTS, ROOT, StudyData, read_json, tensor_sha

OUT = ROOT / "results/corrector_consistency"
STARTS = (400, 420)
SPLITS = ("train", "development", "test")


def sources():
    current = RESULTS / "cache_manifest.json"
    config = RESULTS / "config.json"
    original = ROOT / "results/residual_dynamics/protocol.json"
    manifest = read_json(current)
    assert sha(current) == read_json(config)["cache_manifest_sha256"]
    assert sha(original) == manifest["protocol_sha256"]
    protocol = read_json(original)
    assert manifest["seeds"] == protocol["member_seeds"] == list(range(500000, 500032))
    weight_hashes = manifest["weights"]
    for name, digest in weight_hashes.items():
        assert sha(Path(name)) == digest, name
    provenance = dict(window_cache_manifest_sha256=sha(current), previous_config_sha256=sha(config),
                      original_protocol_sha256=sha(original), fixed_weights=weight_hashes,
                      member_seeds=manifest["seeds"], starts=list(STARTS), splits={})
    for split, n in zip(SPLITS, (46, 12, 12)):
        spec = manifest["splits"][split]
        records = protocol["records"][split]
        assert spec["files"] == [r["file"] for r in records] and len(records) == n
        provenance["splits"][split] = dict(files=spec["files"],
            native={str(s): spec["native"][str(s)] for s in STARTS},
            observed_sha256=spec["observed_sha256"], labels_sha256=spec["labels_sha256"])
    for a, b in combinations(SPLITS, 2):
        assert set(provenance["splits"][a]["files"]).isdisjoint(provenance["splits"][b]["files"])
    return manifest, provenance


class Split:
    def __init__(self, manifest, split, device):
        spec = manifest["splits"][split]
        obs = StudyData.load(LOCAL / f"{split}_observed.pt", spec["observed_sha256"], "cpu")
        labels = StudyData.load(LOCAL / f"{split}_labels.pt", spec["labels_sha256"], "cpu")
        self.cases, xs, ps, ys = [], [], [], []
        for start in STARTS:
            row = spec["native"][str(start)]
            identity = row["identity"]
            assert identity == dict(split=split, start=start, files=spec["files"],
                                    seeds=manifest["seeds"], frozen=manifest["native_checkpoint"])
            keep = [(i, c) for i, c in enumerate(spec["cases"]) if c["start"] == start]
            assert [c["file"] for _, c in keep] == spec["files"]
            for j, (_, c) in enumerate(keep):
                assert c["index"] == j
                assert c["history"] == [start, start + 20] and c["target"] == [start + 20, start + 40]
            k = start - 380
            history = obs[:, :, k:k+20].contiguous()
            target = labels[[i for i, _ in keep]].contiguous()
            assert tensor_sha(history) == row["history_sha256"] and tensor_sha(target) == row["truth_sha256"]
            assert torch.equal(target, obs[:, :, k+20:k+40])
            paths = StudyData.load(LOCAL / f"{split}_{start}.pt", row["sha256"], device)
            assert paths.shape == (len(keep), 32, 2, 20, 32, 64)
            assert torch.isfinite(paths).all()
            self.cases.extend(c for _, c in keep)
            xs.append(history.to(device))
            ps.append(paths.mean(1))
            ys.append(target.to(device))
        self.x, self.p, self.y = (torch.cat(parts) for parts in (xs, ps, ys))
        assert self.x.shape == self.p.shape == self.y.shape
