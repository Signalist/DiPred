"""Select fixed records and validate the local evidence against the original run."""

import csv
import subprocess
from collections import defaultdict

import torch

from research.residual_uncertainty_diffusion.core import (
    ARMS, CACHE, OUT, ROOT, Correction, condition_inputs, digest, dump, read, state_hash, tensor_hash,
)
from research.residual_uncertainty_diffusion.experiment import freeze_hash
from research.trust_causal_audit_v2.common import DATA, MEMBERS, load_backbone
from research.trust_causal_audit_v2.data import H5Reader, batch_noise
from .tables import identity


REFERENCE = "bb9bfa805461b70004914cea02cb145be826d878"
NATIVE_WEIGHT = ROOT / "reliability/checkpoints/current.pt"
SOURCES = (
    "reliability/backbone.py", "reliability/networks.py",
    "research/trust_causal_audit_v2/common.py", "research/trust_causal_audit_v2/data.py",
    "research/trust_causal_audit_v2/model.py",
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def by_re(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["Re"]].append(row)
    return [sorted(grouped[re], key=lambda r: r["trajectory_id"]) for re in sorted(grouped, key=int)]


def select(destination):
    head = git("rev-parse", "HEAD")
    assert git("branch", "--show-current") == "trust"
    subprocess.run(["git", "merge-base", "--is-ancestor", REFERENCE, head], cwd=ROOT, check=True)
    rows = {
        split: [r for r in read(OUT / f"{split}_DATA_MANIFEST.json") if r["start"] == 420]
        for split in ("train", "development")
    }
    dev_groups = by_re(rows["development"])
    dev = [row for group in dev_groups for row in group]
    stage = [row["cache_key"] for group in dev_groups for row in group[:2]]
    fd = [group[0]["cache_key"] for group in dev_groups]
    train_groups = by_re(rows["train"])
    train = [group[i] for i in range(max(map(len, train_groups)))
             for group in train_groups if i < len(group)][:6]
    assert len(dev) == len({r["trajectory_id"] for r in dev}) == 12
    assert len(stage) == 6 and len(fd) == 3 and len(train) == 6
    for row in dev + train:
        assert row["history_interval"] == row["previous_target_interval"] == [420, 440]
        assert row["target_interval"] == [440, 460]
    dump(destination / "FIXED_MANIFEST.json", dict(
        reference=REFERENCE, head_before_run=head, branch="trust",
        worktree_status_before_run=git("status", "--short").splitlines(),
        selection_rule="numeric Re round-robin and first unused legal start420 filename in lexical order",
        training_eligible=[r["trajectory_id"] for group in train_groups for r in group],
        stage_keys=stage, fd_all_keys=fd, fd_individual_key=fd[0],
        member_ids=list(MEMBERS), records=dev + train,
    ))
    return dev + train, stage, fd


class Evidence:
    def __init__(self):
        self.files = {}
        old_hashes = read(OUT / "ARTIFACT_HASHES.json")
        for name in ("DECISION_REPORT.md", "CHECKPOINTS.json", "CONDITION_SCALES.json", "NOISE_MANIFEST.json",
                     "CACHE_HASHES.json", "DEVELOPMENT.csv", "DEVELOPMENT_MACRO.csv", "SELECTION.json",
                     "FROZEN.json", "TRAINING_COMPLETE.json", "train_DATA_MANIFEST.json",
                     "development_DATA_MANIFEST.json", "ENVIRONMENT.json"):
            path = OUT / name
            self.check(path, old_hashes[str(path.relative_to(ROOT))])
        for name, sha in read(OUT / "FROZEN.json")["code_hashes"].items():
            self.check(ROOT / name, sha)
        for name in SOURCES:
            path = ROOT / name
            assert git("hash-object", name) == git("rev-parse", f"{REFERENCE}:{name}"), name
            self.check(path)
        protected = read(OUT / "PROTECTED.json")
        self.check(NATIVE_WEIGHT, protected[str(NATIVE_WEIGHT.relative_to(ROOT))])
        self.cache_hashes = read(OUT / "CACHE_HASHES.json")
        self.checkpoints = {r["arm"]: r for r in read(OUT / "CHECKPOINTS.json") if r["step"] == 250}
        self.scales = read(OUT / "CONDITION_SCALES.json")
        noise = read(OUT / "NOISE_MANIFEST.json")
        self.noise_files = noise["split_batch_files"]
        self.noise_hashes = {
            (r["split"], r["trajectory_id"], r["member_id"]): r["noise_tensor_hash"]
            for r in noise["identities"] if r["history_start"] == 420
        }

    def check(self, path, expected=None):
        actual = digest(path)
        assert expected is None or actual == expected, f"SHA-256 mismatch: {path}"
        self.files[str(path.relative_to(ROOT))] = actual

    def load_cache(self, filename):
        path = CACHE / filename
        key = str(path.relative_to(ROOT))
        manifest = self.cache_hashes[key]
        assert path.stat().st_size == manifest["bytes"], path
        self.check(path, manifest["sha256"])
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_models(self, matmul_tf32=False):
        native, norm = load_backbone()
        assert torch.backends.cuda.matmul.allow_tf32 == matmul_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert torch.backends.cudnn.deterministic
        frozen = read(OUT / "FROZEN.json")
        train = read(OUT / "TRAINING_COMPLETE.json")
        assert state_hash(native) == frozen["initial_backbone_state"]
        assert state_hash(native.net.out) == train["backbone_out_initial_hash"]
        assert freeze_hash(native, exclude_out=True) == frozen["frozen_non_out"]
        out_states = {"native": {k: v.clone() for k, v in native.net.out.state_dict().items()}}
        corrections = {}
        for arm in ARMS:
            meta = self.checkpoints[arm]
            path = ROOT / meta["path"]
            assert path.name == f"{arm}_250.pt"
            self.check(path, meta["sha256"])
            ck = torch.load(path, map_location="cuda", weights_only=False)
            assert (ck["arm"], ck["step"]) == (arm, 250)
            out_states[arm] = ck["out"]
            correction = Correction().cuda().eval().requires_grad_(False)
            correction.load_state_dict(ck["correction"], strict=True)
            assert state_hash(correction) == meta["correction_state_hash"]
            corrections[arm] = correction
        return native, norm, out_states, corrections

    def load_records(self, records, norm):
        conditions = {}
        for split in ("train", "development"):
            selected = {r["cache_key"] for r in records if r["split"] == split}
            conditions.update({r["cache_key"]: r for r in self.load_cache(f"{split}_conditions.pt")
                               if r["cache_key"] in selected})
        reader = H5Reader(DATA)
        inputs, targets, details = {}, {}, []
        poisoned_inputs = None
        for row in records:
            key = row["cache_key"]
            item = conditions[key]
            assert all(row[k] == v for k, v in identity(item).items())
            for field in ("history", "E", "W", "previous_path"):
                assert item[f"{field}_sha256"] == row[f"{field}_sha256"]
            for field in ("E", "W"):
                assert tensor_hash(item[field]) == row[f"{field}_sha256"]
            assert item["previous_interval"] == row["history_interval"]
            # Cross-check the raw historical slice and its normalized representation.
            history = reader(row["trajectory_id"], *row["history_interval"])
            assert tensor_hash(history) == row["history_sha256"]
            normalized = (history - norm["mean"].cpu()) / norm["std"].cpu()
            assert torch.equal(normalized, item["H"])
            noise, ids = batch_noise(item["key"], self.noise_files[row["split"]])
            assert noise.dtype == torch.float32
            for member, rec in zip(MEMBERS, ids):
                assert rec["member_id"] == member
                assert rec["noise_tensor_hash"] == self.noise_hashes[row["split"], row["trajectory_id"], member]
            inputs[key] = {arm: condition_inputs(item, self.scales, noise, arm) for arm in ARMS}
            assert all(torch.equal(inputs[key]["C0"][f], inputs[key]["C1"][f]) for f in ("history", "noise"))
            # Target is committed only after the publication input exists.
            target = item["target"]
            assert tensor_hash(target) == row["target_sha256"]
            assert torch.equal(target, reader(row["trajectory_id"], *row["target_interval"]))
            targets[key] = target.cuda()
            if key == records[0]["cache_key"]:
                poisoned = dict(item, target=target.flip(1) + 10.)
                assert not torch.equal(poisoned["target"], target)
                poisoned_inputs = condition_inputs(poisoned, self.scales, noise, "C1")
                assert input_hashes(poisoned_inputs) == input_hashes(inputs[key]["C1"])
            details.append(dict(cache_key=key, normalized_history_sha256=tensor_hash(normalized),
                                noise_sha256=tensor_hash(noise), noise_identities=ids,
                                input_sha256={arm: input_hashes(data) for arm, data in inputs[key].items()}))
        return inputs, targets, details, poisoned_inputs

    def load_development_paths(self, keys):
        result = {key: {} for key in keys}
        with (OUT / "DEVELOPMENT.csv").open() as stream:
            scores = {(r["arm"], int(r["step"]), r["trajectory_id"], r["channel"]): r
                      for r in csv.DictReader(stream) if r["start"] == "420" and r["step"] in ("0", "250")}
        for arm, step in (("native", 0), ("C0", 250), ("C1", 250)):
            for row in self.load_cache(f"development_{arm}_{step}_paths.pt"):
                key = row["key"]
                if key in result:
                    assert arm not in result[key]
                    paths = row["paths"]
                    assert paths.shape == (32, 2, 20, 32, 64) and paths.dtype == torch.float32
                    sha = tensor_hash(paths)
                    assert sha == row["sha256"] == scores[arm, step, key.split("__")[1], "overall"]["output_sha256"]
                    result[key][arm] = paths
        assert all(set(paths) == {"native", *ARMS} for paths in result.values())
        return result, scores


def input_hashes(inputs):
    return {name: tensor_hash(value) for name, value in inputs.items()}
