"""Import immutable native ensembles. Build release inputs separately from labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from reliability.backbone import configure
from research.corrector.run import dump, sha
from research.residual_dynamics.run import load_backbone

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/corrector_window_study"
LOCAL = ROOT / ".cache/corrector_window_study"
PRIOR = ROOT / "results/residual_dynamics"
STARTS = (400, 420, 440, 460)
NATIVE_STARTS = (380,) + STARTS
SEEDS = list(range(500000, 500032))
WORKSPACE = ROOT.parent / "Forecast/Runtime"
RAW = ROOT.parents[1] / "OLD/REALPDE/realpdebench-competition/data/train_real"


def read_json(path):
    return json.loads(path.read_text())


def tensor_sha(tensor):
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def save_tensor(path, tensor):
    staged = path.with_suffix(".tmp")
    torch.save(tensor.cpu(), staged)
    os.replace(staged, path)


def read_physics(records, data):
    physics = []
    for record in records:
        with h5py.File(data / record["file"], "r") as handle:
            assert all(handle[c].shape[0] == record["frames"] for c in ("u", "v"))
            block = np.stack([handle[c][380:500, ::2, ::2] for c in ("u", "v")]).astype("float32")
        assert block.shape[::2] == (2, 32) and block.shape[-1] == 64
        assert block.shape[1] >= 100 and np.isfinite(block).all()
        physics.append(torch.from_numpy(block))
    return physics


def windows(physics, start):
    index = start - 380
    return torch.stack([block[:, index:index + 20] for block in physics])


def case_manifest(records, physics):
    cases = []
    for start in STARTS:
        for index, (record, block) in enumerate(zip(records, physics)):
            if block.shape[1] >= start + 40 - 380:
                cases.append(dict(index=index, file=record["file"], re=record["re"],
                                  start=start, earlier=[start - 20, start],
                                  history=[start, start + 20], target=[start + 20, start + 40]))
    return cases


def restore_old(path, entry, identity, history, truth):
    assert sha(path) == entry["sha256"], path
    archive = torch.load(path, map_location="cpu", weights_only=True)
    assert archive["identity"] == entry["identity"] == identity
    assert tensor_sha(history) == entry["history_sha256"] == archive["history_sha256"]
    assert tensor_sha(truth) == entry["truth_sha256"] == archive["truth_sha256"]
    new_residual = -archive["residual"]
    native = truth[:, None] - new_residual
    torch.testing.assert_close(truth[:, None].double() - native.double(),
                               new_residual.double(), rtol=0, atol=1.2e-7)
    return native


@torch.inference_mode()
def prepare(args):
    configure()
    protocol = read_json(PRIOR / "protocol.json")
    previous = read_json(ROOT / "results/corrector_rollout/verification.json")
    frozen = protocol["frozen"]
    weights = previous["checkpoint_sha256_after"]
    verify_weights(weights)
    model, normalization, identity = load_backbone(args)
    assert identity == frozen and not any(p.requires_grad for p in model.parameters())
    assert protocol["starts"] == [400, 420] and protocol["member_seeds"] == SEEDS
    old_entries = {(e["identity"]["split"], e["identity"]["start"]): e
                   for e in read_json(PRIOR / "cache_manifest.json")}
    local = args.local
    local.mkdir(parents=True, exist_ok=True)
    manifest = dict(starts=list(STARTS), native_starts=list(NATIVE_STARTS),
                    weights=weights, native_checkpoint=frozen, seeds=SEEDS,
                    normalization_sha256={k: tensor_sha(v) for k, v in normalization.items()},
                    protocol_sha256=sha(PRIOR / "protocol.json"),
                    legacy_manifest_sha256=sha(PRIOR / "cache_manifest.json"), splits={})
    manifest_path = RESULTS / "cache_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        checkpoints = [local / f"{split}_{start}.pt"
                       for split in existing["splits"] for start in NATIVE_STARTS]
        inputs = [local / f"{split}_{name}.pt" for split in existing["splits"]
                  for name in ("labels", "observed")]
        if all(p.is_file() for p in checkpoints + inputs):
            for split, row in existing["splits"].items():
                for start, native in row["native"].items():
                    assert sha(local / f"{split}_{start}.pt") == native["sha256"]
                for name in ("labels", "observed"):
                    assert sha(local / f"{split}_{name}.pt") == row[f"{name}_sha256"]
            print("REUSE verified complete cache", flush=True)
            return
    for split, records in protocol["records"].items():
        physics = read_physics(records, args.data)
        cases = case_manifest(records, physics)
        labels = torch.stack([physics[c["index"]][:, c["start"]-360:c["start"]-340] for c in cases])
        # Observed fields are stored with a padding mask implicit in the case list.
        observed = torch.stack([torch.nn.functional.pad(p, (0, 0, 0, 0, 0, 120-p.shape[1]))
                                for p in physics])
        save_tensor(local / f"{split}_labels.pt", labels)
        save_tensor(local / f"{split}_observed.pt", observed)
        entry = dict(cases=cases, files=[r["file"] for r in records], native={},
                     labels_sha256=sha(local / f"{split}_labels.pt"),
                     observed_sha256=sha(local / f"{split}_observed.pt"))
        manifest["splits"][split] = entry
        for start in NATIVE_STARTS:
            source_identity = dict(split=split, start=start, files=entry["files"],
                                   seeds=SEEDS, frozen=frozen)
            history = windows(physics, start)
            dest = local / f"{split}_{start}.pt"
            resume = local / f"{split}_{start}.json"
            provenance = dict(identity=source_identity, history_sha256=tensor_sha(history))
            if start in (400, 420):
                source = args.legacy / f"{split}_{start}.pt"
                provenance["legacy_sha256"] = old_entries[split, start]["sha256"]
                provenance["truth_sha256"] = tensor_sha(windows(physics, start + 20))
            if resume.exists() and dest.exists():
                record = read_json(resume)
                assert record == dict(**provenance, sha256=sha(dest)), dest
                print("REUSE completed", split, start, flush=True)
            else:
                if start in (400, 420):
                    native = restore_old(source, old_entries[split, start], source_identity,
                                         history, windows(physics, start + 20))
                    print("IMPORTED", split, start, flush=True)
                else:
                    norm_history = (history.cuda() - normalization["mean"]) / normalization["std"]
                    outputs = [(model.rollout(norm_history, seed, sign=1) * normalization["std"]
                                + normalization["mean"]).cpu() for seed in SEEDS]
                    native = torch.stack(outputs, dim=1)
                    print("GENERATED", split, start, flush=True)
                assert native.shape == (len(records), 32, 2, 20, 32, 64)
                assert torch.isfinite(native).all()
                save_tensor(dest, native)
                record = dict(**provenance, sha256=sha(dest))
                dump(resume, record)
            entry["native"][str(start)] = record
        print("CASES", split, len(cases), flush=True)
    assert manifest["splits"]["train"]["cases"]
    verify_weights(weights)
    if manifest_path.exists():
        assert manifest == read_json(manifest_path), "Recreated data must match the published manifest"
    else:
        dump(manifest_path, manifest)


def verify_weights(weights):
    for path, fingerprint in weights.items():
        assert sha(Path(path)) == fingerprint, path


class StudyData:
    """Hold label and release tensors separately. A case index is the only join key."""
    def __init__(self, local, manifest, split, device):
        identity = manifest["splits"][split]
        self.cases = identity["cases"]
        self.observed = self.load(local / f"{split}_observed.pt", identity["observed_sha256"], device)
        self.labels = self.load(local / f"{split}_labels.pt", identity["labels_sha256"], device)
        self.paths = {int(start): self.load(local / f"{split}_{start}.pt", row["sha256"], device)
                      for start, row in identity["native"].items()}
        self.means = {start: paths.mean(1) for start, paths in self.paths.items()}
        self.spreads = {start: self.paths[start].std(1, unbiased=False) for start in STARTS}

    @staticmethod
    def load(path, digest, device):
        assert sha(path) == digest, path
        return torch.load(path, map_location=device, weights_only=True).requires_grad_(False)

    def release(self, indices):
        releases = []
        for i in indices:
            case = self.cases[i]
            j, start = case["index"], case["start"]
            k = start - 380
            observed = self.observed[j]
            recent = observed[:, k:k+20]
            earlier = observed[:, k-20:k]
            memory = recent - self.means[start-20][j]
            releases.append((recent, earlier, self.paths[start][j], self.means[start][j],
                             self.spreads[start][j], memory))
        return tuple(torch.stack(values) for values in zip(*releases))

    def target(self, indices):
        return self.labels[list(indices)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--data", type=Path, default=RAW)
    parser.add_argument("--local", type=Path, default=LOCAL)
    parser.add_argument("--legacy", type=Path, default=WORKSPACE / "runs/residual_dynamics_20261119")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
