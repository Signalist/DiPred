"""Frozen phase3 validation. No optimizer, parameter update, or model selection."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.residual_dynamics_phase3 import run as phase3
from research.residual_dynamics.run import raw, load_backbone
from common import WORK, DATA, EXTRA, digest, dump, csvwrite
from causal import BiasHead, inputs
from mechanisms import simple_predict
from metrics import empirical_crps, energy_score, consecutive_event
from reliability.backbone import configure

OUT = ROOT / "results/residual_dynamics_phase3_followup"
OLD = ROOT / "results/residual_dynamics_phase3"
CONFIG = json.loads((OUT / "protocol.json").read_text())
PROTOCOL_HASH = "98a6dfad9c88882856ae20e7222ad1e9f9b4aee38d8ed43ac12a05f6342945e3"
assert digest(OUT / "protocol.json") == PROTOCOL_HASH
RECORDS = CONFIG["records"]
METRICS = ("crps", "mse", "energy", "brier")


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def event(action, **fields):
    entry = dict(action=action, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields)
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def check_inputs():
    assert digest(OUT / "protocol.json") == PROTOCOL_HASH
    for name, expected in CONFIG["dependencies"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in CONFIG["models"].items():
        assert digest(OLD / "weights" / name) == expected, name
    assert digest(ROOT / "reliability/checkpoints/current.pt") == CONFIG["frozen"]["packaged_checkpoint_sha256"]
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def load_head(kind, seed):
    checkpoint = torch.load(OLD / "weights" / f"{kind}_{seed}.pt", map_location="cuda", weights_only=False)
    selected = next(row for row in read_csv(OLD / "selection.csv")
                    if row["kind"] == kind and int(row["seed"]) == seed)
    expected_step = 1000 if kind == "crps" else 0
    assert checkpoint["step"] == int(selected["step"]) == expected_step
    assert int(selected["strength"]) == int(kind == "crps")
    assert checkpoint["kind"] == kind and checkpoint["scale"] > 0
    model = BiasHead().cuda().eval().requires_grad_(False)
    model.load_state_dict(checkpoint["selected"], strict=True)
    return model, checkpoint["scale"]


@torch.no_grad()
def predict_bias(history, paths, norm, model, scale):
    features = inputs(history, paths, None, "current_fields", norm)
    values = torch.cat([model(batch) * scale for batch in features.split(CONFIG["head_batch"])])
    return values.reshape(len(paths), 20, 2, 32, 64).transpose(1, 2)


def compare_rows(actual, expected, keys, metrics, label):
    index = {tuple(str(row[key]) for key in keys): row for row in expected}
    assert len(index) == len(expected) == len(actual), (label, len(actual), len(expected))
    maximum = dict.fromkeys(metrics, 0.)
    for row in actual:
        other = index[tuple(str(row[key]) for key in keys)]
        for metric in metrics:
            error = abs(float(row[metric]) - float(other[metric]))
            maximum[metric] = max(maximum[metric], error)
            tolerance = CONFIG["tolerances"]["reproduction"][metric]
            tolerance += CONFIG["tolerances"]["reproduction_rtol"] * abs(float(other[metric]))
            assert error <= tolerance, (label, metric, row, other, error)
    return dict(label=label, rows=len(actual), maximum_abs_error=maximum)


@torch.no_grad()
def reproduce():
    """Use the exact phase3 chunking, tensor layouts, and full scoring function."""
    checks = []
    phase3.blocks.clear()
    phase3.long.clear()
    zero_checks = []
    for split in ("development", "test"):
        data = phase3.load(split)
        phase3.score(data, torch.zeros_like(data.b), "native")
        for kind in ("raw", "std", "crps"):
            for seed in CONFIG["head_seeds"]:
                model, scale = load_head(kind, seed)
                bias = phase3.output(model, data, scale, kind)
                if kind != "crps":
                    assert torch.count_nonzero(bias) == 0
                    zero_checks.append(dict(split=split, kind=kind, seed=seed, max_abs=float(bias.abs().max())))
                phase3.score(data, bias, kind, seed)
        del data
    selected_methods = {"native", "raw", "std", "crps"}
    expected = [row for row in read_csv(OLD / "scores.csv") if row["method"] in selected_methods]
    checks.append(compare_rows(phase3.blocks, expected, ["split", "trajectory", "start", "method", "seed"],
                               METRICS, "all original block scores"))
    expected_detail = [row for row in read_csv(OLD / "lead_channel.csv") if row["method"] in selected_methods]
    checks.append(compare_rows(phase3.long, expected_detail,
                               ["split", "trajectory", "start", "method", "seed", "lead", "ch"],
                               ("crps", "mse"), "all original lead/channel scores"))
    from common import aggregate as group
    for name, keys in (("per_seed", ["split", "method", "seed"]),
                       ("per_trajectory", ["split", "method", "trajectory"])):
        actual = group(phase3.blocks, keys, METRICS)
        expected = [row for row in read_csv(OLD / f"{name}.csv") if row["method"] in selected_methods]
        checks.append(compare_rows(actual, expected, keys, METRICS, name))
        csvwrite(OUT / f"reproduction_{name}.csv", actual)
    # Full original batch and member order: independently replay from source history.
    model, norm, frozen = load_backbone(SimpleNamespace(workspace=WORK))
    assert frozen == CONFIG["frozen"]
    history = raw(DATA, RECORDS["test"], 400)
    stored = torch.load(EXTRA / "test_400_native.pt", map_location="cuda", weights_only=False)["paths"]
    errors = []
    for index, seed in enumerate(CONFIG["member_seeds"]):
        value = model.rollout((history - norm["mean"]) / norm["std"], seed, sign=1)
        value = value * norm["std"] + norm["mean"]
        error = float((value - stored[:, index]).abs().max())
        assert error <= CONFIG["tolerances"]["cache_replay"], (seed, error)
        errors.append(error)
    csvwrite(OUT / "reproduction_scores.csv", phase3.blocks)
    csvwrite(OUT / "reproduction_lead_channel.csv", phase3.long)
    dump(OUT / "reproduction.json", dict(passed=True, protocol_sha256=PROTOCOL_HASH, checks=checks,
                                        zero_selected=zero_checks, native_replay_errors=errors,
                                        native_replay_history_sha256=tensor_hash(history)))
    print("PASS old windows, per-seed/per-trajectory/lead/channel scores, zero selected, all32 native replay", flush=True)


def score_window(paths, truth, norm, start, seed, method):
    rows, detail = [], []
    # Match the phase3 memory layout for double precision scoring.
    std = norm["std"].reshape(2, 1, 1)
    for record, field, target in zip(RECORDS["test"], paths, truth):
        pred = field.transpose(1, 2).double()
        target = target.transpose(0, 1).double()
        crps = empirical_crps(pred / std, target / std).mean((-1, -2))
        mse = (pred.mean(0) - target).square().mean((-1, -2))
        observed = int(consecutive_event(target.transpose(0, 1)[None], CONFIG["event_threshold"])[0])
        probability = float(consecutive_event(pred.transpose(1, 2), CONFIG["event_threshold"]).double().mean())
        metadata = dict(trajectory=record["file"], Re=record["group"], start=start, seed=seed, method=method)
        rows.append(dict(**metadata, crps=float(crps.mean()), mse=float(mse.mean()),
                         energy=float(energy_score(pred / std, target / std)), event=observed,
                         probability=probability, brier=(probability - observed) ** 2))
        for lead in range(20):
            for channel, name in enumerate("uv"):
                detail.append(dict(**metadata, lead=lead + 1, channel=name,
                                   crps=float(crps[lead, channel]), mse=float(mse[lead, channel])))
    return rows, detail


def shift_invariance(paths, corrected):
    """Check all pair differences and centered fields, plus covariance bounds.

    Equal centered member fields imply equal covariance at every pair of field
    coordinates, including all cross-lead, cross-channel and spatial entries.
    Avoid allocating the 81920 by 81920 feature covariance matrix.
    """
    native = paths.double().flatten(1)
    shifted = corrected.double().flatten(1)
    difference = shifted - native
    pair_error = float((difference.amax(0) - difference.amin(0)).abs().max())
    first = native - native.mean(0)
    second = shifted - shifted.mean(0)
    centered_error = float((first - second).abs().max())
    width_error = float((native.std(0) - shifted.std(0)).abs().max())
    gram_error = float(((first @ first.T - second @ second.T) / first.shape[1]).abs().max())
    # All 40 channel/lead pairs, averaging over space and members.
    first40 = first.reshape(32, 40, -1).permute(1, 0, 2).flatten(1)
    second40 = second.reshape(32, 40, -1).permute(1, 0, 2).flatten(1)
    covariance_error = float(((first40 @ first40.T - second40 @ second40.T) / (31 * 32 * 64)).abs().max())
    covariance_bound = 32 / 31 * (2 * float(first.abs().max()) * centered_error + centered_error ** 2)
    assert max(pair_error, centered_error, width_error) <= CONFIG["tolerances"]["shift_fields"]
    assert max(gram_error, covariance_error, covariance_bound) <= CONFIG["tolerances"]["covariance"]
    return dict(pair_max_abs=pair_error, centered_max_abs=centered_error, width_max_abs=width_error,
                member_gram_max_abs=gram_error, channel_lead_covariance_max_abs=covariance_error,
                all_feature_covariance_error_bound=covariance_bound)


@torch.no_grad()
def extend():
    reproduction = json.loads((OUT / "reproduction.json").read_text())
    assert reproduction["passed"] and reproduction["protocol_sha256"] == PROTOCOL_HASH
    model, norm, frozen = load_backbone(SimpleNamespace(workspace=WORK))
    assert frozen == CONFIG["frozen"]
    heads = {seed: load_head("crps", seed) for seed in CONFIG["head_seeds"]}
    cache = OUT / "cache"
    cache.mkdir(exist_ok=True)
    source_hashes = {row["file"]: digest(DATA / row["file"]) for row in RECORDS["test"]}
    dump(OUT / "data_hashes.json", source_hashes)
    history = raw(DATA, RECORDS["test"], CONFIG["starts"][0])
    rows, detail, invariants, manifest = [], [], [], []
    started = time.monotonic()
    for start in CONFIG["starts"]:
        assert time.monotonic() - started < CONFIG["budget"]["max_seconds"], "fixed resource budget exhausted"
        path = cache / f"issued_{start}.pt"
        metadata_path = cache / f"issued_{start}.json"
        identity = dict(start=start, history_indices=[start, start + 19], target_indices=[start + 20, start + 39],
                        trajectories=[row["file"] for row in RECORDS["test"]],
                        member_seeds=CONFIG["member_seeds"], protocol_sha256=PROTOCOL_HASH,
                        history_sha256=tensor_hash(history), source_hashes=source_hashes)
        if path.exists() and metadata_path.exists():
            saved = json.loads(metadata_path.read_text())
            assert saved["identity"] == identity and digest(path) == saved["sha256"]
            bundle = torch.load(path, map_location="cuda", weights_only=False)
            assert torch.equal(bundle["history"], history)
            paths, biases = bundle["paths"], bundle["biases"]
            for seed, (head, scale) in heads.items():
                assert torch.equal(biases[seed], predict_bias(history, paths, norm, head, scale))
            event("reuse_issued_without_current_truth", start=start, sha256=saved["sha256"])
        else:
            assert not path.exists() and not metadata_path.exists(), "incomplete issued cache requires diagnosis"
            normalized = (history - norm["mean"]) / norm["std"]
            paths = torch.stack([model.rollout(normalized, seed, sign=1) * norm["std"] + norm["mean"]
                                 for seed in CONFIG["member_seeds"]], 1)
            biases = {seed: predict_bias(history, paths, norm, head, scale)
                      for seed, (head, scale) in heads.items()}
            torch.save(dict(history=history.cpu(), paths=paths.cpu(),
                            biases={seed: value.cpu() for seed, value in biases.items()},
                            identity=identity), path)
            saved = dict(identity=identity, sha256=digest(path))
            dump(metadata_path, saved)
            event("issued_all32_and_all3_heads_before_truth", start=start, sha256=saved["sha256"])
        # Construct every corrected complete20 path before opening current truth.
        issued = {seed: paths - bias[:, None] for seed, bias in biases.items()}
        truth = raw(DATA, RECORDS["test"], start + 20)
        event("revealed_complete20", start=start, truth_sha256=tensor_hash(truth))
        manifest.append(dict(**saved, truth_sha256=tensor_hash(truth), paths_sha256=tensor_hash(paths)))
        for seed, value in [(0, paths), *issued.items()]:
            scores, cells = score_window(value, truth, norm, start, seed, "native" if seed == 0 else "crps")
            rows.extend(scores)
            detail.extend(cells)
            if seed:
                for index, record in enumerate(RECORDS["test"]):
                    invariants.append(dict(start=start, seed=seed, trajectory=record["file"],
                                           **shift_invariance(paths[index], value[index])))
        history = truth
        csvwrite(OUT / "scores.csv", rows)
        csvwrite(OUT / "lead_channel.csv", detail)
        csvwrite(OUT / "shift_invariance.csv", invariants)
        dump(OUT / "cache_manifest.json", manifest)
        print("extension", start, "completed", len(manifest), "/20", "seconds", round(time.monotonic() - started, 1), flush=True)
    dump(OUT / "extension_complete.json", dict(windows=len(manifest), blocks=12 * len(manifest),
                                              scores=len(rows), detail=len(detail), protocol_sha256=PROTOCOL_HASH,
                                              seconds=time.monotonic() - started))


def operator():
    spec = json.loads((OLD / "coefficients.json").read_text())["linear"]
    spec["coef"] = torch.tensor(spec["coef"], dtype=torch.float64, device="cuda")
    return spec


def decompose(bias, spec):
    """bias [trajectory, lead, channel, y, x], all arithmetic float64."""
    bias = bias.double()
    inherited = [bias[:, 0]]
    accumulated = [torch.zeros_like(bias[:, 0])]
    innovations = []
    for lead in range(1, bias.shape[1]):
        innovation = bias[:, lead] - simple_predict(bias[:, lead - 1], None, spec)
        innovations.append(innovation)
        inherited.append(simple_predict(inherited[-1], None, spec))
        accumulated.append(simple_predict(accumulated[-1], None, spec) + innovation)
    return torch.stack(inherited, 1), torch.stack(accumulated, 1), torch.stack(innovations, 1)


def correlation(first, second):
    first, second = first.flatten(), second.flatten()
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    return float((first * second).sum() / denominator) if float(denominator) > 0 else float("nan")


@torch.no_grad()
def diagnose():
    spec = operator()
    training = phase3.load("train")
    _, _, train_innovation = decompose(training.b, spec)
    fixed_mean = train_innovation.mean(0)
    torch.save(dict(mean=train_innovation.mean(0).cpu(), trajectories=RECORDS["train"],
                    starts=[400, 420], operator_sha256=digest(OLD / "coefficients.json")),
               OUT / "training_innovation_mean.pt")
    del training, train_innovation
    decompositions, correlations, audits = [], [], []

    def analyze(bias, split, starts, records):
        inherited, accumulated, innovations = decompose(bias, spec)
        total = bias.double()
        field_error = float((total - inherited - accumulated).abs().max())
        assert field_error <= CONFIG["tolerances"]["decomposition_field"]
        terms = dict(inherited=inherited.square().mean((-1, -2)),
                     accumulated=accumulated.square().mean((-1, -2)),
                     cross=(2 * inherited * accumulated).mean((-1, -2)),
                     total=total.square().mean((-1, -2)))
        energy_error = float((terms["total"] - terms["inherited"] - terms["accumulated"] - terms["cross"]).abs().max())
        assert energy_error <= CONFIG["tolerances"]["decomposition_energy"]
        audits.append(dict(split=split, starts=starts, field_error=field_error, energy_error=energy_error))
        for index, innovation in enumerate(innovations):
            record = records[index % len(records)]
            start = starts[index // len(records)]
            meta = dict(split=split, trajectory=record["file"], Re=record["group"], start=start)
            for lead in range(20):
                for channel, name in enumerate("uv"):
                    decompositions.append(dict(**meta, lead=lead + 1, channel=name,
                                               **{key: float(value[index, lead, channel]) for key, value in terms.items()}))
            for channel, name in enumerate("uv"):
                centered = innovation - fixed_mean
                for lead in [0, *range(2, 20)]:
                    # lead0 pools all adjacent innovation pairs; lead h pairs r_h,r_(h+1).
                    first = slice(None, -1) if lead == 0 else lead - 2
                    second = slice(1, None) if lead == 0 else lead - 1
                    correlations.append(dict(**meta, channel=name, lead=lead,
                        raw=correlation(innovation[first, channel], innovation[second, channel]),
                        train_centered=correlation(centered[first, channel], centered[second, channel])))
    for split in ("development", "test"):
        data = phase3.load(split)
        analyze(data.b, split, [400, 420], RECORDS[split])
        del data
    for saved in json.loads((OUT / "cache_manifest.json").read_text()):
        start = saved["identity"]["start"]
        path = OUT / "cache" / f"issued_{start}.pt"
        assert digest(path) == saved["sha256"]
        bundle = torch.load(path, map_location="cuda", weights_only=False)
        truth = raw(DATA, RECORDS["test"], start + 20)
        assert tensor_hash(truth) == saved["truth_sha256"]
        bias = (bundle["paths"].mean(1) - truth).transpose(1, 2)
        analyze(bias, "extension", [start], RECORDS["test"])
    csvwrite(OUT / "decomposition.csv", decompositions)
    csvwrite(OUT / "innovation_correlations.csv", correlations)
    dump(OUT / "decomposition_audit.json", dict(checks=audits,
        train_mean_sha256=digest(OUT / "training_innovation_mean.pt"),
        protocol_sha256=PROTOCOL_HASH, coefficients_sha256=digest(OLD / "coefficients.json")))
    print("PASS decomposition", len(decompositions), "cells", len(correlations), "correlations", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["reproduce", "extend", "diagnose", "all"])
    args = parser.parse_args()
    configure()
    check_inputs()
    dump(OUT / "environment.json", dict(torch=torch.__version__, cuda=torch.version.cuda,
         gpu=torch.cuda.get_device_name(), cudnn=torch.backends.cudnn.version(),
         protocol_sha256=PROTOCOL_HASH, threads=torch.get_num_threads(),
         tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
         deterministic_cudnn=torch.backends.cudnn.deterministic))
    stages = [reproduce, extend, diagnose] if args.stage == "all" else [globals()[args.stage]]
    for stage in stages:
        stage()


if __name__ == "__main__":
    main()
