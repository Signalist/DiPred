"""Verify, train, select on development CRPS, then evaluate both test periods."""
import argparse
import hashlib
import os
import shlex
import sys
import time

import torch

from research.corrector.run import dump, sha, table
from research.corrector_window_study.data import (LOCAL, RESULTS, StudyData, read_json,
                                                  tensor_sha, verify_weights)
from research.corrector_window_study.metrics import NAMES, save_results, scores, seed_trajectory_rows
from research.corrector_window_study.model import (ARMS, MEMBERS, SLOTS, WindowCorrector,
                                                   empirical_crps, inputs, objective,
                                                   training_normalization)
from research.corrector_window_study.verify import verify

STEPS = (0, 250, 500, 750, 1000)
TRAIN_SEEDS = (42, 43)
BATCH = 2
LR = 1e-3


def new_model(device):
    return WindowCorrector().to(device=device, memory_format=torch.channels_last)


def physical_forecast(model, release, arm, normalization):
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        correction = model(*inputs(release, arm, normalization))
    scale = normalization[1]
    return release[2] + correction.float() * scale[:, None]


def batches(dataset):
    for start in range(0, len(dataset.cases), BATCH):
        yield list(range(start, min(start+BATCH, len(dataset.cases))))


@torch.inference_mode()
def development_crps(model, data, arm, normalization):
    model.eval()
    values = []
    scale = normalization[1]
    for idx in batches(data):
        release = data.release(idx)
        prediction = physical_forecast(model, release, arm, normalization)
        target = data.target(idx)
        score = empirical_crps((prediction - target[:, None]) / scale[:, None],
                               torch.zeros_like(target))
        values.append(score.mean((1, 2, 3, 4)))
    return float(torch.cat(values).double().mean())


def atomic_checkpoint(path, content):
    tmp = path.with_suffix(".tmp")
    torch.save(content, tmp)
    os.replace(tmp, path)


def model_hash(model):
    fingerprints = "".join(tensor_sha(v.detach()) for v in model.state_dict().values())
    return hashlib.sha256(fingerprints.encode()).hexdigest()


def fit(arm, seed, data, dev, normalization, local, schedule, config_sha):
    torch.manual_seed(seed)
    model = new_model(data.observed.device)
    init_sha = model_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scale = normalization[1]
    log = []
    buffer = []
    timer = time.monotonic()
    initial_step = 0

    for step in STEPS:
        path = local / f"{arm}_{seed}_{step}.pt"
        if not path.exists():
            break
        ck = torch.load(path, weights_only=False, map_location=data.observed.device)
        assert (ck["arm"], ck["seed"], ck["step"], ck["init_sha"], ck["config_sha"]) == (
            arm, seed, step, init_sha, config_sha)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        log.append(dict(ck["row"], checkpoint_sha256=sha(path)))
        initial_step = step + 1

    if initial_step > 0:
        print(f"REUSE {arm} seed={seed} through step={initial_step-1}", flush=True)
    for step in range(initial_step, STEPS[-1] + 1):
        if step:
            model.train()
            idx = schedule[step-1]
            release = data.release(idx)
            prediction = physical_forecast(model, release, arm, normalization)
            target = data.target(idx)
            loss = objective((prediction - target[:, None]) / scale[:, None],
                             torch.zeros_like(target), ARMS[arm][3])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            buffer.append(loss.detach())
        if step in STEPS:
            dev_crps = development_crps(model, dev, arm, normalization)
            train_loss = float(torch.stack(buffer).mean()) if buffer else None
            assert torch.isfinite(torch.tensor(dev_crps))
            row = dict(arm=arm, seed=seed, step=step, development_crps=dev_crps,
                       training_loss=train_loss, initialization_sha256=init_sha)
            path = local / f"{arm}_{seed}_{step}.pt"
            atomic_checkpoint(path, dict(arm=arm, seed=seed, step=step, init_sha=init_sha,
                                         config_sha=config_sha, row=row, model=model.state_dict(),
                                         optimizer=optimizer.state_dict()))
            log.append(dict(row, checkpoint_sha256=sha(path)))
            buffer.clear()
            print(f"FIT {arm} seed={seed} step={step} dev_crps={dev_crps:.9f} "
                  f"train_loss={train_loss} elapsed={time.monotonic()-timer:.1f}s", flush=True)

    assert [row["step"] for row in log] == list(STEPS)
    best = min(log, key=lambda row: (row["development_crps"], row["step"]))
    print(f"SELECT {arm} seed={seed} step={best['step']}", flush=True)
    return log, best


def load_checkpoint(data, local, config):
    arm = config["arm"]
    path = local / f"{arm}_{config['seed']}_{config['step']}.pt"
    assert sha(path) == config["checkpoint_sha256"]
    ck = torch.load(path, weights_only=False, map_location=data.observed.device)
    assert ck["row"]["development_crps"] == config["development_crps"]
    model = new_model(data.observed.device).eval().requires_grad_(False)
    model.load_state_dict(ck["model"])
    return model


@torch.inference_mode()
def score_cases(data, normalization, arm, model):
    case_scores = []
    for idx in batches(data):
        release = data.release(idx)
        predicted = (physical_forecast(model, release, arm, normalization)
                     if model is not None else release[2])
        metrics, channels, leads = scores(predicted, data.target(idx), normalization[1])
        case_scores.extend(({k: v[b] for k, v in metrics.items()}, channels[b], leads[b])
                           for b in range(len(idx)))
    return case_scores


def development_diagnostics(dev, normalization, training_log, local):
    # Preserve probability and point scores for checkpoints that did not win selection.
    rows = []
    for config in training_log:
        arm = config["arm"]
        model = load_checkpoint(dev, local, config)
        case_scores = score_cases(dev, normalization, arm, model)
        values = {metric: sum(row[metric] for row, _, _ in case_scores) / len(case_scores)
                  for metric in NAMES}
        assert abs(values["crps"] - config["development_crps"]) < 1e-8
        rows.append(dict(arm=arm, seed=config["seed"], step=config["step"], **values))
    print(f"DEV DIAGNOSTICS {len(rows)} checkpoints, used only to describe training", flush=True)
    return rows


def evaluate(test, normalization, selected, local):
    records = {}
    seed_rows = []
    for config in [dict(arm="native")] + selected:
        arm = config["arm"]
        model = None if arm == "native" else load_checkpoint(test, local, config)
        case_scores = score_cases(test, normalization, arm, model)
        records.setdefault(arm, []).extend(case_scores)
        seed_rows.extend(seed_trajectory_rows(case_scores, test.cases, arm, config.get("seed", "")))
        print(f"EVAL {arm} seed={config.get('seed', '')} cases={len(case_scores)}", flush=True)
    return records, seed_rows


def main():
    import warnings
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=type(LOCAL), default=LOCAL)
    args = parser.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    device = torch.device("cuda")
    assert torch.cuda.is_bf16_supported()
    print("COMMAND", shlex.join(["python", *sys.argv]), torch.__version__,
          torch.cuda.get_device_name(), flush=True)
    cache_path = RESULTS / "cache_manifest.json"
    manifest = read_json(cache_path)
    verify_weights(manifest["weights"])
    data = StudyData(args.local, manifest, "train", device)
    dev = StudyData(args.local, manifest, "development", device)
    normalization = training_normalization(data)
    preflight = verify(data, manifest, normalization)
    print("VERIFIED", preflight, flush=True)
    dump(RESULTS / "verification.json", preflight)
    schedules = {seed: torch.randint(len(data.cases), (STEPS[-1], BATCH),
                                     generator=torch.Generator().manual_seed(seed))
                 for seed in TRAIN_SEEDS}
    config = dict(arms=ARMS, slots=dict(SLOTS), training_seeds=list(TRAIN_SEEDS), steps=list(STEPS),
                  batch_size=BATCH, members=MEMBERS, adam_lr=LR, optimizer="Adam defaults",
                  forward_precision="CUDA bfloat16 autocast; float32 loss and parameters",
                  normalization=dict(mean=normalization[0].flatten().tolist(),
                                     scale=normalization[1].flatten().tolist()),
                  schedule_sha256={seed: tensor_sha(schedule) for seed, schedule in schedules.items()},
                  cache_manifest_sha256=sha(cache_path),
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name())
    dump(RESULTS / "config.json", config)
    config_sha = sha(RESULTS / "config.json")
    training_log, selected = [], []
    for seed, schedule in schedules.items():
        for arm in ARMS:
            log, best = fit(arm, seed, data, dev, normalization, args.local, schedule.tolist(), config_sha)
            training_log.extend(log)
            selected.append(best)
            table(RESULTS / "training.csv", training_log)
        initials = [row for row in training_log if row["seed"] == seed and row["step"] == 0]
        assert len(initials) == 6
        assert len({row["initialization_sha256"] for row in initials}) == 1
        assert len({row["development_crps"] for row in initials}) == 1

    assert len(selected) == 12 and len(training_log) == 60
    dump(RESULTS / "selected.json", selected)
    table(RESULTS / "development_checkpoints.csv",
          development_diagnostics(dev, normalization, training_log, args.local))
    del data, dev
    test = StudyData(args.local, manifest, "test", device)
    records, seed_rows = evaluate(test, normalization, selected, args.local)
    save_results(RESULTS, records, seed_rows, test.cases)
    verify_weights(manifest["weights"])
    print("PASS: cache identity, preflight, 12 formal fits, development selection, test scoring", flush=True)


if __name__ == "__main__":
    main()
