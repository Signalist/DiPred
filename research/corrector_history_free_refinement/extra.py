"""Independent continuation of a verified S3 checkpoint under pure MSE."""
from shutil import copyfile
from pathlib import Path
from time import perf_counter

import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.data import write_tensor
from research.corrector_residual_feedback.fit import STEPS, seed, state_digest
from research.corrector_residual_optimization.fit import checks, predict, track_weights
from research.corrector_spacetime_optimization.fit import config_hash, random_state, restore
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_window_study.data import ROOT, read_json, tensor_sha


def fit_extra(study, name, net, source, train, features, scale, schedule, dev=None, dev_features=None):
    """The caller supplies an independently loaded teacher and its source identity."""
    out = study.out
    device = train.p.device
    net.requires_grad_(True)
    seed()
    steps = 6000
    targets = (train.y - train.p) / scale
    config = dict(seed=42, updates=steps, steps=list(STEPS), lr=.0003, lr_end=1e-6, clip=1., batch=8,
                  weight_decay=1e-5, scale=scale, source=source, schedule=tensor_sha(schedule),
                  train=tensor_sha(features), target=tensor_sha(targets),
                  development=tensor_sha(dev_features) if dev_features is not None else None,
                  optimizer_reset=True, layout=source["layout"])
    digest = config_hash(config)
    path = out / "structured_fits" / f"{name}.json"
    journal = study.cache / f"{name}.pt"
    if path.exists():
        result = read_json(path)
        assert result["config"] == config
        for item in result["training"] + result["states"]:
            assert sha(out / item.get("checkpoint", item.get("path"))) == item["sha256"]
        return result

    optimizer = torch.optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps-1, eta_min=config["lr_end"])
    rows, measured, start, elapsed = [], [], 0, 0.
    if journal.exists():
        state = torch.load(journal, map_location="cpu", weights_only=False)
        assert state["config_hash"] == digest
        restore(net, optimizer, scheduler, state)
        start, rows, elapsed = (state[key] for key in ("step", "rows", "elapsed"))
        measured = [row.to(device) for row in state["stats"]]
    started = perf_counter()
    cohorts = [("train", train, features, track_weights(train))]
    if dev is not None:
        cohorts.append(("dev", dev, dev_features, track_weights(dev)))
    schedule_device = schedule.to(device)
    for step in range(start, steps+1):
        mark = step in STEPS
        if mark and (not rows or rows[-1]["step"] != step):
            checkpoint = out / "checkpoints" / f"{name}_{step}.pt"
            weights = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            if checkpoint.exists():
                assert state_digest(torch.load(checkpoint, map_location="cpu", weights_only=True)) == state_digest(weights)
            else:
                write_tensor(checkpoint, weights)
            stats = torch.stack(measured) if measured else None
            if stats is not None:
                assert torch.isfinite(stats).all()
            values = stats.mean(0).tolist() if stats is not None else [None]*4
            data = dict(run=name, step=step, checkpoint=str(checkpoint.relative_to(out)), sha256=sha(checkpoint),
                        lr=optimizer.param_groups[0]["lr"], phase="s3_extra_mse",
                        **dict(zip(("loss", "grad", "postclip", "clip_frequency"), values)))
            for label, cohort, design, weight in cohorts:
                output = predict(net, design, scale)
                assert torch.isfinite(output).all()
                scores, error = checks(output, cohort, weight)
                data.update({f"{label}_{key}": value for key, value in scores.items()})
                data[f"{label}_identity_error"] = error
            rows.append(data)
            measured.clear()
            print("STEP", name, step, data["train_mse"], data.get("dev_mse"), flush=True)
        if mark or step % 100 == 0:
            state = dict(model=net.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         rng=random_state(), step=step, rows=rows, config_hash=digest, schedule=config["schedule"],
                         elapsed=elapsed+perf_counter()-started, stats=torch.stack(measured).cpu() if measured else [])
            write_tensor(journal, state)
            if mark:
                snapshot = out / "training_states" / f"{name}_{step}.pt"
                if not snapshot.exists():
                    staging = snapshot.with_suffix(".tmp")
                    copyfile(journal, staging)
                    staging.replace(snapshot)
        if step == steps:
            break
        ix = schedule_device[step]
        net.train()
        optimizer.zero_grad(set_to_none=True)
        loss = (net(features[ix]) - targets[ix]).square().mean()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step()
        scheduler.step()
        measured.append(torch.stack((loss.detach(), grad.detach(), grad.detach().clamp_max(1.), (grad>1).float())))
    states = []
    for row in rows:
        snapshot = out / "training_states" / Path(row["checkpoint"]).name
        states.append(dict(step=row["step"], path=str(snapshot.relative_to(out)), sha256=sha(snapshot), borrowed=False))
    selected = min(rows, key=lambda row: (row["dev_mse"], row["step"])) if dev is not None else rows[-1]
    result = dict(config=config, config_hash=digest, directory=str(out.relative_to(ROOT)), training=rows,
                  states=states, final=rows[-1], selected=selected, elapsed_seconds=elapsed+perf_counter()-started)
    atomic_json(path, result)
    journal.unlink(missing_ok=True)
    print("FINISHED", name, result["elapsed_seconds"], flush=True)
    return result
