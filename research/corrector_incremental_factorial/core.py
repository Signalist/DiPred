"""Development-only factorial fitting with verified archived prefixes."""
from shutil import copyfile
from statistics import mean

import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.data import ROOT, guard
from research.corrector_residual_feedback.fit import state_digest
from research.corrector_spacetime_optimization.fit import config_hash
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_spacetime_validation.gradients import cosine, monitor_indices, vectors
from research.corrector_simple_baseline.fit import table
from research.corrector_window_study.data import read_json, tensor_sha
from research.corrector_incremental_structure.fit import Fits, open_head
from research.corrector_incremental_structure.model import inputs, labels
from research.corrector_incremental_structure.teachers import main_bundle

from .run import PREVIOUS

REGISTRATION = ("PROTOCOL.md", "hypotheses.json", "experiment_plan.json", "data_roles.json")
BASE = ("D0", "DB", "I0", "IB")
FACTORIAL = BASE + tuple(f"{name}_match2" for name in BASE)


def stamp(record, folder):
    return dict(record, directory=str(folder.relative_to(ROOT)))


def load_head(record, row, basis, device):
    return open_head(record, row, ROOT / record["directory"], basis, device)


def step_row(record, step):
    return next(r for r in record["training"] if r["step"] == step)


def verify_reuse(record, engine, audit, wanted, shared=None):
    cfg = record["config"]
    assert record["config_hash"] == config_hash(cfg)
    assert cfg["model"] == wanted and cfg["inputs"] == engine.identity
    assert cfg["scale"] == engine.train.scale and cfg["basis"] == tensor_sha(engine.basis)
    assert cfg["schedule"] == cfg["prefix_schedule"] == tensor_sha(engine.schedule)
    assert cfg["seed"] == 42 and cfg["layout"] == "NCDHW" and cfg["batch"] == 8
    assert cfg["updates"] == record["final"]["step"] == 6000
    assert (cfg["lr"], cfg["lr_end"], cfg["weight_decay"], cfg["clip"]) == (.001, 1e-6, 1e-5, 1.)
    if shared is not None:
        assert cfg["shared"] == sha(shared)
    else:
        assert cfg["first"] == 0 and cfg["shared"] is None
    parent = ROOT / record["directory"]
    for rows, field in ((record["training"], "checkpoint"), (record["states"], "path")):
        for row in rows:
            path = parent / row[field]
            assert sha(path) == row["sha256"] == audit["checkpoints"][str(path.relative_to(ROOT))]


def prefix(record, destination, through=1000):
    """Materialize the read-only copies expected by the existing trainer."""
    origin = ROOT / record["directory"]
    by_step = {r["step"]: r for r in record["states"]}
    for row in record["training"]:
        step = row["step"]
        if step > through:
            break
        state_row = by_step[step]
        for key, item in (("checkpoint", row), ("path", state_row)):
            source, target = origin / item[key], destination / item[key]
            if source != target and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                stage = target.with_suffix(".copy")
                copyfile(source, stage)
                stage.replace(target)
            assert sha(target) == item["sha256"], target
    shared = origin / by_step[through]["path"]
    state = torch.load(shared, map_location="cpu", weights_only=False)
    assert state["config_hash"] == record["config_hash"] and state["step"] == through
    assert state["scheduler"]["last_epoch"] == through
    assert set(state["rng"]) == {"python", "numpy", "torch", "cuda"}
    assert {int(x["step"]) for x in state["optimizer"]["state"].values()} == {through}
    row = step_row(record, through)
    assert state["rows"][-1] == row
    weight = torch.load(origin / row["checkpoint"], map_location="cpu", weights_only=True)
    assert state_digest(state["model"]) == state_digest(weight)
    return shared


def probes(engine, record, step, terms, batches):
    cfg = record["config"]["model"]
    head = load_head(record, step_row(record, step), engine.basis, engine.device)
    ids = sorted({i for batch in batches for i in batch})
    pos = {n: i for i, n in enumerate(ids)}
    remapped = [[pos[n] for n in batch] for batch in batches]
    return vectors(head, inputs(engine.train, ids, cfg), labels(engine.train, ids, cfg), remapped, terms)


def details(name, step, packets, weights):
    for batch, pack in enumerate(packets):
        main = pack["mse"][1]
        full = main + sum(pack[k][1] * w for k, w in weights.items())
        norm = float(full.norm())
        for term, (loss, grad, _) in pack.items():
            weight = weights.get(term, 1.)
            yield dict(arm=name, step=step, batch=batch, term=term, loss=loss, weight=weight,
                       norm=float(grad.norm()), mse_norm=float(main.norm()),
                       weighted_ratio=weight * float(grad.norm()) / float(main.norm()),
                       cosine_to_mse=cosine(grad, main), total_norm=norm,
                       clip_factor=min(1., 1. / (norm + 1e-6)))


def calibrate(packets, name, rho):
    return rho * mean(float(r["mse"][1].norm()) for r in packets) / mean(float(r[name][1].norm()) for r in packets)


def core(study):
    out = study.out
    for name in REGISTRATION:
        assert (out / name).is_file()
    plan = read_json(out / "experiment_plan.json")
    prior = guard(PREVIOUS)
    audit = read_json(PREVIOUS / "artifact_audit.json")
    assert audit["passed"] and audit["freeze_sha256"] == sha(PREVIOUS / "FROZEN.json")
    train, features, _ = study.get("train")
    x = main_bundle(study, train, features, "train")
    dev, features, _ = study.get("development")
    y = main_bundle(study, dev, features, "development")
    for split in ("train", "development"):
        assert sha(out / f"teacher_{split}.json") == prior["bindings"][f"teacher_{split}.json"]
    basis = torch.load(PREVIOUS / "temporal_basis.pt", map_location="cpu", weights_only=True)
    engine = Fits(out, study.cache, x, y, study.schedule, basis)
    old = {name: stamp(read_json(PREVIOUS / "structured_fits" / f"{name}.json"), PREVIOUS)
           for name in (*BASE, "IB_match2", "IB_match", "IB_energy", "IB_lag")}
    anchor = old["IB_match2"]["config"]["model"]["weights"]["match2"]
    assert anchor == prior["models"]["IB_match2"]["config"]["model"]["weights"]["match2"]
    for name, record in old.items():
        shared = PREVIOUS / "shared_IB.pt" if record["config"]["shared"] else None
        verify_reuse(record, engine, audit, prior["models"][name]["config"]["model"], shared)
    prefix_files = {name: prefix(old[name], out) for name in BASE}
    atomic_json(out / "reuse_audit.json", dict(
        previous_freeze=sha(PREVIOUS / "FROZEN.json"), previous_audit=sha(PREVIOUS / "artifact_audit.json"),
        records={k: v["config_hash"] for k, v in old.items()}, anchor=anchor,
        prefixes={k: dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for k, p in prefix_files.items()},
        historical_limitations=read_json(PREVIOUS / "calibrated_state_archive.json")))
    models = dict(old)

    def fit(base, name, weights):
        cfg = dict(old[base]["config"]["model"], weights=weights)
        models[name] = stamp(engine.fit(name, cfg, shared=prefix_files[base]), out)

    batches = monitor_indices(x.data.cases)
    initial = {name: probes(engine, old[name], 1000, ("match2",), batches) for name in BASE}
    rho = plan["rho"]
    coefficients = {name: calibrate(pack, "match2", rho) for name, pack in initial.items()}
    actual = {name: anchor * rho / val for name, val in coefficients.items()}
    supplement = any(abs(actual[name] / rho - 1) > plan["gradient_relative_threshold"] for name in BASE[:-1])
    atomic_json(out / "initial_calibration.json", dict(anchor=anchor, rho=rho, coefficients=coefficients,
                                                       actual_ratios=actual, supplement=supplement, batches=batches))
    for base in BASE[:-1]:
        fit(base, base + "_match2", {"match2": anchor})
    if supplement:
        for base in BASE[:-1]:
            fit(base, base + "_match2_rho", {"match2": coefficients[base]})

    i0 = models["I0_match2"]["final"]["dev_mse"]
    ib = models["IB_match2"]["final"]["dev_mse"]
    g = plan["extensions"]
    change = []
    if i0 < models["I0"]["final"]["dev_mse"] and i0 <= ib * g["i0_competitive_ratio"]:
        change.append(("I0", ("match", "energy")))
    if 1 - ib / models["IB"]["final"]["dev_mse"] >= g["ib_structure_min_improvement"]:
        change.append(("IB", ("diag", "perm")))
    extensions = {}
    for base, keys in change:
        packet = probes(engine, old[base], 1000, keys, batches)
        for term in keys:
            extensions[base + "_" + term] = dict(base=base, term=term, weight=calibrate(packet, term, rho))
    text = ["# 开发侧扩展记录", "", f"固定步 I0_match2 开发 MSE 为 {i0:.10g}，IB_match2 为 {ib:.10g}。", ""]
    for name, e in extensions.items():
        text.append(f"- {name} 从 {e['base']} 的第 1000 步接续，辅助项 {e['term']}，权重 {e['weight']:.12g}。"
                    "对照为纯 MSE 与正常 TM2，主指标为开发轨迹等权物理 MSE。固定 6000 步停止。")
    text.extend(["", "优先完成时间匹配、能量惩罚和结构控制。其余扩展未启动。", ""])
    (out / "exploration_log.md").write_text("\n".join(text), encoding="utf-8")
    for name, e in extensions.items():
        fit(e["base"], name, {e["term"]: e["weight"]})

    rows = []
    for name, pack in initial.items():
        rows.extend(details(name+"_fixed_lambda_probe", 1000, pack, {"match2": anchor}))
    for name, record in models.items():
        weights = record["config"]["model"]["weights"]
        if not weights:
            continue
        for step in (1000, 2000, 4000, 6000):
            rows.extend(details(name, step, probes(engine, record, step, tuple(weights), batches), weights))
    table(out / "gradient_diagnostics.csv", rows)
    from .mechanisms import interventions

    diagnostic = interventions(models, y, basis, plan, "development", "final")
    table(out / "history_interventions_development.csv", diagnostic)
    best = min(models, key=lambda n: (models[n]["selected"]["dev_mse"], models[n]["selected"]["step"], n))
    overall = best if models[best]["selected"]["dev_mse"] < old["I0"]["training"][0]["dev_mse"] else "S3"
    selection = dict(best_new=best, overall=overall, factorial=list(FACTORIAL), extensions=extensions,
                     gradient_supplement=supplement, primary=plan["primary"])
    atomic_json(out / "selections.json", selection)
    for name in ("outer_plan.json", "period_plan.json"):
        copyfile(PREVIOUS / name, out / name)
    names = (*REGISTRATION, "reuse_audit.json", "initial_calibration.json", "gradient_diagnostics.csv",
             "history_interventions_development.csv", "exploration_log.md", "selections.json", "outer_plan.json",
             "period_plan.json", "teacher_train.json", "teacher_development.json", "native_train.json",
             "native_development.json", "triplet_manifest.json")
    assert 3 + 3 * int(supplement) + len(extensions) + 15 <= 33
    atomic_json(out / "FROZEN.json", dict(
        prior_test_manifest_blob=prior["prior_test_manifest_blob"], bindings={n: sha(out / n) for n in names},
        selection=selection, models=models, anchor=anchor, prior_freeze=sha(PREVIOUS / "FROZEN.json"),
        basis_sha256=sha(PREVIOUS / "temporal_basis.pt"), outer_new=plan["outer_new"]))
    print("FROZEN", selection, flush=True)
