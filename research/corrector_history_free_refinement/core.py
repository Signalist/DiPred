"""Verify source models, fit the registered objectives and freeze development selection."""
from collections import defaultdict
from copy import copy
from shutil import copyfile
from statistics import mean

import torch

from research.corrector.run import sha
from research.corrector_incremental_factorial.core import calibrate, details, load_head, prefix, probes, stamp, verify_reuse
from research.corrector_incremental_structure.fit import Fits, predict as predict_increment
from research.corrector_incremental_structure.teachers import full_teacher, main_bundle
from research.corrector_residual_feedback.data import guard
from research.corrector_residual_feedback.model import predict
from research.corrector_residual_feedback.run import references
from research.corrector_residual_optimization.fit import checks, track_weights
from research.corrector_simple_baseline.fit import table
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_spacetime_validation.gradients import monitor_indices
from research.corrector_window_study.data import ROOT, read_json

from .extra import fit_extra
from .run import FACTORIAL, S3_RESULTS, STRUCTURE

REGISTRATION = ("PROTOCOL.md", "hypotheses.json", "experiment_plan.json", "data_roles.json")
REFERENCES = ("I0", "I0_match2", "I0_match2_rho", "IB", "IB_match2", "IB_match", "IB_energy")
MONITOR = (1000, 2000, 4000, 6000)


def prior_study():
    frozen = guard(FACTORIAL)
    audit = read_json(FACTORIAL / "artifact_audit.json")
    assert audit["passed"] and audit["freeze_sha256"] == sha(FACTORIAL / "FROZEN.json")
    assert frozen["selection"]["overall"] == "IB_match2"
    basis_path = STRUCTURE / "temporal_basis.pt"
    assert sha(basis_path) == frozen["basis_sha256"]
    return frozen, audit, torch.load(basis_path, map_location="cpu", weights_only=True)


def teacher_distribution(study, view, basis):
    teacher, _ = full_teacher(study, view.current.device)
    full = predict(teacher, view.base, study.cfg["external_rms"])
    cohorts = defaultdict(list)
    for i, case in enumerate(view.data.cases):
        cohorts[str(case["re"])].append(i)
    cohorts["all"] = list(range(len(view.data.cases)))
    rows = []
    for role, value in (("OOF", view.current), ("full_in_sample", full)):
        for re, ids in sorted(cohorts.items()):
            weight = track_weights(view.data)[ids]
            weight = weight / weight.sum()
            for kind, field in (("correction", value[ids]), ("remaining", view.target[ids] - value[ids])):
                for name, channel in zip(("u", "v"), field.double().unbind(1)):
                    energy = channel.square().mean((1, 2, 3))
                    modes = (channel.movedim(1, -1) @ basis.to(channel)).square().mean((1, 2))
                    for index, mode in enumerate((weight[:, None] * modes).sum(0).tolist()):
                        rows.append(dict(teacher=role, re=re, field=kind, channel=name, mode=index,
                                         mode_energy=mode, rms=float((weight*energy).sum().sqrt())))
    table(study.out / "teacher_distribution.csv", rows)


def gradient_rows(engine, record, name, batches):
    weight = record["config"]["model"]["weights"]
    for step in MONITOR:
        packets = probes(engine, record, step, tuple(weight), batches)
        for row in details(name, step, packets, weight):
            layers = packets[row["batch"]][row["term"]][2]
            yield dict(row, conv1=layers[0], conv2=layers[1], conv3=layers[2])


def replay_development(record, view, basis):
    weights = track_weights(view.data)
    rows = {}
    for archived in record["training"]:
        net = load_head(record, archived, basis, view.current.device)
        output = predict_increment(net, view, record["config"]["model"])
        metrics, error = checks(output, view.data, weights)
        delta = output-view.current.double()
        remaining = view.target-view.current.double()
        c, u = [float((x.mean((1,2,3,4))*weights).sum()) for x in (delta*remaining,delta.square())]
        updated = {f"dev_{name}":value for name,value in metrics.items()}
        updated.update(dev_identity_error=error,dev_C_delta=c,dev_U_delta=u,dev_G_delta=2*c-u)
        assert abs(archived["dev_mse"]-updated["dev_mse"]) < 2e-10
        rows[str(archived["step"])] = updated
    return rows


def fit_main(study):
    out = study.out
    plan = read_json(out / "experiment_plan.json")
    for name in REGISTRATION:
        assert (out / name).is_file()
    previous, audit, basis = prior_study()
    train, train_z, _ = study.get("train")
    dev, dev_z, _ = study.get("development")
    train_view = main_bundle(study, train, train_z, "train")
    dev_view = main_bundle(study, dev, dev_z, "development")
    for split in ("train", "development"):
        name = f"teacher_{split}.json"
        assert sha(FACTORIAL / name)==previous["bindings"][name]
        old, this = (read_json(root / name) for root in (FACTORIAL,out))
        excluded = {"distribution"} if split == "train" else {"current","distribution"}
        assert {k:v for k,v in old.items() if k not in excluded} == {k:v for k,v in this.items() if k not in excluded}
    original_dev = read_json(FACTORIAL / "teacher_development.json")
    replay = original_dev != dev_view.identity
    engine = Fits(out, study.cache, train_view, dev_view, study.schedule, basis)
    known = {n: previous["models"][n] for n in REFERENCES}
    archived_engine = copy(engine)
    archived_engine.identity = dict(engine.identity,development=original_dev)
    replays = {}
    for name, record in known.items():
        model = record["config"]["model"]
        plain = known["IB" if model["history"] else "I0"]
        shared = ROOT / plain["directory"] / next(s["path"] for s in plain["states"] if s["step"] == 1000)
        if model["history"]:
            shared = STRUCTURE / "shared_IB.pt"
        verify_reuse(record, archived_engine, audit, model, shared if record["config"]["shared"] else None)
        if replay:
            replays[name] = replay_development(record,dev_view,basis)
    source = prefix(known["I0"], out)
    batches = monitor_indices(train.cases)
    old_calibration = read_json(FACTORIAL / "initial_calibration.json")
    assert batches == old_calibration["batches"]
    terms = tuple(term for term in plan["families"].values() if term)
    packets = probes(engine, known["I0"], 1000, terms, batches)
    coefficients = {family: {str(rho): calibrate(packets, term, rho) for rho in plan["rho"]}
                    for family, term in plan["families"].items() if term}
    reused_tm2 = known["I0_match2_rho"]["config"]["model"]["weights"]["match2"] == coefficients["TM2"]["0.3"]
    atomic_json(out / "reuse_audit.json", dict(
        source_freeze_sha256=sha(FACTORIAL / "FROZEN.json"), source_audit_sha256=sha(FACTORIAL / "artifact_audit.json"),
        original_IB_selection=previous["selection"]["overall"],
        model_configs={name: item["config_hash"] for name, item in known.items()},
        source_prefix=dict(path=str(source.relative_to(ROOT)), sha256=sha(source)),
        fixed_batch_identity=batches, reuse_TM2_03=reused_tm2, legacy_missing_snapshots=14,
        development_replay=dict(recomputed=replay,old_teacher=original_dev["current"],new_teacher=dev_view.identity["current"],
            max_mse_difference={name:max(abs(row["dev_mse"]-replays[name][str(row["step"])]["dev_mse"])
                                    for row in known[name]["training"]) for name in replays})))
    atomic_json(out / "calibration.json", dict(batches=batches, coefficients=coefficients,
                norms={term:mean(float(packet[term][1].norm()) for packet in packets) for term in ("mse", *terms)}))
    models = {"M0": known["I0"], "I0_match2_fixed": known["I0_match2"],
              **{name:known[name] for name in ("IB","IB_match2","IB_match","IB_energy")}}
    if not reused_tm2:
        models["I0_match2_legacy_rho"] = known["I0_match2_rho"]
    matrix = {"M0": ["M0"]}

    replayed_prefix = dict(original_development=original_dev, metrics=replays["I0"]) if replay else None

    def fit(name, term, weight):
        config = dict(models["M0"]["config"]["model"], weights={term: weight})
        models[name] = stamp(engine.fit(name, config, shared=source, replayed_prefix=replayed_prefix), out)

    for family, weights in coefficients.items():
        matrix[family] = []
        for rho, weight in weights.items():
            name = f"I0_{family}_rho{rho.replace('.', '')}"
            matrix[family].append(name)
            if family == "TM2" and rho == "0.3" and reused_tm2:
                models[name] = known["I0_match2_rho"]
            else:
                fit(name, plan["families"][family], weight)

    s3, _, _ = references()
    assert s3["selected"]["step"] == 6000
    independent_s3, digest = full_teacher(study, train.p.device)
    src = dict(path=str((S3_RESULTS / s3["selected"]["checkpoint"]).relative_to(ROOT)),
               sha256=digest, layout="channels_last_3d")
    models["S3_extra_MSE"] = fit_extra(study, "S3_extra_MSE", independent_s3, src, train, train_z,
                                     study.cfg["external_rms"], study.schedule, dev, dev_z)
    del independent_s3
    assert sha(ROOT / src["path"]) == src["sha256"]

    def criterion(name):
        item = models[name]
        old_name = item["config"].get("name")
        metrics = replays.get(old_name, {}).get(str(item["final"]["step"]), item["final"])
        return metrics["dev_mse"], name

    choices = {family:min(names, key=criterion) for family,names in matrix.items()}
    winner = min(choices.values(), key=criterion)
    time = min((choices["TM1"], choices["TM2"]), key=criterion)
    extra_outer = criterion("S3_extra_MSE")[0] <= criterion(winner)[0]*plan["extra_competitive_ratio"]

    exploration = []
    target = models[time]["final"]["train_U_delta"]
    points = sorted((models[name]["final"]["train_U_delta"], name) for name in matrix["EN"])
    en_energy = models[choices["EN"]]["final"]["train_U_delta"]
    if abs(en_energy/target-1) > plan["energy_relative_threshold"] and points[0][0] < target < points[1][0]:
        from math import exp, log

        left, right = points
        fraction = log(target/left[0])/log(right[0]/left[0])
        weights = [models[name]["config"]["model"]["weights"]["energy"] for _,name in points]
        beta = exp(log(weights[0]) + fraction*log(weights[1]/weights[0]))
        name = "I0_EN_energy_matched"
        atomic_json(out / "extension_registration.json", dict(arm=name, target=target, beta=beta,
                    endpoints=points, weights=weights, variable="EN weight", stop=6000))
        fit(name, "energy", beta)
        actual = models[name]["final"]["train_U_delta"]
        exploration.append(f"训练侧能量对照的目标为 {target:.10g}，实测 {actual:.10g}，相对误差 {actual/target-1:+.3%}。")
    else:
        exploration.append(f"最佳时间目标训练能量为 {target:.10g}，最佳 EN 为 {en_energy:.10g}。实际能量及两档 EN 的区间未触发额外配置。")
    exploration.append("其余有限扩展留给独立数据确认。")
    (out / "exploration_log.md").write_text("# 开发侧对照\n\n" + "\n\n".join(exploration) + "\n", encoding="utf-8")

    rows = []
    for name, item in models.items():
        if "model" in item["config"] and item["config"]["model"]["history"] == 0:
            rows.extend(gradient_rows(engine, item, name, batches))
    table(out / "gradient_diagnostics.csv", rows)
    teacher_distribution(study, train_view, basis)
    selection = dict(families=choices, matrix=matrix, candidate=winner, best_time=time, extra_outer=extra_outer,
                     selected_step={name:item["selected"]["step"] for name,item in models.items()},
                     primary=[["P1",winner,"S3"],["P2",choices["TM2"],choices["TM1"]],["P3",time,choices["EN"]]])
    atomic_json(out / "selections.json", selection)
    plans = ("outer_plan.json", "period_plan.json")
    for filename in plans:
        copyfile(FACTORIAL / filename, out / filename)
    main_new = sum(item["directory"].endswith(out.name) for item in models.values())
    planned_outer = 9 + 3*extra_outer
    assert main_new + planned_outer <= plan["max_total"]
    bindings = (*REGISTRATION, *plans, "calibration.json", "reuse_audit.json", "selections.json",
                "exploration_log.md", "teacher_distribution.csv", "gradient_diagnostics.csv",
                "triplet_manifest.json", "native_train.json", "native_development.json",
                "teacher_train.json", "teacher_development.json")
    if (out / "extension_registration.json").is_file():
        bindings = (*bindings, "extension_registration.json")
    atomic_json(out / "FROZEN.json", dict(prior_test_manifest_blob=previous["prior_test_manifest_blob"],
                bindings={name:sha(out / name) for name in bindings}, models=models, selection=selection,
                prior_freeze=sha(FACTORIAL / "FROZEN.json"), basis_sha256=previous["basis_sha256"],
                source_s3=src, main_new=main_new, outer_max=planned_outer))
    print("FROZEN", selection, flush=True)
