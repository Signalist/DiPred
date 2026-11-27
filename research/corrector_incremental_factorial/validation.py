"""Score the frozen factorial and its isolated Re replications."""

import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.data import ROOT, guard
from research.corrector_residual_feedback.model import base, predict as teacher_predict
from research.corrector_simple_baseline.fit import table
from research.corrector_spacetime_validation.expansion import period_data
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_spacetime_validation.outer import subset
from research.corrector_window_study.data import read_json
from research.corrector_incremental_structure.fit import Fits, predict
from research.corrector_incremental_structure.score import IncrementalScores
from research.corrector_incremental_structure.teachers import bundle, full_teacher, main_bundle, outer_bundles
from research.corrector_incremental_structure.validation import (
    confirmation_inventory, inference_cost, read_rows, reuse_period_cache, target_invariance,
)

from .core import BASE, FACTORIAL, load_head, prefix, stamp, verify_reuse
from .mechanisms import factorial_scores, interventions
from .run import PREVIOUS

REUSED_OUTER = ("I0", "IB", "IB_match2")


def add(book, study, view, split, name, policy, estimate):
    book.add_view(view, split, name, policy, estimate, study.norm["std"])


def compare(freeze, controls):
    pri = freeze["selection"]["primary"]

    def pairs(split, present):
        result = [(label, a, b) for label in ("P1", "P2", "P3") for a, b in pri[label]]
        result.append(("P4", freeze["selection"]["best_new"], "S3"))
        for arm in present:
            result.extend([("native", arm, "Native"), ("s3", arm, "S3")])
            if arm in controls:
                result.append(("own_mse", arm, controls[arm]))
        for a in ("I", "D"):
            for suffix in ("", "_match2"):
                result.append(("history", a+"B"+suffix, a+"0"+suffix))
        for suffix in ("0", "B"):
            for loss in ("", "_match2"):
                result.append(("increment", "I"+suffix+loss, "D"+suffix+loss))
        return sorted({(label, a, b) for label, a, b in result if a != b and a in present and b in present})

    return pairs


def outer(study, train, book, freeze, audit_prior, basis):
    lineage, saved = {}, {}
    old_lineage = read_json(PREVIOUS / "outer_lineage.json")
    for region in ("interpolation", "upper", "lower"):
        print("OUTER START", region, flush=True)
        active, held, schedule = outer_bundles(study, train, region)
        earlier = old_lineage["regimes"][region]
        assert {k: v for k, v in earlier.items() if k != "candidates"} == active.identity
        engine = Fits(study.out, study.cache, active, None, schedule, basis)
        data = {}
        for name in REUSED_OUTER:
            tag = f"outer_{region}_{name}"
            record = stamp(read_json(PREVIOUS / "structured_fits" / f"{tag}.json"), PREVIOUS)
            shared = PREVIOUS / f"shared_outer_{region}_IB.pt" if name == "IB_match2" else None
            verify_reuse(record, engine, audit_prior, freeze["models"][name]["config"]["model"], shared)
            assert record["selected"] == record["final"]
            assert record["final"]["sha256"] == earlier["candidates"][name]["sha256"]
            data[name] = record
        i0_prefix = prefix(data["I0"], study.out)
        for name in ("D0", "DB"):
            tag = f"outer_{region}_{name}"
            model = freeze["models"][name]["config"]["model"]
            data[name] = stamp(engine.fit(tag, model), study.out)
        for name, shared in (("D0_match2", prefix(data["D0"], study.out)),
                             ("DB_match2", prefix(data["DB"], study.out)),
                             ("I0_match2", i0_prefix)):
            model = freeze["models"][name]["config"]["model"]
            assert model["weights"] == {"match2": freeze["anchor"]}
            tag = f"outer_{region}_{name}"
            data[name] = stamp(engine.fit(tag, model, shared=shared), study.out)
        split = "holdout_" + region
        add(book, study, held, split, "Native", "final", torch.zeros_like(held.current))
        add(book, study, held, split, "S3", "final", held.current)
        for name, record in data.items():
            net = load_head(record, record["final"], basis, held.current.device)
            cfg = record["config"]["model"]
            r = predict(net, held, cfg)
            target_invariance(held, lambda: None, dict(record, kind="structured"), net, r)
            add(book, study, held, split, name, "final", r)
        saved.update({f"outer_{region}_{name}": rec for name, rec in data.items()})
        lineage[region] = dict(held.identity, candidates={n: dict(final=d["final"], directory=d["directory"],
                                                                config=d["config"], target_invariance=True)
                                                         for n, d in data.items()})
        atomic_json(study.out / "outer_lineage.json", dict(freeze_sha256=sha(study.out / "FROZEN.json"), regimes=lineage))
        print("OUTER COMPLETE", region, len(data), flush=True)
    return saved


@torch.inference_mode()
def ordinary(study, freeze, book, basis, plan):
    models = freeze["models"]
    future, costs, fields, effects = [], [], [], []
    for tag, label in (("development", "development"), ("test", "historical_test")):
        data, features, _ = study.get(tag)
        view = main_bundle(study, data, features, tag)
        teacher, teacher_hash = full_teacher(study, data.p.device)
        active = ("Native", "S3", *models)
        fixed = plan["fixed_case"]
        is_dev = tag == "development"
        if is_dev:
            index, = [i for i, c in enumerate(data.cases) if (c["file"],c["start"]) == (fixed["file"],fixed["start"])]
            lead = fixed["lead"] - 1
            source_fields = {"truth_residual": view.target[index, :, lead].cpu(),
                             "s3_correction": view.current[index, :, lead].double().cpu()}
        else:
            previous = data.y
            data.y = previous.flip(0) + .03125
            try:
                f, _ = study.inputs(data)
                assert torch.equal(f, features)
                assert torch.equal(teacher_predict(teacher, f, study.cfg["external_rms"]), view.current)
            finally:
                data.y = previous
        for name in active:
            item = models.get(name)
            for policy in ("selected", "final"):
                net = None if item is None else load_head(item, item[policy], basis, data.p.device)
                if net is None:
                    r = view.current if name == "S3" else torch.zeros_like(view.current)
                else:
                    r = predict(net, view, item["config"]["model"])
                    if not is_dev:
                        target_invariance(view, lambda: None, dict(item, kind="structured"), net, r)
                        future.append(dict(arm=name, policy=policy, bitwise=True))
                add(book, study, view, label, name, policy, r)
                if is_dev and policy=="selected":
                    costs.append(inference_cost(study, teacher, view, name,
                                                dict(item, kind="structured") if item else None, net))
                    if name in ("IB_match2", freeze["selection"]["best_new"]):
                        value = r[index, :, lead].double().cpu()
                        source_fields[name+"_increment"] = value - source_fields["s3_correction"]
                        source_fields[name+"_remaining"] = source_fields["truth_residual"] - value
        if is_dev:
            for field, v in source_fields.items():
                for c, channel in enumerate(("u","v")):
                    fields.extend(dict(field=field, channel=channel, y=y, x=x, value=float(v[c,y,x]))
                                  for y in range(v.shape[-2]) for x in range(v.shape[-1]))
        else:
            effects.extend(interventions(models, view, basis, plan, label, "final"))
        print("SCORED", label, flush=True)

    reuse_period_cache(study)
    period, evidence = period_data(study, data)
    pscale = study.cfg["external_rms"]
    b = base(period.x, period.p, study.norm)
    current = teacher_predict(teacher, b, pscale)
    history = teacher_predict(teacher, base(period.previous_x, period.previous_p, study.norm), pscale)
    ids = [i for i,c in enumerate(period.cases) if c["beyond_prior_window"]]
    full = bundle(period, b, current, history, study.scale, {"cohort": "period"})
    later = bundle(subset(period, ids), b[ids], current[ids], history[ids], study.scale, {"cohort": "period_new"})
    assert len(period.cases)==96 and len(ids)==48
    for name in active:
        item = models.get(name)
        if item is None:
            r = current if name=="S3" else torch.zeros_like(current)
        else:
            net = load_head(item, item["final"], basis, data.p.device)
            r = predict(net, full, item["config"]["model"])
        add(book, study, full, "period", name, "final", r)
        add(book, study, later, "period_new", name, "final", r[ids])
    atomic_json(study.out/"period_provenance.json", evidence)
    independent = confirmation_inventory(study, read_json(study.out/"period_plan.json"))
    table(study.out/"inference_cost.csv", costs)
    table(study.out/"fixed_fields.csv", fields)
    effects.extend(read_rows(study.out/"history_interventions_development.csv"))
    table(study.out/"history_interventions.csv", effects)
    return dict(future_invariance=future, teacher=teacher_hash, period_windows=96,
                later_windows=48, independent=independent)


def reproduction(out, book):
    signed = read_json(PREVIOUS/"artifact_audit.json")
    assert sha(PREVIOUS/"scores.csv")==signed["files"]["scores.csv"]
    before = {(r["split"],r["policy"],r["arm"]):float(r["mse"]) for r in read_rows(PREVIOUS/"scores.csv")}
    matches = []
    for row in book.recomputed:
        key = tuple(row[k] for k in ("split","policy","arm"))
        corpus, _, arm = key
        if key not in before or arm not in {"Native","S3","I0","IB","D0","DB","IB_match2"}:
            continue
        if corpus not in ("development","historical_test","period","period_new") and not corpus.startswith("holdout_"):
            continue
        error = abs(row["mse"]-before[key])
        assert error<1e-12, key
        matches.append(dict(row, absolute_error=error))
    assert len(matches)>=43
    atomic_json(out/"historical_reproduction.json", matches)
    return len(matches)


def validate(study, figures):
    out = study.out
    freeze = guard(out)
    prior = read_json(PREVIOUS/"artifact_audit.json")
    assert freeze["prior_freeze"]==sha(PREVIOUS/"FROZEN.json") and prior["passed"]
    basis_path = PREVIOUS/"temporal_basis.pt"
    assert sha(basis_path)==freeze["basis_sha256"]
    basis = torch.load(basis_path, map_location="cpu", weights_only=True)
    plan = read_json(out/"experiment_plan.json")
    train, _, _ = study.get("train")
    values = (train.y.double()-train.p.double()).abs()[:,:,::2,::2,::2]
    thresholds = [float(torch.quantile(values[:,c].contiguous().flatten(), .95)) for c in (0,1)]
    book = IncrementalScores(basis, thresholds)
    records = outer(study, train, book, freeze, prior, basis)
    del train, values
    evidence = ordinary(study, freeze, book, basis, plan)
    evidence["previous_reproduction_rows"] = reproduction(out,book)
    controls = {}
    for name in freeze["models"]:
        anchor = name.split("_")[0]
        if name!=anchor and anchor in BASE:
            controls[name] = anchor
    result = book.write(out, controls, compare(freeze,controls))
    rows = []
    for name, record in {**freeze["models"],**records}.items():
        rows.extend(dict(r, arm=name, directory=record["directory"]) for r in record["training"])
    fields = sorted({k for r in rows for k in r})
    table(out/"training.csv", [{k:r.get(k) for k in fields} for r in rows])
    effects = factorial_scores(read_rows(out/"trajectories.csv"), plan["bootstrap"])
    table(out/"factorial_effects.csv", [r for r in effects if r["kind"]=="factorial" and r["order"]==1])
    table(out/"factorial_interactions.csv", [r for r in effects if r["order"]>1 or r["kind"]=="conditional"])
    atomic_json(out/"verification.json", dict(result, **evidence, freeze_sha256=sha(out/"FROZEN.json")))
    from .publishables import finish

    finish(study, figures, freeze, records)
