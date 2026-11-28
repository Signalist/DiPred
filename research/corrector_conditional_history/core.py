"""Train and freeze the development arms before outer assessment."""
from pathlib import Path
from shutil import copyfile

import torch

from research.corrector.run import sha
from research.corrector_incremental_factorial.core import prefix, stamp
from research.corrector_incremental_structure.fit import Fits
from research.corrector_incremental_structure.teachers import main_bundle
from research.corrector_remaining_residual_transfer.core import verify_archive
from research.corrector_remaining_residual_transfer.run import STRUCTURE
from research.corrector_residual_feedback.data import guard
from research.corrector_simple_baseline.fit import table
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_window_study.data import ROOT, read_json

from .donors import prepare
from .run import PREVIOUS

REGISTRATION = ("PROTOCOL.md","hypotheses.json","experiment_plan.json","data_roles.json","donor_plan.json")


def prior():
    freeze = guard(PREVIOUS)
    audit = read_json(PREVIOUS / "artifact_audit.json")
    assert audit["passed"] and audit["freeze_sha256"] == sha(PREVIOUS/"FROZEN.json")
    temporal = STRUCTURE/"temporal_basis.pt"
    assert sha(temporal) == freeze["basis_sha256"]
    return freeze, audit, torch.load(temporal,map_location="cpu",weights_only=True)


def shared_archive(record, base, region=None):
    step = record["config"]["first"]
    if not step:
        return None
    directory = ROOT / base["directory"]
    row = next(r for r in base["states"] if r["step"]==step)
    ordinary = directory / row["path"]
    if sha(ordinary)==record["config"]["shared"]:
        return ordinary
    tag = record["config"]["name"].split("_")[0]
    if region:
        tag = f"outer_{region}_IB"
    legacy = STRUCTURE/f"shared_{tag}.pt"
    assert sha(legacy)==record["config"]["shared"]
    return legacy


def training_queue(engine, baselines, names, donors, scope=""):
    records = {}
    for name in names:
        arm, kind, loss = name.split("_")
        model = dict(baselines[f"{arm}_{loss}"]["config"]["model"])
        if kind=="pair":
            model["paired_history"] = True
        donor_plan=donors if kind=="shuffled" else None
        shared=None
        if loss=="EN":
            base=f"{arm}_{kind}_MSE"
            if base not in records:
                assert kind=="pair" and base not in names
                plain=dict(model,weights={})
                records[base]=stamp(engine.fit(scope+base+"_prefix",plain,end_step=1000),engine.out)
            shared=prefix(records[base],engine.out)
        records[name] = stamp(engine.fit(scope+name,model,shared=shared,
                                          donor_plan=donor_plan),engine.out)
    return records


def fit_core(study):
    out = study.out
    for file in REGISTRATION:
        assert (out/file).is_file()
    plan, recipe = (read_json(out/f) for f in ("experiment_plan.json","donor_plan.json"))
    old, audit, basis = prior()
    data, features, _ = study.get("train")
    train = main_bundle(study,data,features,"train")
    data, features, _ = study.get("development")
    dev = main_bundle(study,data,features,"development")
    for split in ("train","development"):
        file=f"teacher_{split}.json"
        assert sha(out/file)==old["bindings"][file]
    engine = Fits(out,study.cache,train,dev,study.schedule,basis)
    previous = old["models"]
    reusable = {name:previous[name] for name in (*plan["ordinary"],*plan["historical_reference"])}
    for name,record in reusable.items():
        base=previous[name.split("_")[0]+"_MSE"]
        verify_archive(record,engine,audit,record["config"]["model"],shared_archive(record,base))
    atomic_json(out/"reuse_audit.json",dict(previous_freeze=sha(PREVIOUS/"FROZEN.json"),
        main={name:dict(directory=r["directory"],config_hash=r["config_hash"]) for name,r in reusable.items()}))
    slots, rows, coverage = prepare(out,"main",train,study.schedule,recipe["seed"],recipe["time_slack_frames"])
    table(out/"history_donors_main.csv",rows)
    atomic_json(out/"donor_main.json",coverage)
    from .diagnostics import ridge, observable_thresholds

    ridge(out,train,plan["ridge"])
    atomic_json(out/"observable_thresholds.json",observable_thresholds(train))
    for name in ("temporal_correlations.csv","spatial_correlations.csv","transfer_modes.csv",
                 "outer_plan.json","period_plan.json","later_plan.json"):
        assert sha(PREVIOUS/name)==old["bindings"][name]
        copyfile(PREVIOUS/name,out/name)
    new = training_queue(engine,reusable,plan["new"],slots)
    models = {**reusable,**new}
    pairs = {arm:min((f"{arm}_pair_{loss}" for loss in ("MSE","EN")),
                    key=lambda name:(models[name]["final"]["dev_mse"],name)) for arm in ("IB","ID")}
    candidates = [*plan["ordinary"],*(n for n in new if "_pair_" in n)]
    selection = {rule:min(candidates,key=lambda n:(models[n][rule]["dev_mse"],models[n][rule]["step"],n))
                 for rule in ("final","selected")}
    selection.update(pair_outer=pairs, previous_final=plan["previous_final"],candidates=candidates)
    atomic_json(out/"selections.json",selection)
    files=(*REGISTRATION,"teacher_train.json","teacher_development.json","native_train.json","native_development.json",
           "triplet_manifest.json","reuse_audit.json","donor_batches_main.pt","history_donors_main.csv","donor_main.json",
           "observable_thresholds.json","conditional_history_scores.csv","outer_plan.json","period_plan.json","later_plan.json",
           "temporal_correlations.csv","spatial_correlations.csv","transfer_modes.csv","selections.json")
    atomic_json(out/"MAIN_FROZEN.json",dict(models=models,selection=selection,
        bindings={name:sha(out/name) for name in files},previous_freeze=sha(PREVIOUS/"FROZEN.json"),
        prior_test_manifest_blob=old["prior_test_manifest_blob"],basis_sha256=old["basis_sha256"],
        backbone=study.triplets.source.identity))
    print("MAIN FROZEN",selection,flush=True)
