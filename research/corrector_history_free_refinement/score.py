"""Score the registered time ranges and audit the delivered artifacts."""
from math import isfinite
from pathlib import Path
from statistics import mean

from PIL import Image
import torch

from research.corrector.run import sha
from research.corrector_incremental_factorial.core import load_head
from research.corrector_incremental_structure.audit import verify_state
from research.corrector_incremental_structure.teachers import bundle, full_teacher, main_bundle
from research.corrector_incremental_structure.validation import (
    confirmation_inventory, read_rows, reuse_period_cache, target_invariance,
)
from research.corrector_residual_feedback.data import OUT as FEEDBACK, guard
from research.corrector_residual_feedback.fit import state_digest
from research.corrector_residual_feedback.model import base, predict as teacher_predict
from research.corrector_simple_baseline.fit import table
from research.corrector_simple_baseline.measure import groups
from research.corrector_spacetime_optimization.fit import config_hash
from research.corrector_spacetime_validation.expansion import period_data
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_spacetime_validation.objectives import verify_math
from research.corrector_spacetime_validation.outer import subset
from research.corrector_window_study.data import ROOT, read_json, tensor_sha

from .evaluate import add, check_publication, estimate, is_extra, later_period, load_extra
from .run import FACTORIAL, STRUCTURE


@torch.inference_mode()
def score_remaining(study, frozen, book, basis, plan):
    out = study.out
    models = frozen["models"]
    winner = frozen["selection"]["candidate"]
    evidence = {"future_perturbation": []}
    fields = []
    for tag, split in (("development","development"),("test","historical_test")):
        data, features, _ = study.get(tag)
        view = main_bundle(study,data,features,tag)
        teacher, digest = full_teacher(study,data.p.device)
        assert digest == frozen["source_s3"]["sha256"]
        for policy in ("final", "selected"):
            add(book,study,view,split,"Native",policy,torch.zeros_like(view.current))
            add(book,study,view,split,"S3",policy,view.current)
            for name, record in models.items():
                output, net = estimate(record,policy,view,basis)
                if tag == "test":
                    original = data.y
                    if is_extra(record):
                        data.y = original.roll(1,0)+.03125
                        try:
                            altered, _ = study.inputs(data)
                            assert torch.equal(teacher_predict(net,altered,record["config"]["scale"]),output)
                        finally:
                            data.y = original
                    else:
                        target_invariance(view,lambda:None,dict(record,kind="structured"),net,output)
                    evidence["future_perturbation"].append(dict(arm=name,policy=policy,bitwise=True))
                fixed = plan["field"]
                if tag=="development" and policy=="final" and name==winner:
                    case, = [i for i,c in enumerate(data.cases) if (c["file"],c["start"])==(fixed["file"],fixed["start"])]
                    lead = fixed["lead"]-1
                    arrays = dict(true_residual=view.target[case,:,lead],S3=view.current[case,:,lead],
                                  increment=(output-view.current.double())[case,:,lead],
                                  remaining=(view.target-output.double())[case,:,lead])
                    for quantity, tensor in arrays.items():
                        for channel, xy in zip(("u","v"),tensor.tolist()):
                            fields.extend(dict(field=quantity,channel=channel,y=i,x=j,value=value)
                                          for i,line in enumerate(xy) for j,value in enumerate(line))
                    extra = load_extra(models["S3_extra_MSE"],"final",data.p.device)
                    evidence["publication"] = check_publication(study,teacher,record,view,extra,basis,case)
                add(book,study,view,split,name,policy,output)
        print("SCORED",split,flush=True)
    table(out/"fixed_fields.csv",fields)
    atomic_json(out/"publication_verification.json",evidence["publication"])

    reuse_period_cache(study)
    period, provenance = period_data(study,data)
    last_ids = [i for i,c in enumerate(period.cases) if c["beyond_prior_window"]]
    assert len(period.cases)==96 and len(last_ids)==48
    scale = study.cfg["external_rms"]
    rows = []
    additional = later_period(study,data,period,plan["later_same_trajectory"])
    assert len(additional.cases) == 168
    for i, c in enumerate(additional.cases):
        rows.append(dict(c, x_sha256=tensor_sha(additional.x[i]),p_sha256=tensor_sha(additional.p[i]),
                         y_sha256=tensor_sha(additional.y[i]),denoising_labels="99 66 33 0",zero_initial_noise=True))
    table(out/"later_window_provenance.csv",rows)
    for split, cohort in (("period",period),("period_new",subset(period,last_ids)),("period_extended",additional)):
        design = base(cohort.x,cohort.p,study.norm)
        current = teacher_predict(teacher,design,scale)
        previous = teacher_predict(teacher,base(cohort.previous_x,cohort.previous_p,study.norm),scale)
        view = bundle(cohort,design,current,previous,study.scale,{"split":split})
        add(book,study,view,split,"Native","final",torch.zeros_like(current))
        add(book,study,view,split,"S3","final",current)
        for name, record in models.items():
            output, _ = estimate(record,"final",view,basis)
            add(book,study,view,split,name,"final",output)
        print("SCORED",split,len(cohort.cases),flush=True)
    evidence["windows"] = dict(period=96,period_new=48,period_extended=len(additional.cases))
    evidence["independent"] = confirmation_inventory(study, read_json(out/"period_plan.json"))
    atomic_json(out/"period_provenance.json",dict(provenance,later_sha256=sha(out/"later_window_provenance.csv")))
    return evidence


def training_reports(out, all_models):
    training, costs, checksums = [], [], {}
    for name, record in all_models.items():
        directory = ROOT / record["directory"]
        borrowed = directory != out
        cfg = record["config"]
        assert record["config_hash"] == config_hash(cfg)
        assert cfg["seed"] == 42 and cfg["updates"] == record["final"]["step"] == 6000
        assert [r["step"] for r in record["training"]] == [r["step"] for r in record["states"]] == cfg["steps"]
        for weight, state in zip(record["training"],record["states"]):
            checkpoint, archive = directory/weight["checkpoint"], directory/state["path"]
            for path, row in ((checkpoint,weight),(archive,state)):
                assert sha(path) == row["sha256"]
                checksums[str(path.relative_to(ROOT))] = row["sha256"]
            assert isfinite(weight["train_mse"])
            if weight["step"]:
                assert all(isfinite(weight[k]) for k in ("loss","grad","postclip","clip_frequency"))
            if not borrowed and not state["borrowed"]:
                if is_extra(record):
                    snapshot = torch.load(archive,map_location="cpu",weights_only=False)
                    assert snapshot["config_hash"] == record["config_hash"] and snapshot["schedule"] == cfg["schedule"]
                    assert snapshot["step"] == snapshot["scheduler"]["last_epoch"] == state["step"]
                    assert set(snapshot["rng"]) == {"python","numpy","torch","cuda"}
                    expected = {state["step"]} if state["step"] else set()
                    assert {int(v["step"]) for v in snapshot["optimizer"]["state"].values()} == expected
                    assert state_digest(snapshot["model"]) == state_digest(torch.load(checkpoint,map_location="cpu",weights_only=True))
                    if not state["step"]:
                        source = ROOT / cfg["source"]["path"]
                        assert source != checkpoint and sha(source) == cfg["source"]["sha256"]
                        assert state_digest(snapshot["model"]) == state_digest(torch.load(source,map_location="cpu",weights_only=True))
                else:
                    verify_state(out,cfg,state,checkpoint)
            training.append(dict(weight,arm=name,archive=str(archive.relative_to(ROOT)),reused=borrowed))
        if not is_extra(record) and cfg["shared"]:
            has_history = cfg["model"]["history"] > 0
            if name.startswith("outer_"):
                region = name.split("_")[1]
                base_name = f"outer_{region}_{'IB' if has_history else 'M0'}"
                if has_history:
                    counterpart = read_json(STRUCTURE/"structured_fits"/f"{base_name}.json")
                    shared = STRUCTURE/f"shared_outer_{region}_IB.pt"
                else:
                    counterpart = all_models[base_name]
                    original = next(row for row in counterpart["states"] if row["step"]==1000)
                    shared = ROOT/counterpart["directory"]/original["path"]
            else:
                counterpart = all_models["IB" if has_history else "M0"]
                original = next(row for row in counterpart["states"] if row["step"]==1000)
                shared = STRUCTURE/"shared_IB.pt" if has_history else ROOT/counterpart["directory"]/original["path"]
            assert sha(shared)==cfg["shared"]
            for field in ("training","states"):
                own_prefix = [(r["step"],r["sha256"]) for r in record[field] if r["step"]<=1000]
                original_prefix = [(r["step"],r["sha256"]) for r in counterpart[field] if r["step"]<=1000]
                assert own_prefix==original_prefix, (name,field)
            assert cfg["schedule"]==cfg["prefix_schedule"]==counterpart["config"]["schedule"]
        costs.append(dict(arm=name,reused=borrowed,training_seconds=record["elapsed_seconds"],
                          own_updates=record.get("cost",{}).get("own_updates",6000),
                          oof_teachers=0 if is_extra(record) else 4,checkpoint_directory=record["directory"]))
    cols = sorted(set().union(*(row.keys() for row in training)))
    table(out/"training.csv",[{k:row.get(k) for k in cols} for row in training])
    table(out/"training_cost.csv",costs)
    return checksums


def teacher_costs(out, lineage):
    main = []
    for fold in range(4):
        record = read_json(FEEDBACK/f"fit_oof_{fold}.json")
        main.append(dict(cohort="main",role="OOF",fold=fold,seconds=record["elapsed_seconds"],
                         updates=record["final"]["step"],reuse=True))
    outer = []
    folder = ROOT/"results/corrector_spacetime_validation"
    for region, meta in lineage["regimes"].items():
        for reference in meta["refs"]:
            fold = reference["fold"]
            record = read_json(folder/"fits"/f"{reference['run']}.json")
            outer.append(dict(cohort=region,role="OOF" if fold<4 else "holdout S3",fold=fold,
                              seconds=record["elapsed_seconds"],updates=record["final"]["step"],reuse=True))
    table(out/"teacher_training_cost.csv",main+outer)
    return sum(row["seconds"] for row in main)


def audit_case_roles(out, evidence):
    triplets = read_json(out/"triplet_manifest.json")
    accumulated = set()
    backbone = None
    for split, n_windows, n_tracks in (("train",822,46),("development",24,12),("test",24,12)):
        cases = triplets[split]["cases"]
        files = {c["file"] for c in cases}
        assert len(cases)==n_windows and len(files)==n_tracks and files.isdisjoint(accumulated)
        accumulated.update(files)
        keys = []
        for case in cases:
            start = case["start"]
            keys.append([case["file"],start])
            assert case["earlier"]==[start-20,start] and case["history"]==[start,start+20]
            assert case["target"]==[start+20,start+40] and case["issue_frame"]==start+20
            assert case["score_frame"]==start+40
        teacher = read_json(out/f"teacher_{split}.json")
        assert teacher["keys"]==keys
        if split=="train":
            assert len(teacher["refs"])==4
            ids = []
            for row in teacher["refs"]:
                predicted = set(row["predicted"])
                assert set(row["train"])==files-predicted
                assert predicted=={keys[i][0] for i in row["ids"]}
                ids.extend(row["ids"])
            assert sorted(ids)==list(range(n_windows))
        else:
            ref, = teacher["refs"]
            assert ref["scope"]=="full"
            assert ref["sha256"]==evidence["frozen_s3_sha256"]
        recipe = read_json(out/f"native_{split}.json")["added_recipe"]
        assert recipe["zero_initial_noise"] and recipe["denoising_labels"]==[99,66,33,0]
        if backbone is None:
            backbone=recipe["backbone"]
        assert backbone==recipe["backbone"]
    return backbone


def complete(study,figures,frozen,outer,evidence):
    out = study.out
    assert guard(out)==frozen
    for key in ("zero_increment_bitwise","public_loader_bitwise","future_bitwise","backbone_frozen","s3_frozen"):
        assert evidence["publication"][key]
    assert all(evidence["publication"]["cache_invariance"].values())
    replay = evidence["publication"]["cache_replay"]
    assert replay["cached_teacher_bitwise"]
    assert all(replay[key]<=replay["absolute_tolerance"] for key in (
        "native_max_abs_error","s3_correction_max_abs_error","final_max_abs_error"))
    evidence["frozen_s3_sha256"] = frozen["source_s3"]["sha256"]
    evidence["backbone_identity"] = audit_case_roles(out,evidence)
    assert evidence["identity_max_error"] < 1e-15 and evidence["delta_identity_max_error"] < 1e-15
    assert all(row["bitwise"] for row in evidence["future_perturbation"])
    assert len(evidence["future_perturbation"])==2*len(frozen["models"])
    assert max(row["absolute_error"] for row in evidence["independent_recomputation"])<1e-15
    digest = sha(out/"FROZEN.json")
    assert read_json(out/"native_test.json")["test_freeze_sha256"]==digest
    all_models = dict(frozen["models"],**outer)
    hashes = training_reports(out,all_models)
    evidence.update(freeze_sha256=digest,loss_math=verify_math(),runtime=dict(
        pytorch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(), torch_threads=torch.get_num_threads(),
        deterministic_cudnn=torch.backends.cudnn.deterministic,
        tf32_cudnn=torch.backends.cudnn.allow_tf32, tf32_matmul=torch.backends.cuda.matmul.allow_tf32))
    assert evidence["runtime"]["deterministic_cudnn"]
    assert not evidence["runtime"]["tf32_cudnn"] and not evidence["runtime"]["tf32_matmul"]
    cal = read_json(out/"calibration.json")
    diagnostics = read_rows(out/"gradient_diagnostics.csv")
    for name, rows in groups(diagnostics,("arm","step","term")).items():
        assert len(rows)==len(cal["batches"])
        fields = ("loss","norm","cosine_to_mse","total_norm","conv1","conv2","conv3","clip_factor")
        assert all(isfinite(float(row[field])) for row in rows for field in fields)
    for family in ("TM1","TM2","EN"):
        for name,rho in zip(frozen["selection"]["matrix"][family],(.1,.3)):
            rows = [row for row in diagnostics if row["arm"]==name and row["step"]=="1000" and row["term"]!="mse"]
            ratio = sum(float(r["norm"])*float(r["weight"]) for r in rows)/sum(float(r["mse_norm"]) for r in rows)
            assert abs(ratio-rho)<1e-6
    scores = {(r["split"],r["policy"],r["arm"]):float(r["mse"]) for r in read_rows(out/"scores.csv")}
    selection = frozen["selection"]
    terms = {"M0":None,"TM1":"match","TM2":"match2","EN":"energy"}
    def fixed_dev(name):
        return scores["development","final",name],name
    for family,names in selection["matrix"].items():
        assert len(names)==(1 if family=="M0" else 2)
        assert selection["families"][family]==min(names,key=fixed_dev)
        for i,name in enumerate(names):
            model = frozen["models"][name]
            expected = {} if family=="M0" else {terms[family]:cal["coefficients"][family][("0.1","0.3")[i]]}
            assert model["config"]["model"] == dict(family="I",history=0,weights=expected,
                                                      rank=20,mix="original",sequence=None)
            for policy in ("final","selected"):
                assert abs(scores["development",policy,name]-model[policy]["dev_mse"])<2e-10
    assert selection["candidate"]==min(selection["families"].values(),key=fixed_dev)
    assert selection["best_time"]==min((selection["families"][key] for key in ("TM1","TM2")),key=fixed_dev)
    for key, rows in groups(read_rows(out/"leads.csv"),("split","policy","arm")).items():
        assert sorted(int(r["lead"]) for r in rows)==list(range(1,21))
        assert abs(mean(float(r["mse"]) for r in rows)-scores[key])<1e-15
    assert len(scores)==evidence["scores"]
    source_scores = read_rows(FACTORIAL/"scores.csv")
    signed = read_json(FACTORIAL/"artifact_audit.json")
    assert signed["files"]["scores.csv"]==sha(FACTORIAL/"scores.csv")
    aliases = {"I0":"M0","I0_match2":"I0_match2_fixed"}
    aliases["I0_match2_rho"] = ("I0_match2_legacy_rho" if "I0_match2_legacy_rho" in frozen["models"]
                               else frozen["selection"]["matrix"]["TM2"][1])
    replay = []
    for row in source_scores:
        key = row["split"],row["policy"],aliases.get(row["arm"],row["arm"])
        if key in scores:
            drift = abs(scores[key]-float(row["mse"]))
            assert drift<2e-10,(key,drift)
            replay.append(dict(split=key[0],policy=key[1],arm=key[2],absolute_mse_difference=drift))
    table(out/"historical_reproduction.csv",replay)
    evidence["historical_reproduction_rows"] = len(replay)
    mechanism = []
    for row in read_rows(out/"scores.csv"):
        mechanism.append(dict(split=row["split"],policy=row["policy"],arm=row["arm"],
            total_global_cosine=row["global_cosine"],total_window_mean_cosine=row["cosine"],
            total_global_energy_ratio=row["energy_ratio"],total_window_mean_rms_ratio=row["rms_ratio"],
            increment_global_cosine=row["global_delta_cosine"],increment_window_mean_cosine=row["delta_cosine"],
            increment_global_energy_ratio=float(row["U_delta"])/float(row["s3_mse"]),
            increment_window_mean_rms_ratio=row["window_delta_rms_ratio"]))
    table(out/"direction_energy_summary.csv",mechanism)
    lineage = read_json(out/"outer_lineage.json")
    assert lineage["freeze_sha256"]==digest and len(lineage["regimes"])==3
    evidence["original_main_oof_training_seconds"] = teacher_costs(out,lineage)
    for region, record in lineage["regimes"].items():
        held = set(record["train_removed"])
        assert held == set(record["backbone_seen"])
        registered = read_json(out/"outer_plan.json")[region]
        active = set(registered["active_files"])
        assert held==set(registered["held_files"]) and active.isdisjoint(held)
        assert len(record["refs"])==5
        for ref in record["refs"]:
            assigned = set(ref["held_files"])
            assert set(ref["train_files"])==active-assigned
            assert {file for file,_ in ref["target_windows"]}==assigned
            assert (assigned==held) if ref["fold"]==4 else assigned<=active
            assert sha(ROOT/ref["verified_checkpoint"])==ref["final_sha256"]
        for family,name in frozen["selection"]["families"].items():
            item = record["candidates"][name]
            assert item["config"]["model"]==frozen["models"][name]["config"]["model"]
            assert item["config"]["scale"]==record["scale"] and item["final"]["step"]==6000
            assert item["config"]["inputs"]["development"] is None
        for item in record["candidates"].values():
            assert item["target_invariance"]
        if frozen["selection"]["extra_outer"]:
            source = record["candidates"]["S3_extra_MSE"]["config"]["source"]
            teacher = next(ref for ref in record["refs"] if ref["fold"]==4)
            assert source["path"]==teacher["verified_checkpoint"] and source["sha256"]==teacher["final_sha256"]
    new_main = sum(item["directory"].endswith(out.name) for item in frozen["models"].values())
    new_outer = sum(item["directory"].endswith(out.name) for item in outer.values())
    assert new_main==frozen["main_new"] and new_outer<=frozen["outer_max"] and new_main+new_outer<=30
    counts = dict(new_main=new_main,new_outer=new_outer,reused_main=len(frozen["models"])-new_main,
                  reused_outer=len(outer)-new_outer,reused_teachers=20)
    atomic_json(out/"verification.json",dict(evidence,training=counts))
    from .report import report, plot

    report(out,frozen)
    plot(out,figures,frozen)
    image_hashes = {}
    for image in sorted(figures.glob("*.png")):
        with Image.open(image) as picture:
            assert picture.format=="PNG" and min(picture.size)>300
            picture.verify()
        image_hashes[image.name] = sha(image)
    assert len(image_hashes)==9
    files = {str(p.relative_to(out)):sha(p) for p in out.rglob("*")
             if p.is_file() and p.suffix in (".md",".json",".csv",".pt",".gz") and p.name!="artifact_audit.json"}
    source_paths = list(Path(__file__).parent.glob("*.py")) + [ROOT/"research/corrector_incremental_structure/fit.py"]
    sources = {str(p.relative_to(ROOT)):sha(p) for p in sorted(source_paths)}
    atomic_json(out/"artifact_audit.json",dict(passed=True,freeze_sha256=digest,files=files,
                figures=image_hashes,checkpoints=hashes,source_files=sources,training=counts))
    print("AUDIT COMPLETE",counts,flush=True)
