"""Regime stress tests with fresh trajectory-isolated S3 reference lineages."""
from types import SimpleNamespace

import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.data import guard
from research.corrector_residual_feedback.model import base,predict
from research.corrector_residual_feedback.run import rms,save_plan
from research.corrector_residual_optimization.fit import shared_plan
from research.corrector_window_study.data import WORKSPACE,read_json,tensor_sha

from .data import ROOT
from .expansion import feedback_inputs
from .fit import Fits,atomic_json,load_net


def subset(data,ids):
    return SimpleNamespace(cases=[data.cases[i] for i in ids],**{
        k:getattr(data,k)[ids] for k in ("p","x","y","previous_x","previous_p")})


def backbone_lineage(study):
    protocol = WORKSPACE/"runs/physical20_augmented_dynamics_20261109/protocol.json"
    manifest = ROOT/"metadata/manifest.json"
    data = read_json(protocol)
    assert sha(manifest)==data["manifest_sha256"]
    tracks = sorted(r["file"] for r in read_json(manifest)["files"]["train"])
    assert len(tracks)==data["fit_trajectories"]==47
    assert all(set(spec["held_files"])<=set(tracks) for spec in study.splits.values())
    return dict(sha256=sha(protocol),manifest_sha256=sha(manifest),training_files=tracks,
                training_frames=data["fit_frames"],checkpoint=study.triplets.source.identity,
                interpretation="frozen-backbone Corrector regime stress test; backbone trained on held trajectories")


def outer_models(study):
    frozen = guard(study.out)
    assert read_json(study.out/"outer_split_manifest.json")==study.splits
    train,current,design = study.get("train")
    del current,design
    out = study.out
    jobs = []
    lineage = dict(freeze_sha256=sha(out/"FROZEN.json"),backbone=backbone_lineage(study),regimes={})
    for label,split in study.splits.items():
        held = subset(train,split["held"])
        active = subset(train,split["active"])
        active_files,held_files = set(split["active_files"]),set(split["held_files"])
        assert active_files.isdisjoint(held_files)
        schedule = shared_plan(active)
        assert tensor_sha(schedule)==split["sampling_sha256"]
        scale = rms(active)
        branch_pred = torch.empty_like(active.x)
        histories = base(active.previous_x,active.previous_p,study.norm)
        chronology = dict(train_files=sorted(active_files),held_files=sorted(held_files),train_scale=scale,
                          train_native_sha256=tensor_sha(active.p),held_native_sha256=tensor_sha(held.p),
                          references=[],candidates={})
        for recipe in split["reference_plan"]:
            fold = recipe["fold"]
            files = set(recipe["training_files"])
            exclusions = active_files-files
            assert sorted(exclusions or held_files)==recipe["prediction_files"]
            subset_ids = [i for i,c in enumerate(active.cases) if c["file"] in files]
            reference_train = subset(active,subset_ids)
            assert set(c["file"] for c in reference_train.cases)==files
            assert (held_files|exclusions).isdisjoint(files)
            scale_ref = rms(reference_train)
            ids = [i for i,c in enumerate(active.cases) if c["file"] in exclusions]
            name = f"outer_{label}_reference_{fold}"
            reference_plan = shared_plan(reference_train)
            assert tensor_sha(reference_plan)==recipe["sampling_sha256"]
            entry = save_plan(out,name,reference_plan)
            current = base(reference_train.x,reference_train.p,study.norm)
            engine = Fits(out,study.cache,reference_train,current,scale_ref,reference_plan,
                          study.initial,reference_layout=True)
            record = engine.fit(name,"S3",{},origin=dict(outer=label,reference=fold,files=sorted(files)))
            row = record["final"]
            net = load_net(out/row["checkpoint"],row["sha256"],"S3",train.p.device,"NCDHW")
            if exclusions:
                branch_pred[ids] = predict(net,histories[ids],scale_ref)
                predicted = branch_pred[ids]
                issued_keys = [[active.cases[i]["file"],active.cases[i]["start"]-20] for i in ids]
            else:
                s3_prediction = predict(net,base(held.x,held.p,study.norm),scale_ref)
                held_prediction = predict(net,base(held.previous_x,held.previous_p,study.norm),scale_ref)
                predicted = held_prediction
                issued_keys = [[c["file"],c["start"]-20] for c in held.cases]
            chronology["references"].append(dict(run=name,fold=fold,train_files=sorted(files),held_files=sorted(exclusions or held_files),
                                                  scale=scale_ref,schedule=entry,final_sha256=row["sha256"],
                                                  prediction_sha256=tensor_sha(predicted),prediction_ids=issued_keys))
            del reference_train,current,engine,net
        assert torch.isfinite(branch_pred).all()
        assert sorted(k for f in chronology["references"][:4] for k in f["held_files"])==sorted(active_files)
        chronology["oof_physical_sha256"] = tensor_sha(branch_pred)
        chronology["held_physical_sha256"] = tensor_sha(held_prediction)
        shared_schedule = save_plan(out,f"outer_{label}",schedule)
        assessments = {"Native":torch.zeros_like(held.p),"S3":s3_prediction}
        common_train,common_test = (base(d.x,d.p,study.norm) for d in (active,held))
        true_train = active.x.double()-active.previous_p.double()
        true_held = held.x.double()-held.previous_p.double()
        for b in ("B","E"):
            z = feedback_inputs(common_train,true_train,branch_pred,b,scale)
            dz = feedback_inputs(common_test,true_held,held_prediction,b,scale)
            shared = out/f"shared_outer_{label}_{b}.pt"
            engine = Fits(out,study.cache,active,z,scale,schedule,study.initial)
            for tag,key in frozen["selection"]["outer"].items():
                if not tag.startswith(b+"_"):
                    continue
                cfg = frozen["models"][key]["config"]
                weights = cfg["weights"]
                assert cfg["branch"]==b
                record = engine.fit(f"outer_{label}_{tag}",b,weights,shared=shared if weights else None,
                                    save_shared=shared if not weights else None,origin=dict(outer=label,model=key,
                                                                                           files=sorted(active_files)))
                row = record["final"]
                net = load_net(out/row["checkpoint"],row["sha256"],b,held.p.device)
                pre = predict(net,dz,scale)
                shuffled = held.y
                held.y = shuffled.roll(1,0)
                try:
                    current = base(held.x,held.p,study.norm)
                    rebuilt = feedback_inputs(current,held.x.double()-held.previous_p.double(),held_prediction,b,scale)
                    assert torch.equal(pre,predict(net,rebuilt,scale))
                finally:
                    held.y = shuffled
                assessments[tag] = pre
                chronology["candidates"][tag] = dict(frozen_from=key,weights=weights,step=row["step"],
                                                       checkpoint=row["checkpoint"],sha256=row["sha256"],scale=scale,
                                                       train_input_sha256=tensor_sha(z),held_input_sha256=tensor_sha(dz),
                                                       future_perturbation_bitwise=True,
                                                       schedule=shared_schedule,elapsed_seconds=record["elapsed_seconds"])
            del engine,z,dz
        assert len(assessments)==10
        jobs.append((label,held,assessments))
        lineage["regimes"][label] = chronology
        atomic_json(out/"reference_lineage.json",lineage)
        print("OUTER COMPLETE",label,len(active.cases),len(held.cases),flush=True)
        del active,branch_pred,histories,common_train,true_train
    return jobs,lineage
