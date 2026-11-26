"""Causal matched present/past teachers and strictly isolated outer reuse."""
from collections import defaultdict
from types import SimpleNamespace

import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.data import OUT as FEEDBACK,write_tensor,guard
from research.corrector_residual_feedback.model import base,predict
from research.corrector_residual_feedback.run import checked_model,rms,references
from research.corrector_residual_optimization.fit import load as load_s3,shared_plan,track_weights
from research.corrector_spacetime_validation.data import OUT as VALIDATION
from research.corrector_spacetime_validation.fit import atomic_json,load_net
from research.corrector_spacetime_validation.outer import subset
from research.corrector_window_study.data import read_json,tensor_sha
from research.corrector_residual_feedback.data import PRIOR as S3_ROOT


def full_teacher(study,device):
    cfg,_,_=references()
    assert cfg["selected"]["sha256"]==study.prior["reference"]["selected"]["sha256"]
    net=load_s3(cfg["config"],cfg["selected"],S3_ROOT,device).eval().requires_grad_(False)
    return net,cfg["selected"]["sha256"]


def bundle(data,base_fields,current,historical,scale,identity):
    assert current.shape==historical.shape==data.p.shape
    past=data.x.double()-data.previous_p.double()
    target=data.y.double()-data.p.double()
    assert torch.isfinite(current).all() and torch.isfinite(historical).all()
    assert (data.p.double()+current.double()+target-current.double()-data.y.double()).abs().max()<1e-14
    return SimpleNamespace(data=data,base=base_fields,current=current,historical=historical,true=past,
                           target=target,scale=scale,identity=identity)


def distribution(view,prediction):
    w=track_weights(view.data)
    quant=[]
    for src in (prediction.double(),view.target-prediction.double()):
        energy=src.square().mean((1,2,3,4))
        per=energy.sqrt()
        quant.append(dict(rms=float((energy*w).sum().sqrt()),
                          per_window_rms_quantiles=per.quantile(per.new_tensor([.1,.5,.9])).tolist()))
    return dict(prediction=quant[0],remaining=quant[1])


@torch.inference_mode()
def main_bundle(study,data,features,split):
    identity=dict(split=split,keys=[[c["file"],c["start"]] for c in data.cases],native=tensor_sha(data.p),
                  observed=tensor_sha(data.x),scale=study.scale,refs=[])
    model,full_hash=full_teacher(study,data.p.device)
    full_prediction=predict(model,features,study.cfg["external_rms"])
    if split=="train":
        current=torch.empty_like(data.p)
        historical=torch.empty_like(data.p)
        prior=base(data.previous_x,data.previous_p,study.norm)
        used=[]
        for row in study.crossfit["folds"]:
            ids=row["holdout_indices"]
            train,held=set(row["training_files"]),set(row["holdout_files"])
            assert train.isdisjoint(held) and all(data.cases[i]["file"] in held for i in ids)
            indices=row["training_indices"]
            assert {data.cases[i]["file"] for i in indices}==train
            fit=read_json(FEEDBACK/f"fit_oof_{row['fold']}.json")
            config=fit["config"]
            assert fit["final"]==row["reference"] and config["scale"]==row["rms"] and config["channels"]==6
            observed=dict(train_P=tensor_sha(data.p[indices]),train_Y=tensor_sha(data.y[indices]),
                          train_input=tensor_sha(features[indices]))
            assert all(config[k]==v for k,v in observed.items())
            sample=row["schedule"]
            assert config["schedule_sha256"]==sample["tensor_sha256"]
            assert sha(FEEDBACK/sample["file"])==sample["sha256"]
            net=checked_model(6,row["reference"],FEEDBACK,data.p.device).eval().requires_grad_(False)
            current[ids]=predict(net,features[ids],row["rms"])
            historical[ids]=predict(net,prior[ids],row["rms"])
            used+=ids
            identity["refs"].append(dict(fold=row["fold"],train=sorted(train),predicted=sorted(held),
                                         ids=ids,sha256=row["reference"]["sha256"],native_rms=row["rms"],
                                         training_inputs=observed,schedule_sha256=sample["tensor_sha256"]))
        assert sorted(used)==list(range(len(data.cases)))
        assert torch.equal(historical,data.historical_prediction)
    else:
        current,historical=full_prediction,data.historical_prediction
        identity["refs"]=[dict(scope="full",sha256=full_hash,native_rms=study.cfg["external_rms"])]
    identity.update(current=tensor_sha(current),historical=tensor_sha(historical))
    views=bundle(data,features,current,historical,study.scale,identity)
    identity["distribution"]={"used":distribution(views,current),"full":distribution(views,full_prediction)}
    path=study.out/f"teacher_{split}.json"
    if split in ("train","development") and (study.out/"FROZEN.json").is_file():
        assert read_json(path)==identity,path
    else:
        atomic_json(path,identity)
    return views


@torch.inference_mode()
def outer_bundles(study,train,label):
    out=study.out
    original=read_json(VALIDATION/"outer_split_manifest.json")[label]
    lineage=read_json(VALIDATION/"reference_lineage.json")["regimes"][label]
    prior_audit=read_json(VALIDATION/"artifact_audit.json")
    assert prior_audit["passed"] and prior_audit["freeze_sha256"]==sha(VALIDATION/"FROZEN.json")
    held,active=(subset(train,original[k]) for k in ("held","active"))
    assert tensor_sha(active.p)==lineage["train_native_sha256"] and tensor_sha(held.p)==lineage["held_native_sha256"]
    scale=rms(active)
    assert scale==lineage["train_scale"]
    train_files,held_files=set(original["active_files"]),set(original["held_files"])
    bx,bh=(base(x.x,x.p,study.norm) for x in (active,held))
    prev=base(active.previous_x,active.previous_p,study.norm)
    current,historical=(torch.empty_like(active.p) for _ in range(2))
    used=[]
    refs=[]
    for row,recipe in zip(lineage["references"],original["reference_plan"]):
        files=set(row["train_files"])
        test=set(row["held_files"])
        assert files.isdisjoint(held_files|test) and row["train_files"]==recipe["training_files"]
        assert row["held_files"]==recipe["prediction_files"]
        record=read_json(VALIDATION/"fits"/f"{row['run']}.json")
        pointer=record["final"]
        path=VALIDATION/pointer["checkpoint"]
        assert sha(path)==pointer["sha256"]==row["final_sha256"]
        assert prior_audit["checkpoints"][str(path.relative_to(VALIDATION.parents[1]))]==pointer["sha256"]
        assert record["config"]["layout"]=="NCDHW"
        assert record["config"]["origin"]["files"]==row["train_files"]
        assert record["config"]["scale"]==row["scale"] and record["config"]["schedule"]==recipe["sampling_sha256"]
        training_indices=[i for i,c in enumerate(active.cases) if c["file"] in files]
        assert {active.cases[i]["file"] for i in training_indices}==files
        labels=(active.y[training_indices]-active.p[training_indices])/row["scale"]
        training_inputs=dict(train=tensor_sha(bx[training_indices]),target=tensor_sha(labels),development=None)
        assert training_inputs==record["config"]["inputs"]
        net=load_net(path,row["final_sha256"],"S3",active.p.device,"NCDHW").eval().requires_grad_(False)
        if row["fold"]<4:
            ids=[i for i,c in enumerate(active.cases) if c["file"] in test]
            assert files==train_files-test
            current[ids]=predict(net,bx[ids],row["scale"])
            historical[ids]=predict(net,prev[ids],row["scale"])
            assert tensor_sha(historical[ids])==row["prediction_sha256"]
            used+=ids
        else:
            assert test==held_files and files==train_files
            hold_current=predict(net,bh,row["scale"])
            hold_previous=predict(net,base(held.previous_x,held.previous_p,study.norm),row["scale"])
            assert tensor_sha(hold_previous)==row["prediction_sha256"]==lineage["held_physical_sha256"]
        refs.append(dict(row,verified_checkpoint=str(path.relative_to(VALIDATION.parents[1])),
                         training_inputs=training_inputs,
                         target_windows=[[c["file"],c["start"]] for c in (held.cases if row["fold"]==4 else
                                          [active.cases[i] for i in ids])]))
    assert sorted(used)==list(range(len(active.cases)))
    assert tensor_sha(historical)==lineage["oof_physical_sha256"]
    sampling=shared_plan(active)
    assert tensor_sha(sampling)==original["sampling_sha256"]
    identity=dict(regime=label,phys_re=original["re"],refs=refs,train_removed=sorted(held_files),
                  scale=scale,train_current=tensor_sha(current),train_history=tensor_sha(historical),
                  held_current=tensor_sha(hold_current),held_history=tensor_sha(hold_previous),
                  backbone_seen=sorted(held_files & set(read_json(VALIDATION/"reference_lineage.json")["backbone"]["training_files"])))
    atomic_json(out/f"teacher_outer_{label}.json",identity)
    return bundle(active,bx,current,historical,scale,identity),bundle(held,bh,hold_current,hold_previous,scale,identity),sampling


def chains(cases,expected=3):
    by_track=defaultdict(dict)
    for i,c in enumerate(cases):
        assert c["start"] not in by_track[c["file"]]
        by_track[c["file"]][c["start"]]=i
    rows={}
    for file,trajectory in sorted(by_track.items()):
        windows=[]
        for start in sorted(trajectory):
            positions=[start+20*k for k in range(expected)]
            if all(p in trajectory for p in positions):
                windows.append([trajectory[p] for p in positions])
        if windows:
            rows[file]=windows
    assert set(rows)==set(by_track), "Each training trajectory must have a valid chain"
    return rows


def chain_plan(cases,updates=6000,batch=8):
    tracks=list(chains(cases).values())
    rng=torch.Generator().manual_seed(42)
    first=torch.randint(len(tracks),(updates,batch),generator=rng).tolist()
    second=torch.rand(updates,batch,generator=rng).tolist()
    return torch.tensor([[tracks[n][int(t*len(tracks[n]))] for n,t in zip(a,b)] for a,b in zip(first,second)])


def check_chain_plan(view,plan):
    data=view.data
    by_file=chains(data.cases)
    refs={i:r["fold"] for r in view.identity["refs"] for i in r["ids"]}
    expected={tuple(seq) for windows in by_file.values() for seq in windows}
    sampled=set(map(tuple,plan.reshape(-1,3).tolist()))
    assert sampled<=expected
    assert {data.cases[a]["file"] for a,b,c in sampled}==set(by_file)
    gap=0.
    for i,j in sorted({(a,b) for a,b,c in expected}|{(b,c) for a,b,c in expected}):
        assert refs[i]==refs[j]
        assert torch.equal(data.y[i],data.x[j])
        assert torch.equal(data.x[i],data.previous_x[j])
        assert torch.equal(data.p[i],data.previous_p[j])
        discrepancy=float((view.current[i]-view.historical[j]).abs().max())
        gap=max(gap,discrepancy)
        assert torch.allclose(view.current[i],view.historical[j],atol=1e-7,rtol=1e-6)
    return dict(eligible_chains=len(expected),sampled_chains=len(sampled),trajectories=len(by_file),
                max_adjacent_s3_difference=gap)
