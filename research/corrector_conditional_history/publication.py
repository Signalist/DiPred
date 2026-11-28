"""Measure the frozen causal release paths on the predeclared case."""
import torch

from research.corrector_history_free_refinement.deploy import Publisher
from research.corrector_incremental_factorial.core import load_head
from research.corrector_incremental_structure.teachers import full_teacher
from research.corrector_remaining_residual_transfer.publication import TransferPublisher, timing
from research.corrector_remaining_residual_transfer.publication import load_publisher as load_previous_publisher
from research.corrector_remaining_residual_transfer.run import STRUCTURE
from research.corrector_residual_feedback.data import guard
from research.corrector.run import sha

from .run import OUT
from research.corrector_residual_feedback.data import LABELS
from research.corrector_simple_baseline.fit import table
from research.corrector_window_study.data import tensor_sha


def load_publisher(arm="S3",policy="final"):
    """Load a registered reference or deployable head without trajectory access."""
    if policy not in ("final","selected"):
        raise ValueError("Choose final or selected.")
    release=load_previous_publisher("S3")
    freeze=guard(OUT)
    if arm=="S3":
        return release
    if arm not in freeze["selection"]["candidates"]:
        raise ValueError(f"{arm} is not a registered deployable candidate.")
    record=freeze["models"][arm]
    path=STRUCTURE/"temporal_basis.pt"
    assert sha(path)==freeze["basis_sha256"]
    basis=torch.load(path,map_location="cpu",weights_only=True)
    device=next(release.native_model.backbone.parameters()).device
    net=load_head(record,record[policy],basis,device).eval().requires_grad_(False)
    fields={"I0":[],"IB":["R"],"ID":["E"]}[arm.split("_")[0]]
    return TransferPublisher(release.native_model,net,record["config"]["model"],fields)


@torch.inference_mode()
def release_checks(evaluator):
    study=evaluator.study
    freeze=evaluator.freeze
    view,outputs=evaluator.example
    fixed=evaluator.plan["field"]
    i,=[j for j,c in enumerate(view.data.cases) if (c["file"],c["start"])==(fixed["file"],fixed["start"])]
    ix=slice(i,i+1)
    data=view.data
    x,earlier=data.x[ix],data.previous_x[ix]
    cached=(data.previous_p[ix],view.historical[ix])
    device=x.device
    teacher,teacher_sha=full_teacher(study,device)
    backbone=study.triplets.source.backbone
    runtime=Publisher(backbone,teacher,study.norm,study.cfg["external_rms"],view.scale)
    s3=TransferPublisher(runtime,None,None,[])
    candidates=dict.fromkeys(("S3",freeze["selection"]["final"],freeze["selection"]["selected"],"ID_TM2",
                              *freeze["selection"]["pair_outer"].values()))
    rows,proof=[],[]
    for arm in candidates:
        if arm=="S3":
            net=None
            paths={"s3_fallback":lambda:s3.issue(x)}
        else:
            record=freeze["models"][arm]
            cfg=record["config"]["model"]
            net=load_head(record,record["final"],evaluator.basis,device).eval().requires_grad_(False)
            family=arm.split("_")[0]
            fields={"I0":[],"IB":["R"],"ID":["E"]}[family]
            engine=TransferPublisher(runtime,net,cfg,fields)
            paths={"cached":lambda:engine.issue(x,*cached)}
            if fields:
                paths["recompute"]=lambda:engine.issue(x,earlier=earlier)
            if cfg.get("paired_history"):
                paths["missing"]=lambda:engine.issue(x,object(),object(),object(),history_available=False)
        for mode,issue in paths.items():
            calls=[]
            def hook(_,args):
                if len(calls)%4==0:
                    assert torch.count_nonzero(args[0])==0
                calls.append(int(args[1].item()))
            handle=backbone.net.register_forward_pre_hook(hook)
            try:
                result=issue()
            finally:
                handle.remove()
            assert calls==LABELS*(2 if mode=="recompute" else 1)
            assert torch.isfinite(result.forecast).all()
            before=data.y
            try:
                data.y=torch.zeros_like(before)
                assert torch.equal(result.forecast,issue().forecast)
            finally:
                data.y=before
            fields={"s3_fallback":"S3","cached":arm,"missing":arm+"__missing"}
            margin=0.
            if mode in fields:
                predicted=data.p[ix].double()+outputs[fields[mode]][ix]
                margin=float((result.forecast-predicted).abs().max())
                assert margin<1e-6
            metrics=timing(issue,device)
            rows.append(dict(arm=arm,mode=mode,head_params=sum(p.numel() for p in net.parameters()) if net else 0,
                example_mse=float((data.y[ix].double()-result.forecast).square().mean()),
                score_replay_max_abs=margin,gpu=torch.cuda.get_device_name(device),**metrics))
            proof.append(dict(arm=arm,mode=mode,labels=calls,output_sha256=tensor_sha(result.forecast),
                              altered_future_bitwise=True))
    arm=freeze["selection"]["final"]
    portable=load_publisher(arm)
    cfg=freeze["models"][arm]["config"]["model"]
    reference=load_head(freeze["models"][arm],freeze["models"][arm]["final"],evaluator.basis,device)
    fields={"I0":[],"IB":["R"],"ID":["E"]}[arm.split("_")[0]]
    local=TransferPublisher(runtime,reference,cfg,fields)
    assert all(torch.equal(a,b) for a,b in zip(portable.issue(x,*cached),local.issue(x,*cached)))
    table(study.out/"inference_cost.csv",rows)
    lead=fixed["lead"]-1
    paired=freeze["selection"]["pair_outer"]["ID"]
    fields=dict(history_R=view.true[i,:,lead],history_S3=view.historical[i,:,lead],
                history_E=(view.true-view.historical.double())[i,:,lead],
                current_E=(view.target-view.current.double())[i,:,lead],
                d_missing=(outputs[paired+"__missing"]-view.current.double())[i,:,lead],
                d_full=(outputs[paired]-view.current.double())[i,:,lead],
                delta_H=(outputs[paired]-outputs[paired+"__missing"])[i,:,lead],
                final_E=(view.target-outputs[paired])[i,:,lead])
    table(study.out/"fixed_fields.csv",[dict(field=name,channel=c,y=y,x=x,value=value)
          for name,z in fields.items() for c,plane in zip(("u","v"),z.tolist())
          for y,row in enumerate(plane) for x,value in enumerate(row)])
    return dict(case=fixed,teacher=teacher_sha,paths=proof,backbone_frozen=not any(p.requires_grad for p in backbone.parameters()),
                s3_frozen=not any(p.requires_grad for p in teacher.parameters()),paired_example=paired)
