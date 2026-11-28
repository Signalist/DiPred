"""Causal, cache-aware publication and per-stage release-cost checks."""
from statistics import median
from time import perf_counter
from types import SimpleNamespace
from typing import NamedTuple

import torch

from research.corrector.run import sha
from research.corrector_history_free_refinement.deploy import Publisher
from research.corrector_incremental_factorial.core import load_head
from research.corrector_incremental_structure.model import inputs,residual
from research.corrector_incremental_structure.teachers import full_teacher
from research.corrector_residual_feedback.data import LABELS
from research.corrector_residual_feedback.model import base
from research.corrector_simple_baseline.fit import table
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_window_study.data import ROOT,read_json,tensor_sha


class Release(NamedTuple):
    forecast: torch.Tensor
    native: torch.Tensor
    s3: torch.Tensor


class TransferPublisher:
    def __init__(self, native, net, cfg, fields):
        self.native_model=native
        self.net=net
        self.cfg=cfg
        self.fields=fields

    def __call__(self, observed, previous_native=None, previous_s3=None, earlier=None):
        return self.issue(observed,previous_native,previous_s3,earlier).forecast

    @torch.inference_mode()
    def issue(self, observed, previous_native=None, previous_s3=None, earlier=None, history_available=True):
        """Publish the forecast and both frozen reference fields for the next release."""
        if observed.shape!=(1,2,20,32,64) or observed.dtype!=torch.float32:
            raise ValueError("Expected one FP32 observed window of shape [1, 2, 20, 32, 64].")
        forecast=self.native_model.native(observed)
        design=base(observed,forecast,self.native_model.norm)
        current=self.native_model.correct(design)
        if self.net is None:
            return Release(forecast.double()+current.double(),forecast,current)
        if not history_available:
            if not self.cfg.get("paired_history",False):
                raise ValueError("An explicit missing-history state requires a paired head.")
            view=SimpleNamespace(base=design,current=current,scale=self.native_model.residual_scale)
            features=inputs(view,slice(None),self.cfg,history_available=False)
            corrected=forecast.double()+residual(view,slice(None),self.cfg,self.net(features))
            return Release(corrected,forecast,current)
        needs_native=bool(set(self.fields)&{"R","E"})
        needs_teacher=bool(set(self.fields)&{"P","E"})
        if (needs_native and previous_native is None) or (needs_teacher and previous_s3 is None):
            if earlier is None:
                raise ValueError("Missing prior release cache and earlier observation.")
            if previous_native is None:
                previous_native=self.native_model.native(earlier)
            if needs_teacher and previous_s3 is None:
                previous_s3=self.native_model.correct(base(earlier,previous_native,self.native_model.norm))
        actual=observed.double()-previous_native.double() if needs_native else None
        history=previous_s3 if needs_teacher else None
        # Explicit slot configs need both algebraic sources even when the head reads just P.
        view=SimpleNamespace(base=design,current=current,scale=self.native_model.residual_scale,
                             true=actual if actual is not None else torch.zeros_like(current,dtype=torch.float64),
                             historical=history if history is not None else torch.zeros_like(current))
        ix=slice(None)
        corrected=forecast.double()+residual(view,ix,self.cfg,self.net(inputs(view,ix,self.cfg)))
        return Release(corrected,forecast,current)


def load_publisher(arm="S3", policy="final"):
    """Load one registered release model without access to trajectory data."""
    from reliability.backbone import configure
    from reliability.networks import DynamicsDiffusion20
    from research.corrector_residual_feedback.data import PRIOR
    from research.corrector_residual_feedback.run import references
    from research.corrector_residual_optimization.fit import load

    from .run import OUT,STRUCTURE

    configure()
    if policy not in ("final","selected"):
        raise ValueError("Choose the final or selected checkpoint policy.")
    freeze=read_json(OUT/"FROZEN.json")
    lineage_path=OUT/"reference_lineage.json"
    plan_path=OUT/"experiment_plan.json"
    for path in (lineage_path,plan_path):
        assert sha(path)==freeze["bindings"][path.name]
    lineage=read_json(lineage_path)
    plan=read_json(plan_path)
    device=torch.device("cuda")
    native_path=ROOT/"reliability/checkpoints/current.pt"
    assert sha(native_path)==lineage["backbone"]["packaged_checkpoint_sha256"]
    stored=torch.load(native_path,map_location=device,weights_only=False)
    assert stored["format_version"]==1
    backbone=DynamicsDiffusion20(stored["dynamics_scale"]).to(device)
    backbone.load_state_dict(stored["predictor"],strict=True)
    teacher=references()[0]
    reference=lineage["development"]["refs"][0]
    assert teacher["selected"]["sha256"]==reference["sha256"]
    net=load(teacher["config"],teacher["selected"],PRIOR,device)
    pipeline=Publisher(backbone,net,stored["normalization"],reference["native_rms"],lineage["train"]["scale"])
    if arm=="S3":
        return TransferPublisher(pipeline,None,None,[])
    record=freeze["models"][arm]
    temporal=STRUCTURE/"temporal_basis.pt"
    assert sha(temporal)==freeze["basis_sha256"]
    basis=torch.load(temporal,map_location="cpu",weights_only=True)
    head=load_head(record,record[policy],basis,device).eval().requires_grad_(False)
    return TransferPublisher(pipeline,head,record["config"]["model"],plan["inputs"][arm.split("_")[0]])


def timing(fn, device, repeat=21):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(device)
    initial=torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples=[]
    for _ in range(repeat):
        start=perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append((perf_counter()-start)*1000)
    return dict(ms_median=median(samples),ms_min=min(samples),ms_max=max(samples),
                extra_peak_bytes=torch.cuda.max_memory_allocated(device)-initial)


@torch.inference_mode()
def publication_checks(study,view,freeze,basis,plan):
    i,=[i for i,c in enumerate(view.data.cases) if (c["file"],c["start"])==(plan["field"]["file"],plan["field"]["start"])]
    data=view.data
    case=data.cases[i]
    assert case["earlier"]==[case["start"]-20,case["start"]]
    assert case["history"]==[case["start"],case["issue_frame"]]
    assert case["target"]==[case["issue_frame"],case["score_frame"]]
    ix=slice(i,i+1)
    observed,earlier=data.x[ix],data.previous_x[ix]
    prev_native,prev_s3=data.previous_p[ix],view.historical[ix]
    teacher,sha_s3=full_teacher(study,observed.device)
    backbone=study.triplets.source.backbone
    pipeline=Publisher(backbone,teacher,study.norm,study.cfg["external_rms"],view.scale)
    native=pipeline.native(observed)
    z=base(observed,native,study.norm)
    current=pipeline.correct(z)
    audit={}
    costs=[]
    direct=TransferPublisher(pipeline,None,None,[])
    backbone_params=sum(x.numel() for x in backbone.parameters())
    s3_params=sum(x.numel() for x in teacher.parameters())
    for mode, fn in (("Native",lambda:pipeline.native(observed)),("S3",lambda:pipeline.correct(z)),
                     ("end_to_end",lambda:direct(observed))):
        costs.append(dict(arm="S3",mode=mode,head_params=0,total_params=backbone_params+s3_params,
                          requires_prior_native=False,requires_prior_s3=False,**timing(fn,observed.device)))
    names={f"{arm}_MSE" for arm in plan["inputs"]}|{f"{arm}_{term}" for arm in plan["focal"] for term in plan["terms"]}
    names.update(freeze["selection"][key] for key in ("final","selected"))
    for arm in sorted(names):
        record=freeze["models"][arm]
        cfg=record["config"]["model"]
        key=arm.split("_")[0]
        slots=plan["inputs"][key]
        head=load_head(record,record["final"],basis,observed.device).eval().requires_grad_(False)
        pub=TransferPublisher(pipeline,head,cfg,slots)
        calls=[]
        def hook(_,args):
            if len(calls)%4==0:
                assert torch.count_nonzero(args[0])==0
            calls.append(int(args[1].item()))
        handle=backbone.net.register_forward_pre_hook(hook)
        try:
            release=pub.issue(observed,prev_native,prev_s3)
            cached=release.forecast
            assert torch.equal(release.native,native) and torch.equal(release.s3,current)
            assert calls==LABELS
            calls.clear()
            cold=pub(observed,earlier=earlier)
            assert calls==LABELS*(2 if slots else 1)
        finally:
            handle.remove()
        assert torch.isfinite(cached).all() and torch.isfinite(cold).all()
        scale=view.scale
        live_view=SimpleNamespace(base=z,current=current,scale=scale,true=observed.double()-prev_native.double(),
                                  historical=prev_s3)
        features=inputs(live_view,slice(None),cfg)
        expected=native.double()+residual(live_view,slice(None),cfg,head(features))
        assert torch.equal(cached,expected)
        zero=load_head(record,record["training"][0],basis,observed.device).eval()
        assert torch.equal(TransferPublisher(pipeline,zero,cfg,slots)(observed,prev_native,prev_s3),
                           direct(observed))
        before=data.y
        try:
            data.y=torch.zeros_like(before)
            assert torch.equal(pub(observed,prev_native,prev_s3),cached)
        finally:
            data.y=before
        size=sum(x.numel() for x in head.parameters())
        path_modes=[("head",lambda:head(features)),("hot",lambda:pub(observed,prev_native,prev_s3))]
        if slots:
            path_modes.append(("cold",lambda:pub(observed,earlier=earlier)))
        for mode,fn in path_modes:
            costs.append(dict(arm=arm,mode=mode,head_params=size,total_params=size+backbone_params+s3_params,
                              requires_prior_native=bool(set(slots)&{"R","E"}),
                              requires_prior_s3=bool(set(slots)&{"P","E"}),**timing(fn,observed.device)))
        audit[arm]=dict(hot_identity=tensor_sha(cached),hot_calls=LABELS,cold_calls=calls,
                        cacheless_max_abs=float((cached-cold).abs().max()),zero_increment_bitwise=True,
                        altered_future_bitwise=True)
    portable_releases={}
    for policy in ("final","selected"):
        arm=freeze["selection"][policy]
        saved=freeze["models"][arm]
        head=load_head(saved,saved[policy],basis,observed.device).eval()
        local=TransferPublisher(pipeline,head,saved["config"]["model"],plan["inputs"][arm.split("_")[0]])
        portable=load_publisher(arm,policy)
        expected=local.issue(observed,prev_native,prev_s3)
        issued=portable.issue(observed,prev_native,prev_s3)
        assert all(torch.equal(a,b) for a,b in zip(issued,expected))
        portable_releases[policy]=dict(arm=arm,bitwise=True,forecast_sha256=tensor_sha(issued.forecast),
                                       native_sha256=tensor_sha(issued.native),s3_sha256=tensor_sha(issued.s3))
    actual=observed.double()-prev_native.double()
    remaining=actual-prev_s3.double()
    assert (remaining+prev_s3.double()-actual).abs().max()<1e-15
    reconstructed=native.double()+current.double()+(data.y[ix].double()-native.double()-current.double())
    error=float((reconstructed-data.y[ix].double()).abs().max())
    assert error<1e-15 and not any(p.requires_grad for p in backbone.parameters())
    table(study.out/"inference_cost.csv",costs)
    atomic_json(study.out/"publication_verification.json",dict(case=case,s3=sha_s3,algebra=True,restoration_max_error=error,
                freeze_sha256=sha(study.out/"FROZEN.json"),gpu=torch.cuda.get_device_name(),repeat=21,warmup=5,
                backbone_frozen=True,s3_frozen=not any(p.requires_grad for p in teacher.parameters()),
                portable_releases=portable_releases,models=audit))
