"""Frozen objective comparisons, matched regime stress tests and publication checks."""
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from research.corrector.run import sha
from research.corrector_incremental_factorial.core import load_head, prefix, stamp, verify_reuse
from research.corrector_incremental_factorial.publishables import equal_regimes
from research.corrector_incremental_structure.fit import Fits, predict
from research.corrector_incremental_structure.score import IncrementalScores
from research.corrector_incremental_structure.teachers import outer_bundles
from research.corrector_incremental_structure.validation import (
    target_invariance,
)
from research.corrector_residual_feedback.data import LABELS, guard
from research.corrector_residual_feedback.model import base, predict as teacher_predict
from research.corrector_simple_baseline.fit import table
from research.corrector_spacetime_validation.fit import atomic_json, load_net
from research.corrector_window_study.data import ROOT, read_json

from .core import prior_study
from .deploy import Publisher, benchmark, load_publisher
from .extra import fit_extra
from .run import FACTORIAL, STRUCTURE


def is_extra(record):
    return "source" in record["config"]


def load_extra(record, policy, device):
    row = record[policy]
    path = ROOT / record["directory"] / row["checkpoint"]
    return load_net(path, row["sha256"], "S3", device, record["config"]["layout"])


def comparison_lists(selection):
    def pairs(split, present):
        result = [tuple(row) for row in selection["primary"]]
        for name in present:
            result.extend((control, name, control) for control in ("Native", "S3", "M0"))
        for names in zip(*(selection["matrix"][family] for family in ("TM1", "TM2", "EN"))):
            for a, b in ((0, 2), (1, 0), (1, 2)):
                result.append(("same_rho", names[a], names[b]))
        winner = selection["candidate"]
        for name in ("I0_match2_fixed", "IB", "IB_match2", "IB_match", "IB_energy", "S3_extra_MSE"):
            result.append(("engineering" if name == "S3_extra_MSE" else "reference", winner, name))
        return sorted({row for row in result if row[1] != row[2] and all(x in present for x in row[1:])})
    return pairs


class RefinementScores(IncrementalScores):
    def add_view(self, view, split, arm, policy, estimate, std):
        start = len(self.rows["window"])
        super().add_view(view, split, arm, policy, estimate, std)
        error = estimate.double()-view.target
        increment = estimate.double()-view.current.double()
        residual = view.target-view.current.double()
        relative_rms = (increment.square().mean((1,2,3,4))/residual.square().mean((1,2,3,4)).clamp_min(1e-30)).sqrt()
        difference = [error.diff(n=n,dim=2).square().mean((1,2,3,4)) for n in (1,2)]
        for row, values in zip(self.rows["window"][start:], torch.stack((*difference, relative_rms), dim=1).tolist()):
            row.update(dict(zip(("error_d1_mse", "error_d2_mse", "window_delta_rms_ratio"), values)))


def add(book, study, view, label, name, policy, output):
    book.add_view(view, label, name, policy, output, study.norm["std"])


@torch.inference_mode()
def estimate(record, policy, view, basis):
    device = view.current.device
    if is_extra(record):
        net = load_extra(record, policy, device)
        return teacher_predict(net, view.base, record["config"]["scale"]), net
    net = load_head(record, record[policy], basis, device)
    return predict(net, view, record["config"]["model"]), net


def run_outer(study, train, freeze, basis, book, audit):
    old = read_json(FACTORIAL / "outer_lineage.json")
    selections = freeze["selection"]
    models, lineage = {}, {}
    for region in ("interpolation", "upper", "lower"):
        print("OUTER START", region, flush=True)
        active, held, schedule = outer_bundles(study, train, region)
        reference = old["regimes"][region]
        assert {k:v for k,v in reference.items() if k != "candidates"} == active.identity
        engine = Fits(study.out, study.cache, active, None, schedule, basis)
        records = {}
        for arm in ("I0", "I0_match2", "IB_match2"):
            tag = f"outer_{region}_{arm}"
            path = ROOT / reference["candidates"][arm]["directory"]
            item = stamp(read_json(path / "structured_fits" / f"{tag}.json"), path)
            if arm == "I0":
                shared = None
            elif arm == "IB_match2":
                shared = STRUCTURE / f"shared_outer_{region}_IB.pt"
            else:
                first = next(row for row in records["M0"]["states"] if row["step"] == 1000)
                shared = ROOT / records["M0"]["directory"] / first["path"]
            label = {"I0":"M0","I0_match2":"I0_match2_fixed"}.get(arm,arm)
            verify_reuse(item, engine, audit, freeze["models"][label]["config"]["model"], shared)
            assert item["selected"] == item["final"] == reference["candidates"][arm]["final"]
            records[label] = item
        shared = prefix(records["M0"], study.out)
        for family in ("EN","TM1","TM2"):
            arm = selections["families"][family]
            model = freeze["models"][arm]["config"]["model"]
            old_tm2 = records["I0_match2_fixed"]
            if family == "TM2" and model == old_tm2["config"]["model"]:
                records[arm] = old_tm2
            else:
                tag = f"outer_{region}_{arm}"
                records[arm] = stamp(engine.fit(tag,model,shared=shared),study.out)
        if selections["extra_outer"]:
            teacher = next(r for r in held.identity["refs"] if r["fold"] == 4)
            source = dict(path=teacher["verified_checkpoint"],sha256=teacher["final_sha256"],layout="NCDHW")
            # The isolated teacher was trained with its own original physical output scale.
            assert teacher["scale"] == active.scale
            net = load_net(ROOT/source["path"], source["sha256"], "S3", active.current.device, source["layout"])
            name = "S3_extra_MSE"
            records[name] = fit_extra(study, f"outer_{region}_{name}", net, source, active.data,
                                      active.base, teacher["scale"], schedule)
        split = f"holdout_{region}"
        add(book, study, held, split, "Native", "final", torch.zeros_like(held.current))
        add(book, study, held, split, "S3", "final", held.current)
        for name, record in records.items():
            output, net = estimate(record, "final", held, basis)
            if is_extra(record):
                original = held.data.y
                held.data.y = original.roll(1,0)+.03125
                try:
                    altered = base(held.data.x,held.data.p,study.norm)
                    assert torch.equal(teacher_predict(net,altered,record["config"]["scale"]),output)
                finally:
                    held.data.y = original
            else:
                target_invariance(held, lambda:None, dict(record,kind="structured"), net, output)
            add(book, study, held, split, name, "final", output)
            models[f"outer_{region}_{name}"] = record
        lineage[region] = dict(held.identity, candidates={n:dict(final=d["final"],directory=d["directory"],
                                             config=d["config"],target_invariance=True) for n,d in records.items()})
        atomic_json(study.out/"outer_lineage.json", dict(freeze_sha256=sha(study.out/"FROZEN.json"),regimes=lineage))
        print("OUTER COMPLETE", region, len(records), flush=True)
    return models


@torch.inference_mode()
def check_publication(study, teacher, record, view, extra_teacher, basis, case_index):
    device = view.current.device
    observed = view.data.x[case_index:case_index+1]
    backbone = study.triplets.source.backbone
    scale = study.cfg["external_rms"]
    candidate = load_head(record, record["final"], basis, device)
    baseline = Publisher(backbone, teacher, study.norm, scale, view.scale)
    publication = Publisher(backbone, teacher, study.norm, scale, view.scale, candidate)
    extra = Publisher(backbone, extra_teacher, study.norm, scale, view.scale)
    cached = [(view.data,"previous_p"),(view.data,"previous_x"),(view.data,"historical_prediction"),
              (view,"true"),(view,"historical")]
    original = [(obj,key,getattr(obj,key)) for obj,key in cached]
    expected = publication(observed)
    assert torch.equal(load_publisher()(observed),expected)
    cache_cases = {}
    for action in ("change","delete","replace"):
        try:
            for obj,key,value in original:
                if action == "delete":
                    delattr(obj,key)
                else:
                    changed = value + .03125 if action == "change" else torch.zeros_like(value)
                    setattr(obj,key,changed)
            cache_cases[action] = torch.equal(publication(observed), expected)
        finally:
            for obj,key,value in original:
                setattr(obj,key,value)
    real_target = view.data.y
    try:
        view.data.y = torch.zeros_like(real_target)
        assert torch.equal(publication(observed),expected)
    finally:
        view.data.y = real_target
    calls = []
    def record_call(_, args):
        if not calls:
            assert not torch.count_nonzero(args[0])
        calls.append(int(args[1].item()))
    hook = backbone.net.register_forward_pre_hook(record_call)
    try:
        assert torch.equal(publication(observed), expected)
    finally:
        hook.remove()
    assert calls == LABELS
    native_live = publication.native(observed)
    features = base(observed, native_live, study.norm)
    reference = publication.correct(features)
    native_cached = view.data.p[case_index:case_index+1]
    cached_features = base(observed, native_cached, study.norm)
    cached_reference = publication.correct(cached_features)
    assert torch.equal(cached_reference, view.current[case_index:case_index+1])
    cached_output = native_cached.double() + (cached_reference.double()
                     + publication.increment(cached_features, cached_reference))
    native_drift = float((native_live-native_cached).abs().max())
    teacher_drift = float((reference-cached_reference).abs().max())
    final_drift = float((expected-cached_output).abs().max())
    # Measure historical-cache reproducibility in physical units against a small FP32 error budget.
    field_scale = max(float(observed.abs().max()),float(cached_output.abs().max()))
    tolerance = 16*torch.finfo(observed.dtype).eps*field_scale
    assert max(native_drift,teacher_drift,final_drift) <= tolerance
    replay = dict(cached_teacher_bitwise=True, native_max_abs_error=native_drift,
                  s3_correction_max_abs_error=teacher_drift, final_max_abs_error=final_drift,
                  absolute_tolerance=tolerance)
    zero = load_head(record, record["training"][0], basis, device)
    zero_publisher = Publisher(backbone, teacher, study.norm, scale, view.scale, zero)
    assert torch.equal(zero_publisher(observed), baseline(observed))
    native = native_live.double()
    y = view.data.y[case_index:case_index+1].double()
    reconstruction = native + (reference.double() + (y-native-reference.double()))
    assert float((reconstruction-y).abs().max()) < 1e-15
    configs = (("S3",baseline),("S3_extra_MSE",extra),("candidate",publication))
    costs = [dict(arm=name, trainable_parameters=sum(p.numel() for p in (net.head or net.s3).parameters())
                  if name != "S3" else 0, **benchmark(net,observed)) for name,net in configs]
    base_cost = costs[0]
    for row in costs:
        row["additional_end_to_end_ms"] = row["ms_end_to_end_median"]-base_cost["ms_end_to_end_median"]
        row["additional_end_to_end_peak_bytes"] = row["bytes_end_to_end_peak_delta"]-base_cost["bytes_end_to_end_peak_delta"]
        row["additional_parameter_bytes_vs_s3"] = 4*(row["total_parameters"]-base_cost["total_parameters"])
    assert all(cache_cases.values()) and costs[2]["increment_parameters"] == 36354
    keys = sorted({k for row in costs for k in row})
    table(study.out/"inference_cost.csv", [{k:row.get(k) for k in keys} for row in costs])
    return dict(denoising_labels=calls, zero_initial_noise=True, cache_invariance=cache_cases,cache_replay=replay,
                zero_increment_bitwise=True, public_loader_bitwise=True, future_bitwise=True,
                altered_cache_fields=[key for _,key in cached], true_remaining_max_error=float((reconstruction-y).abs().max()),
                backbone_frozen=not any(p.requires_grad for p in backbone.parameters()),
                s3_frozen=not any(p.requires_grad for p in teacher.parameters()),
                input_shape=list(observed.shape))


@torch.inference_mode()
def later_period(study, test, old_period, plan):
    source = study.triplets.source
    previous = {(c["file"],c["start"]):(old_period.x[i],old_period.p[i]) for i,c in enumerate(old_period.cases)}
    values = {k:[] for k in ("x","p","y","previous_x","previous_p","cases")}
    device = test.p.device
    calls = []
    def hook_call(_, args):
        if not calls:
            assert not torch.count_nonzero(args[0])
        calls.append(int(args[1].item()))
    hook = source.backbone.net.register_forward_pre_hook(hook_call)
    try:
        for spec in plan:
            starts = spec["starts"]
            a,b = starts[0], starts[-1]+40
            assert starts == list(range(a,spec["frames"]-39,20))
            with h5py.File(source.raw/spec["file"], "r") as f:
                assert all(f[k].shape[0] == spec["frames"] for k in ("u","v"))
                fields = np.stack([f[k][a:b,::2,::2] for k in ("u","v")]).astype(np.float32)
            fields = torch.from_numpy(fields).to(device)
            for start in starts:
                x = fields[:,start-a:start-a+20]
                y = fields[:,start-a+20:start-a+40]
                px,pp = previous[spec["file"],start-20]
                calls.clear()
                mean,std = study.norm["mean"],study.norm["std"]
                p = (source.backbone.rollout((x[None]-mean)/std,seed=500000,sign=0)*std+mean)[0]
                assert calls == LABELS and torch.isfinite(p).all()
                previous[spec["file"],start] = x,p
                case = dict(file=spec["file"],start=start,re=spec["re"],issue_frame=start+20,score_frame=start+40)
                for key,v in zip(values,(x,p,y,px,pp,case)):
                    values[key].append(v)
    finally:
        hook.remove()
    return SimpleNamespace(**{k:v if k=="cases" else torch.stack(v) for k,v in values.items()})


def validate(study, figures):
    out = study.out
    frozen = guard(out)
    _,audit,basis = prior_study()
    assert frozen["prior_freeze"] == sha(FACTORIAL/"FROZEN.json")
    plan = read_json(out/"experiment_plan.json")
    train, _, _ = study.get("train")
    absolute = (train.y.double()-train.p.double()).abs()[:,:,::2,::2,::2]
    book = RefinementScores(basis,[float(torch.quantile(absolute[:,i].contiguous().flatten(),.95)) for i in (0,1)])
    del absolute
    outer = run_outer(study,train,frozen,basis,book,audit)
    del train
    from .score import score_remaining, complete

    evidence = score_remaining(study,frozen,book,basis,plan)
    scores = book.write(out,{n:"M0" for n in ("Native","S3",*frozen["models"])},
                        comparison_lists(frozen["selection"]))
    equal_regimes(out)
    complete(study,figures,frozen,outer,dict(scores,**evidence))
