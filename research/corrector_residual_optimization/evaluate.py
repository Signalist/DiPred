"""Uniform post-freeze scoring with physical decomposition and paired trajectories."""
import numpy as np
import torch

from research.corrector.run import dump, sha
from research.corrector_simple_baseline.fit import table
from research.corrector_simple_baseline.measure import average, correction_rate, errors, groups
from research.corrector_window_study.data import ROOT, WORKSPACE, read_json, tensor_sha

from .data import SIMPLE, split
from .diagnose import analytic_prediction
from .fit import load, predict, read
from .model import inputs


def structure(r,target):
    r,t = r.double(),target.double()
    def energy(value):
        return value.square().mean((1,2,3,4))
    t1,r1 = t.diff(dim=2),r.diff(dim=2)
    return dict(rms_ratio=(energy(r)/energy(t).clamp_min(1e-30)).sqrt(),
                spatial_gradient_match=.5*(energy((r-t).diff(dim=3))+energy((r-t).diff(dim=4))),
                temporal_d1=energy(r1),temporal_d2=energy(r1.diff(dim=2)),
                truth_temporal_d1=energy(t1),truth_temporal_d2=energy(t1.diff(dim=2)),
                temporal_difference_match=energy(r1-t1))


def reduced_metrics(r,target,axes):
    r,t = r.double(),target.double()
    c,e,b,m = ((r*t).mean(axes),r.square().mean(axes),t.square().mean(axes),(r-t).square().mean(axes))
    g = 2*c-e
    assert (b-m-g).abs().max()<1e-15
    return dict(C=c,E=e,G=g,native_mse=b,mse=m)


def aggregate(rows,metadata,detail=()):
    keys = ("split","run",*detail)
    metrics = tuple(k for k in rows[0] if k not in metadata)
    tracks = average(rows,(*keys,"trajectory","re"),metrics)
    return tracks,average(tracks,keys,metrics),average(tracks,(*keys,"re"),metrics)


def report_tables(out,rows,lead,channels,base):
    keys = ("split","run","trajectory","re","start")
    tracks,scores,regimes = aggregate(rows,keys)
    _,leads,regime_leads = aggregate(lead,(*keys,"lead"),("lead",))
    channel_tracks,channel_summary,channel_regimes = aggregate(channels,(*keys,"channel"),("channel",))
    track_groups = groups(tracks,("split","run"))
    ci_rows = []
    for row in scores:
        own = track_groups[row["split"],row["run"]]
        row["improved"] = sum(r["mse"]<r["native_mse"] for r in own)
        row["trajectories"] = len(own)
        row["improved_fraction"] = row["improved"]/len(own)
    for row in regimes:
        own = [r for r in track_groups[row["split"],row["run"]] if r["re"]==row["re"]]
        row["improved"] = sum(r["mse"]<r["native_mse"] for r in own)
        row["trajectories"] = len(own)
    for row in scores:
        subset = row["split"]
        own = sorted(track_groups[subset,row["run"]],key=lambda r:r["trajectory"])
        sources = ["Native"] + ([base] if (subset,base) in track_groups else [])
        n = len(own)
        rng = np.random.default_rng(42)
        ix = rng.integers(n,size=(10000,n))
        value = np.array([r["mse"] for r in own])
        for control in sources:
            other = sorted(track_groups[subset,control],key=lambda r:r["trajectory"])
            assert [r["trajectory"] for r in own]==[r["trajectory"] for r in other]
            reference = np.array([r["mse"] for r in other])
            interval = np.quantile(100*(1-value[ix].mean(1)/reference[ix].mean(1)),(.025,.975))
            ci_rows.append(dict(split=subset,run=row["run"],control=control,trajectories=n,
                                improvement_pct=float(100*(1-value.mean()/reference.mean())),
                                ci_low_pct=float(interval[0]),ci_high_pct=float(interval[1])))
            if control=="Native":
                row.update(ci_low_pct=float(interval[0]),ci_high_pct=float(interval[1]))
    for file,values in (("scores",scores),("windows",rows),("trajectories",tracks),("regimes",regimes),
                        ("lead_windows",lead),("leads",leads),("regime_leads",regime_leads),
                        ("channels",channel_summary),("channel_trajectories",channel_tracks),
                        ("channel_regimes",channel_regimes)):
        table(out/f"{file}.csv",[correction_rate(r) for r in values])
    table(out/"contrasts.csv",ci_rows)
    return scores


def timing(net,x):
    net.eval()
    with torch.inference_mode():
        x = x[:8]
        for _ in range(5):
            net(x)
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(30):
            net(x)
        b.record()
        b.synchronize()
    return a.elapsed_time(b)/(30*len(x))


@torch.inference_mode()
def evaluate(data):
    out = data.out
    frozen_path = out/"FROZEN.json"
    frozen = read_json(frozen_path)
    for name,digest in frozen["bindings"].items():
        assert sha(out/name)==digest,name
    selections = read_json(out/"selections.json")
    cfg = read_json(out/"config.json")
    controls = read_json(out/"linear_controls.json")
    runs = selections["runs"]
    for row in read(out/"training.csv"):
        checkpoint = out/row["checkpoint"] if not row["external_checkpoint"] else row["external_checkpoint"]
        assert sha(checkpoint)==row["sha256"]
    models = ("Native","B1","B2",*runs)
    base = selections["base"]
    pictured = selections["auxiliary"].get("best",base)
    rows,leaders,channels,times = [],[],[],[]
    mismatch,s0_difference = 0.,0.
    intervention = False
    for region in ("train","development","test"):
        source = split(data,region)
        fingerprint = tensor_sha(source.p)
        mlist = models if region!="train" else ("Native","B1","B2","S0",*([base] if base!="S0" else []))
        fixed = {}
        for name in mlist:
            if name=="Native":
                r = torch.zeros_like(source.p)
                assert torch.equal(source.p+r,source.p)
            elif name in ("B1","B2"):
                r = analytic_prediction(source,controls,name)
            else:
                run = runs[name]
                conf = run["config"]
                mode = conf["input"]
                if mode not in fixed:
                    fixed[mode] = inputs(source,data.norm,mode)
                net = load(conf,run["selected"],out,source.p.device)
                r = predict(net,fixed[mode],cfg["scale"])
                if region=="development":
                    times.append(dict(run=name,parameters=sum(p.numel() for p in net.parameters()),
                                      ms_per_window_batch8=timing(net,fixed[mode])))
                if region=="test" and name==pictured:
                    original = source.y
                    source.y = original.flip(0)
                    try:
                        assert not torch.equal(source.y,original)
                        clone = predict(net,inputs(source,data.norm,mode),cfg["scale"])
                        assert torch.equal(source.p+r,source.p+clone)
                        intervention = True
                    finally:
                        source.y = original
            target = source.y.double()-source.p.double()
            stat,lead,gap = errors(r,source.p,source.y,data.norm["std"])
            mismatch = max(mismatch,gap)
            stat.update(structure(r,target))
            ch = reduced_metrics(r,target,(2,3,4))
            stat = {k:v.tolist() for k,v in stat.items()}
            lead = {k:v.tolist() for k,v in lead.items()}
            ch = {k:v.tolist() for k,v in ch.items()}
            for i,case in enumerate(source.cases):
                row = dict(split=region,run=name,trajectory=case["file"],re=case["re"],start=case["start"])
                rows.append(dict(row,**{k:v[i] for k,v in stat.items()}))
                for j in range(20):
                    leaders.append(dict(row,lead=j+1,**{k:v[i][j] for k,v in lead.items()}))
                for j,channel in enumerate(("u","v")):
                    channels.append(dict(row,channel=channel,**{k:v[i][j] for k,v in ch.items()}))
            if region=="development" and name==pictured:
                case = cfg["plot"]
                i = next(i for i,c in enumerate(source.cases) if (c["file"],c["start"])==(case["file"],case["start"]))
                lead_index = case["lead"]-1
                a,b = (v[i,:,lead_index].double().cpu().numpy() for v in (target,r))
                table(out/"fixed_residual.csv",[
                    dict(channel=channel,y=y,x=x,truth=a[j,y,x],prediction=b[j,y,x],remaining=a[j,y,x]-b[j,y,x])
                    for j,channel in enumerate(("u","v")) for y in range(32) for x in range(64)])
        assert tensor_sha(source.p)==fingerprint
    assert intervention
    table(out/"inference_cost.csv",times)
    scores = report_tables(out,rows,leaders,channels,base)
    simple = {r["policy"]:float(r["mse"]) for r in read(SIMPLE/"scores.csv") if r["split"]=="test"}
    for name,ref in (("Native","native"),("S0","selected")):
        value = next(r["mse"] for r in scores if r["split"]=="test" and r["run"]==name)
        s0_difference = max(s0_difference,abs(value-simple[ref]))
    assert s0_difference<2e-13
    for file,digest in ((ROOT/"reliability/checkpoints/current.pt",data.identity["packaged_checkpoint_sha256"]),
                        (WORKSPACE/"runs/physical20_augmented_dynamics_20261109/augmented_diffusion/model.pt",
                         data.identity["original_checkpoint_sha256"])):
        assert sha(file)==digest
    assert all(np.isfinite(r["mse"]) for r in rows)
    dump(out/"verification.json",dict(passed=True,frozen_sha256=sha(frozen_path),native_s0_max_mse_error=s0_difference,
                                      max_decomposition_error=mismatch,exact_zero_native=True,zero_noise_paths_only=True,
                                      source_weights_unchanged=True,prediction_unchanged_by_future=True,
                                      windows=len(rows),lead_windows=len(leaders),
                                      test_trajectory_count=12,selected_base=base,shown_in_development_fields=pictured))
    print("EVALUATED",len(rows),len(leaders),base,flush=True)
