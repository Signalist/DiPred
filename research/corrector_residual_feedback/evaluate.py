"""Frozen physical scoring, crossed contrasts, and observed-history self-feedback."""
import csv
import gzip
from collections import defaultdict
from statistics import mean

import numpy as np
import torch

from research.corrector.run import dump,sha
from research.corrector_residual_optimization.fit import load as old_load,predict as old_predict
from research.corrector_simple_baseline.fit import table
from research.corrector_simple_baseline.measure import groups
from research.corrector_window_study.data import WORKSPACE,ROOT,read_json,tensor_sha

from .data import PRIOR,guard
from .model import ARMS,SLOTS,base,encode,predict
from .run import checked_model,feedback,subset


def measurements(r,point,truth,std):
    r,p,y = (v.double() for v in (r,point,truth))
    target = y-p
    issued = p+r
    delta = issued-y
    parts = dict(native_mse=target.square(),mse=delta.square(),mae=delta.abs(),
                 standardized_mse=(delta/std).square(),C=r*target,correction_energy=r.square())
    error1,error2 = delta.diff(dim=2),delta.diff(dim=2).diff(dim=2)
    spatial = .5*(delta.diff(dim=3).square().mean((3,4))+delta.diff(dim=4).square().mean((3,4)))
    gap = 0.
    result = {}
    for dim,axes in (("window",(1,2,3,4)),("lead",(1,3,4)),("channel",(2,3,4)),
                     ("lead_channel",(3,4))):
        terms = {k:v.mean(axes) for k,v in parts.items()}
        c,e,b,m = (terms[k] for k in ("C","correction_energy","native_mse","mse"))
        gain = 2*c-e
        residual_mse = (r-target).square().mean(axes)
        gap = max(gap,float((b-m-gain).abs().max()),float((residual_mse-m).abs().max()))
        reduce = tuple(a for a in axes if a<3)
        terms.update(G=gain,cosine=c/(b*e).sqrt().clamp_min(1e-30),
                     rms_ratio=(e/b.clamp_min(1e-30)).sqrt(),
                     spatial_match=spatial.mean(reduce) if reduce else spatial)
        for key,v,pad in (("first_difference_match",error1,1),("second_difference_match",error2,2)):
            value = v.square().mean(axes)
            if dim in ("lead","lead_channel"):
                value = torch.cat((value.new_full((*value.shape[:-1],pad),float("nan")),value),dim=-1)
            terms[key] = value
        result[dim] = {k:v.tolist() for k,v in terms.items()}
    assert gap<1e-15
    return result,gap


def metric_rows(data,split_name,arm,policy,r,std):
    values,gap = measurements(r,data.p,data.y,std)
    rows = {k:[] for k in values}
    for i,case in enumerate(data.cases):
        meta = dict(split=split_name,arm=arm,policy=policy,trajectory=case["file"],re=case["re"],start=case["start"])
        for key,parts in values.items():
            if key=="window":
                rows[key].append(dict(meta,**{k:v[i] for k,v in parts.items()}))
            elif key=="channel":
                for c,label in enumerate(("u","v")):
                    rows[key].append(dict(meta,channel=label,**{k:v[i][c] for k,v in parts.items()}))
            elif key=="lead":
                for t in range(20):
                    rows[key].append(dict(meta,lead=t+1,**{k:v[i][t] for k,v in parts.items()}))
            else:
                for c,label in enumerate(("u","v")):
                    for t in range(20):
                        rows[key].append(dict(meta,channel=label,lead=t+1,**{k:v[i][c][t] for k,v in parts.items()}))
    return rows,gap


def average(rows,key):
    excluded = {"split","arm","policy","trajectory","re","start","lead","channel","native_pct","baseline_pct"}
    measurements = [k for k in rows[0] if k not in excluded]
    result = []
    for values,own in groups(rows,key).items():
        metrics = {}
        for m in measurements:
            finite = [v[m] for v in own if v[m] is not None and np.isfinite(v[m])]
            metrics[m] = mean(finite) if finite else None
        result.append(dict(zip(key,values),**metrics))
    return result


def rates(rows,extra=()):
    indices = ("split","policy",*extra)
    baseline = {tuple(r[k] for k in indices):r["mse"] for r in rows if r["arm"]=="A"}
    for row in rows:
        b = baseline[tuple(row[k] for k in indices)]
        row.update(native_pct=100*(1-row["mse"]/row["native_mse"]),
                   baseline_pct=100*(1-row["mse"]/b))
    return rows


def controls(arm):
    pairs = dict(E=("B","C"),F=("B","D"),G=("C","D"),H=("E","F","G"))
    return list(dict.fromkeys(("Native","A",*pairs.get(arm,()))))


def joint_statistics(tracks,capacity_pair=None):
    contrasts,factors = [],[]
    effect_names = dict(true=(0,),predicted=(1,),remaining=(2,),true_predicted=(0,1),
                        true_remaining=(0,2),predicted_remaining=(1,2),three_way=(0,1,2))
    for (split,policy),values in groups(tracks,("split","policy")).items():
        by_arm = {k[0]:sorted(rows,key=lambda r:r["trajectory"]) for k,rows in groups(values,("arm",)).items()}
        names = [r["trajectory"] for r in by_arm["Native"]]
        assert all([r["trajectory"] for r in a]==names for a in by_arm.values())
        n = len(names)
        ix = np.random.default_rng(42).integers(n,size=(10000,n))
        for arm,rows in by_arm.items():
            comparisons = controls(arm)
            if arm==capacity_pair:
                comparisons.append("Capacity")
            for comparator in comparisons:
                different = by_arm[comparator]
                a,b = (np.array([r["mse"] for r in v]) for v in (rows,different))
                improvement = 100*(1-a[ix].mean(1)/b[ix].mean(1))
                gain = b-a
                ci = np.percentile(improvement,[2.5,97.5])
                raw = np.percentile(gain[ix].mean(1),[2.5,97.5])
                contrasts.append(dict(split=split,policy=policy,arm=arm,control=comparator,
                                      gain_pct=100*(1-a.mean()/b.mean()),lower_pct=ci[0],upper_pct=ci[1],
                                      mse_reduction=float(gain.mean()),lower_reduction=raw[0],upper_reduction=raw[1],
                                      improved=int((a<b).sum()),trajectories=n))
        for name,parts in effect_names.items():
            effects = np.zeros(n)
            for arm,mask in ARMS.items():
                sign = np.prod([2*mask[i]-1 for i in parts])
                effects += sign*np.array([r["G"] for r in by_arm[arm]])/4
            lower,upper = np.percentile(effects[ix].mean(1),[2.5,97.5])
            factors.append(dict(split=split,policy=policy,effect=name,G_effect=float(effects.mean()),
                                lower=lower,upper=upper,trajectories=n))
    return contrasts,factors


@torch.inference_mode()
def simulate(data,current,fields,model,mask,scale,self_feedback):
    order = sorted(range(len(data.cases)),key=lambda i:(data.cases[i]["file"],data.cases[i]["start"]))
    outputs = torch.empty_like(data.x)
    states = []
    previous = None
    chain = 0
    for i in order:
        case = data.cases[i]
        connected = (previous is not None and data.cases[previous]["file"]==case["file"] and
                     case["start"]-data.cases[previous]["start"]==20)
        values = tuple(v[i:i+1] for v in fields)
        if connected:
            assert data.cases[previous]["issue_frame"]<case["issue_frame"]
            assert data.cases[previous]["score_frame"]==case["issue_frame"]
            assert torch.equal(data.previous_p[i],data.p[previous])
            assert torch.equal(data.previous_x[i],data.x[previous])
            chain += 1
            if self_feedback:
                prior_pred = outputs[previous:previous+1]
                values = (values[0],prior_pred,values[0]-prior_pred.double())
        else:
            chain = 1
        design = encode(current[i:i+1],values,mask,scale)
        outputs[i] = predict(model,design,scale)[0]
        states.append((i,chain,connected))
        previous = i
    return outputs,states


@torch.inference_mode()
def rollout(data,current,fields,selected,cfg,out,split_name):
    rows = []
    for name in selected["rollout"]:
        net = checked_model(12,selected["models"][name]["selected"],out,current.device)
        args = (data,current,fields,net,ARMS[name],cfg["rms"])
        fixed,order = simulate(*args,False)
        responsive,other = simulate(*args,True)
        assert order==other
        permuted = data.y
        data.y = permuted.flip(0)
        try:
            assert not torch.equal(data.y,permuted)
            intervened,check = simulate(*args,True)
            assert order==check and torch.equal(intervened,responsive)
        finally:
            data.y = permuted
        # Ground truth first participates in the scoring stage, after both releases.
        for i,chain,linked in order:
            case = data.cases[i]
            errors = [float((data.p[i].double()+r[i].double()-data.y[i].double()).square().mean())
                      for r in (fixed,responsive)]
            if not linked:
                assert torch.equal(fixed[i],responsive[i])
            rows.append(dict(split=split_name,arm=name,trajectory=case["file"],re=case["re"],start=case["start"],
                             issue_frame=case["issue_frame"],score_frame=case["score_frame"],
                             chain_window=chain,reset=not linked,fixed_reference_mse=errors[0],self_mse=errors[1],
                             self_gain_pct=100*(1-errors[1]/errors[0])))
    return rows


def rollout_summary(rows):
    """Score linked windows on the same trajectories in both feedback modes."""
    summary = []
    for (split,arm),cohort in groups(rows,("split","arm")).items():
        by_step = groups(cohort,("chain_window",))
        parts = [("all",cohort)]+[(step,items) for (step,),items in sorted(by_step.items())]
        for step,items in parts:
            tracks = groups(items,("trajectory",))
            fixed,own = (np.array([mean(r[field] for r in tracks[key]) for key in sorted(tracks)])
                         for field in ("fixed_reference_mse","self_mse"))
            n = len(tracks)
            sampled = np.random.default_rng(42).integers(n,size=(10000,n))
            boot = 100*(1-own[sampled].mean(1)/fixed[sampled].mean(1))
            low,high = np.percentile(boot,[2.5,97.5])
            summary.append(dict(split=split,arm=arm,chain_window=step,
                                fixed_reference_mse=float(fixed.mean()),self_mse=float(own.mean()),
                                self_gain_pct=100*(1-own.mean()/fixed.mean()),
                                lower_pct=low,upper_pct=high,
                                improved=int((own<fixed).sum()),trajectories=n,windows=len(items)))
    return summary


def time_model(net,z):
    for _ in range(4):
        net(z)
    a,b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    a.record()
    for _ in range(30):
        net(z)
    b.record()
    b.synchronize()
    return a.elapsed_time(b)/(30*len(z))


@torch.inference_mode()
def score(store):
    out = store.out
    frozen = guard(out)
    selected,cfg,ref = (frozen[key] for key in ("selections","config","reference"))
    source = read_json(out/"sources.json")
    fields = read_json(out/"crossfit_manifest.json")["folds"]
    assert all(sha(out/file)==digest for file,digest in frozen["checkpoints"].items())
    dimensions = defaultdict(list)
    all_rollouts,times,field_table = [],[],[]
    biggest_gap = 0.
    restoration_error = 0.
    intervention = False
    for name in ("train","development","test"):
        data = store.get(name)
        prior = tensor_sha(data.p)
        norm = store.source.norm
        history = base(data.x,data.p,norm)
        historical = feedback(data,norm,out,store.cache,name,fields if name=="train" else [],ref,
                              cfg["rms"] if name=="train" else cfg["external_rms"])
        ext = old_load(ref["config"],ref["selected"],PRIOR,history.device)
        temp = old_predict(ext,history,cfg["external_rms"])
        common = dict(Native=torch.zeros_like(data.p),S3=temp)
        if name=="development":
            times.append(dict(arm="S3",parameters=sum(v.numel() for v in ext.parameters()),
                              ms_per_window_batch8=time_model(ext,history[:8])))
        picture = cfg["case"]
        if name=="development":
            which, = [i for i,c in enumerate(data.cases) if (c["file"],c["start"])==(picture["file"],picture["start"])]
            t = picture["lead"]-1
            fields_shown = dict(zip(SLOTS,(v[which,:,t].cpu() for v in historical)))
            fields_shown["current_true"] = (data.y-data.p)[which,:,t].cpu()
        block = {k:[] for k in ("window","lead","channel","lead_channel")}
        for arm in ("Native","S3",*ARMS,"Capacity"):
            if arm in common:
                r = common[arm]
                released = {"selected":r,"final":r}
                if arm=="Native":
                    assert torch.equal(data.p+r,data.p)
                    tgt = data.y.double()-data.p.double()
                    error = float((data.p.double()+tgt-data.y.double()).abs().max())
                    assert error<1e-14
                    restoration_error = max(restoration_error,error)
            else:
                mask = ARMS[selected["capacity_matches"]] if arm=="Capacity" else ARMS[arm]
                z = encode(history,historical,mask,cfg["rms"],capacity=arm=="Capacity")
                choice = selected["models"][arm]
                net = checked_model(12,choice["selected"],out,z.device)
                r = predict(net,z,cfg["rms"])
                if name=="development":
                    times.append(dict(arm=arm,parameters=sum(v.numel() for v in net.parameters()),
                                      ms_per_window_batch8=time_model(net,z[:8])))
                if name=="test" and arm==selected["best_feedback"]:
                    y = data.y
                    data.y = y.flip(0)
                    try:
                        rerun = encode(base(data.x,data.p,norm),historical,mask,cfg["rms"])
                        assert torch.equal(data.p+r,data.p+predict(net,rerun,cfg["rms"]))
                        intervention = True
                    finally:
                        data.y = y
                if choice["selected"]["step"]==choice["final"]["step"]:
                    end = r
                else:
                    end = predict(checked_model(12,choice["final"],out,z.device),z,cfg["rms"])
                released = {"selected":r,"final":end}
            if name=="development" and arm in ("A",selected["best_feedback"]):
                kind = "base" if arm=="A" else "feedback"
                fields_shown[kind] = r[which,:,t].cpu()
                fields_shown[kind+"_remaining"] = fields_shown["current_true"]-fields_shown[kind]
            for policy,estimate in released.items():
                rows,error = metric_rows(data,name,arm,policy,estimate,norm["std"])
                biggest_gap = max(biggest_gap,error)
                for dim in block:
                    block[dim].extend(rows[dim])
        for dim,rows in block.items():
            dimensions[dim].extend(rows)
        if name=="development":
            field_table = [dict(channel=channel,y=y,x=x,**{k:float(v[c,y,x]) for k,v in fields_shown.items()})
                           for c,channel in enumerate(("u","v")) for y in range(32) for x in range(64)]
        if name in ("development","test"):
            all_rollouts.extend(rollout(data,history,historical,selected,cfg,out,name))
        assert tensor_sha(data.p)==prior
    assert intervention
    table(out/"fixed_fields.csv",field_table)
    table(out/"feedback_rollout.csv",all_rollouts)
    table(out/"feedback_rollout_summary.csv",rollout_summary(all_rollouts))
    table(out/"inference_cost.csv",times)
    for file,digest in ((ROOT/"reliability/checkpoints/current.pt",source["backbone"]["packaged_checkpoint_sha256"]),
                        (WORKSPACE/"runs/physical20_augmented_dynamics_20261109/augmented_diffusion/model.pt",
                         source["backbone"]["original_checkpoint_sha256"])):
        assert sha(file)==digest
    for dim,rows in dimensions.items():
        for r in rows:
            for k,v in r.items():
                if isinstance(v,float) and not np.isfinite(v):
                    assert k.endswith("difference_match") and dim.startswith("lead")
                    r[k] = None
        axes = {"window":(),"lead":("lead",),"channel":("channel",),
                "lead_channel":("channel","lead")}[dim]
        rates(rows,("trajectory","start",*axes))
        if dim=="window":
            table(out/"windows.csv",rows)
        elif dim=="lead":
            # The full per-window table exceeds the remote's uncompressed file size limit.
            with gzip.open(out/"lead_windows.csv.gz","wt",newline="") as f:
                writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        identity = ("split","policy","arm",*axes)
        tracks = average(rows,(*identity,"trajectory","re"))
        regimes = average(tracks,(*identity,"re"))
        full = average(tracks,identity)
        if dim=="window":
            contrast,factor = joint_statistics(tracks,capacity_pair=selected["capacity_matches"])
            lookup = {(r["split"],r["policy"],r["arm"]):r for r in contrast if r["control"]=="Native"}
            for r in full:
                support = lookup[r["split"],r["policy"],r["arm"]]
                r.update({k:support[k] for k in ("improved","trajectories","lower_pct","upper_pct")},
                         improved_fraction=support["improved"]/support["trajectories"])
            table(out/"contrasts.csv",contrast)
            table(out/"factorial_effects.csv",factor)
        filenames = {"window":("trajectories","regimes","scores"),"lead":("lead_trajectories","regime_leads","leads"),
                     "channel":("channel_trajectories","channel_regimes","channels"),
                     "lead_channel":("lead_channel_trajectories","lead_channel_regimes","lead_channels")}[dim]
        for filename,rec,key in zip(filenames,(tracks,regimes,full),((*axes,"trajectory"),(*axes,"re"),axes)):
            table(out/f"{filename}.csv",rates(rec,key))
    dump(out/"feedback_test_cache_manifest.json",read_json(out/"feedback_test.json"))
    dump(out/"verification.json",dict(passed=True,freeze_sha256=sha(out/"FROZEN.json"),
                                      score_max_identity_error=biggest_gap,restoration_max_error=restoration_error,
                                      source_weights_unchanged=True,
                                      test_future_mutation_bitwise=True,rollout_future_mutation_bitwise=True,
                                      zero_exact=True,truth_recovers_target=True,seed=42,
                                      window_rows=len(dimensions["window"]),lead_rows=len(dimensions["lead"]),
                                      rollout_rows=len(all_rollouts)))
    print("SCORED",len(dimensions["window"]),"windows",len(all_rollouts),"rollout rows",flush=True)
