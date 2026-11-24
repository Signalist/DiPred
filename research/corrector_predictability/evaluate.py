"""Once-frozen test evaluation, decomposition, and paired summaries."""
import csv
from collections import defaultdict
from statistics import mean
from time import monotonic

import numpy as np
import torch

from research.corrector.run import table
from research.corrector_consistency.metrics import case_metrics

from .alpha import moments, optimum, oracles, perturbation
from .data import OLD
from .fit import correction, read_model
from .model import direction_cases

MODES = ("selected", "final_fit", "final_raw")


@torch.inference_mode()
def add_scores(rows, dev, raw, alpha, mode, policy, seed, step, units, scale, thresholds):
    truth = dev.y.double()-dev.p.double()
    c,e,base = moments(raw, truth)
    corrected = alpha*raw.double()
    if alpha == 0.:
        assert torch.equal(dev.p.double()+corrected,dev.p.double())
    diagnostics = case_metrics(dev.p, dev.y, corrected, units, scale, thresholds)
    match = direction_cases(corrected/scale,truth/scale,"match")
    for i,(case,row) in enumerate(zip(dev.cases,diagnostics)):
        gc, ge = float(alpha*c[i]), float(alpha**2*e[i])
        assert abs((float(base[i])-row["mse"])-(2*gc-ge)) < 1e-14
        rows.append(dict(mode=mode, policy=policy, seed=seed, step=step, alpha=alpha,
                         trajectory=case["file"], start=case["start"], re=case["re"],
                         raw_C=float(c[i]), raw_E=float(e[i]),
                         raw_cosine=float(c[i]/(e[i]*base[i]).sqrt().clamp_min(1e-30)),
                         C=gc, E=ge, G=2*gc-ge, local_dir_match=float(match[i]),
                         **row))


def summaries(rows):
    windows = defaultdict(list)
    for r in rows:
        windows[r["mode"],r["policy"],r["seed"],r["trajectory"],r["re"]].append(r)
    metrics = [k for k in rows[0] if k not in ("mode","policy","seed","trajectory","re","start","step")]
    seed_tracks = []
    for (mode,policy,seed,name,re), group in windows.items():
        assert {r["start"] for r in group} == {400,420} and len(group)==2
        seed_tracks.append(dict(mode=mode,policy=policy,seed=seed,trajectory=name,re=re,
                                **{m:mean(r[m] for r in group) for m in metrics}))
    repeats = defaultdict(list)
    for r in seed_tracks:
        repeats[r["mode"],r["policy"],r["trajectory"],r["re"]].append(r)
    tracks = []
    for (mode,policy,name,re), group in repeats.items():
        assert len({r["seed"] for r in group}) == len(group)
        assert len(group) == (1 if mode=="Native" or mode.startswith("old-") else 3)
        tracks.append(dict(mode=mode,policy=policy,trajectory=name,re=re,
                           **{m:mean(r[m] for r in group) for m in metrics}))
    base = {r["trajectory"]:r["mse"] for r in tracks if r["mode"]=="Native"}
    summary = []
    groups = defaultdict(list)
    for r in tracks:
        groups[r["mode"],r["policy"]].append(r)
    for (mode,policy), group in groups.items():
        assert {r["trajectory"] for r in group} == set(base)
        avg = {m:mean(r[m] for r in group) for m in metrics}
        o = optimum(avg["raw_C"],avg["raw_E"])
        sm = [r for r in seed_tracks if (r["mode"],r["policy"])==(mode,policy)]
        individual = defaultdict(list)
        for r in sm:
            individual[r["seed"]].append(r)
        o_per_seed = []
        for own in individual.values():
            c,e=mean(r["raw_C"] for r in own),mean(r["raw_E"] for r in own)
            o_per_seed.append(-perturbation(c,e,optimum(c,e)))
        summary.append(dict(mode=mode,policy=policy,trajectories=len(group),
                            improvement_pct=100*(1-avg["mse"]/mean(base.values())),
                            positive_raw_inner_trajectories=sum(r["raw_C"]>0 for r in group),
                            pooled_global_oracle_alpha=o,
                            per_seed_global_oracle_improvement_pct=100*mean(o_per_seed)/mean(base.values()),
                            **avg))
    return seed_tracks,tracks,summary


def comparisons(tracks):
    groups = defaultdict(dict)
    for r in tracks:
        groups[r["mode"],r["policy"]][r["trajectory"]]=r
    base = groups["Native","selected"]
    names = sorted(base)
    idx = np.random.default_rng(42).integers(len(names),size=(10000,len(names)))
    metrics = ("mse","standardized_mse","mse_u","mse_v","mae","C","E","G","raw_C","raw_E",
               "raw_cosine","correction_cosine","local_dir_match","spatial_match",
               "high_gradient_mse","high_residual_mse","d1_match","d2_match")
    comparisons = []
    for (mode,policy), own in groups.items():
        if mode=="Native":
            continue
        controls = [("Native","selected")]
        links = {
            "Expanded": ["Small"], "XP-zero": ["Expanded"],
            "XP-active-control": ["XP-zero"],
            "XPF": ["XP-zero","XP-active-control","Expanded"],
            "Direction-smooth": ["Direction-base"],
            "Direction-match": ["Direction-base","Direction-smooth"],
            "Direction-align": ["Direction-base","Direction-smooth","Direction-match"],
        }
        controls += [(k,policy) for k in links.get(mode,[]) if (k,policy) in groups]
        for b,pol in controls:
            control=groups[b,pol]
            for key in metrics:
                a=np.array([own[n][key] for n in names])
                v=np.array([control[n][key] for n in names])
                d=a-v
                lo,hi=np.quantile(d[idx].mean(1),(.025,.975))
                pct,pl,pu="","",""
                if (v>0).all():
                    pct=100*(1-a.mean()/v.mean())
                    pl,pu=np.quantile(100*(1-a[idx].mean(1)/v[idx].mean(1)),(.025,.975))
                comparisons.append(dict(mode=mode,policy=policy,baseline=b,baseline_policy=pol,
                    metric=key,delta=d.mean(),lower95=lo,upper95=hi,improvement_pct=pct,
                    improvement_lower95_pct=pl,improvement_upper95_pct=pu,lower_tracks=int((d<0).sum())))
    return comparisons


@torch.inference_mode()
def evaluate(test, dev, frozen, out, units, scale, thresholds, lookup):
    windows, oracle = [], []
    fields = dict(truth=(dev.y[0]-dev.p[0]).cpu().numpy())
    timing = []
    add_scores(windows,test,torch.zeros_like(test.p),0.,"Native","selected",0,0,units,scale,thresholds)
    for arm,row in frozen["old_checkpoints"].items():
        model = read_model(OLD,"XP",row,test.p.device)
        raw = correction(model,lookup(test,"XP"),scale)
        a = frozen["stage0_alpha"][arm]
        for policy,alpha in (("selected",a),("final_raw",1.)):
            add_scores(windows,test,raw,alpha,"old-"+arm,policy,42,1000,units,scale,thresholds)
        oracle.extend(oracles(raw, test.y.double()-test.p.double(),test.cases,arm))
        fields["old-"+arm] = correction(model, lookup(dev,"XP"), scale)[0].cpu().numpy()

    for mode,runs in frozen["training_candidates"].items():
        net_mode = frozen["direction_base"] if mode.startswith("Direction-") else mode
        test_inputs = lookup(test,net_mode)
        for run in runs:
            reuse = {}
            for policy in MODES:
                row=run[policy]
                key=row["sha256"]
                if key not in reuse:
                    model=read_model(out,net_mode,row,test.p.device)
                    reuse[key]=correction(model,test_inputs,scale)
                raw=reuse[key]
                add_scores(windows,test,raw,row["alpha"],mode,policy,row["seed"],
                           row["step"],units,scale,thresholds)
                if row["seed"]==42 and policy=="final_raw":
                    model=read_model(out,net_mode,row,test.p.device)
                    fields[mode]=correction(model,lookup(dev,net_mode),scale)[0].cpu().numpy()
                    batch=tuple(v[:8] for v in test_inputs)
                    for _ in range(4):
                        model(*batch)
                    torch.cuda.synchronize()
                    start=monotonic()
                    for _ in range(30):
                        model(*batch)
                    torch.cuda.synchronize()
                    timing.append(dict(mode=mode,seed=42,batch=8,iterations=30,
                                       cached_corrector_ms_per_case=1000*(monotonic()-start)/240))

    ss,ts,overall=summaries(windows)
    with (OLD/"windows.csv").open() as f:
        old={(r["arm"],r["trajectory"],int(r["start"])):float(r["mse"])
             for r in csv.DictReader(f) if r["phase"]=="step1000"}
    replay=max(abs(r["mse"]-old[r["mode"][4:],r["trajectory"],r["start"]])
               for r in windows if r["mode"].startswith("old-") and r["policy"]=="final_raw")
    assert replay < 2e-10, replay
    base={r["trajectory"]:r["mse"] for r in ts if r["mode"]=="Native"}
    for r in ts:
        r["improvement_pct"]=100*(1-r["mse"]/base[r["trajectory"]])
    for r in windows:
        r["improvement_pct"]=100*(1-r["mse"]/r["before_mse"])
    leads = [dict(mode=r["mode"],policy=r["policy"],seed=r["seed"],trajectory=r["trajectory"],
                  start=r["start"],lead=i,mse=r[f"lead_{i:02d}"]) for r in windows for i in range(1,21)]
    for name, data in (("windows",windows),("seed_trajectories",ss),("trajectories",ts),
                       ("scores",overall),("contrasts",comparisons(ts)),("leads",leads),
                       ("stage0_ORACLE",oracle),("inference",timing)):
        table(out/f"{name}.csv",data)
    np.savez_compressed(out/"plot_case.npz",**fields)
    return dict(stage0_test_reproduction_max_abs=replay, max_identity_error=max(r["identity_error"] for r in windows),
                windows=len(windows), trajectories=len(base),
                all_window_metrics_finite=all(np.isfinite(x) for r in windows for x in r.values()
                                              if isinstance(x,(int,float))))
