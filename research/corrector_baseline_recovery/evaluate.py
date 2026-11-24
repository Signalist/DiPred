"""Post-freeze physical scoring with paired original-trajectory uncertainty."""
from collections import defaultdict
import csv
from statistics import mean

import numpy as np
import torch

from research.corrector.run import dump, sha, table
from research.corrector_predictability.alpha import moments
from research.corrector_predictability.data import OUT as PREVIOUS, SMALL, Split, normalization, sources
from research.corrector_window_study.data import read_json, tensor_sha

from .fit import MODES, SEEDS, load_model, r0_rows
from .model import input_fields, issue, residual

META = ("mode","policy","seed","trajectory","start","re","step","alpha")


def groups(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in fields)].append(row)
    return grouped


def averages(rows, keys, measurements):
    return [dict(zip(keys,ks), **{name:mean(r[name] for r in values) for name in measurements})
            for ks,values in groups(rows,keys).items()]


def metrics(q, p, y, r):
    target = y-p
    c,e,b = moments(r,target)
    gain = 2*c-e
    error = q-y
    m = error.square().mean(dim=(1,2,3,4))
    gap = m-b+gain
    assert gap.abs().max() < 1e-15
    cl,el,bl = moments(r,target,(1,3,4))
    ml = error.square().mean(dim=(1,3,4))
    assert (ml-bl+2*cl-el).abs().max() < 1e-15
    channels = error.square().mean(dim=(2,3,4))
    result = dict(native_mse=b,mse=m,mae=error.abs().mean(dim=(1,2,3,4)),
                  mse_u=channels[:,0],mse_v=channels[:,1],C=c,E=e,G=gain)
    return result,(cl,el,bl,ml),float(gap.abs().max())


def with_rate(row):
    row["correction_pct"] = (100*(1-row["mse"]/row["native_mse"])
                             if row["native_mse"] else 0.)
    return row


@torch.inference_mode()
def score(out):
    frozen_path = out/"FROZEN.json"
    frozen = read_json(frozen_path)
    for key in ("training","config","sources"):
        name = "training.csv" if key=="training" else key+".json"
        assert sha(out/name)==frozen[f"{key}_sha256"]
    manifest, lineage = sources()
    test = Split(manifest,"test",SMALL)
    test_p_sha = tensor_sha(test.p)
    assert test_p_sha == read_json(PREVIOUS/"verification.json")["test_P_sha256"]
    mean_field, units, scale, _ = normalization(test.x.device)
    inputs = input_fields(test,mean_field,units)
    p,y = (a.double() for a in (test.p,test.y))
    rows, leads = [], []
    max_gap = 0.
    r0_error = 0.
    previous_scores = {}
    with (PREVIOUS/"seed_trajectories.csv").open() as file:
        for r in csv.DictReader(file):
            if r["mode"]=="Expanded":
                previous_scores[r["policy"],int(r["seed"]),r["trajectory"]] = float(r["mse"])

    def record(mode, seed, policy, step, alpha, physical):
        nonlocal max_gap
        q = issue(p,physical,alpha)
        if alpha==0.:
            assert torch.equal(q,p)
        full, detail, gap = metrics(q,p,y,alpha*physical)
        max_gap = max(max_gap,gap)
        full = {name:t.tolist() for name,t in full.items()}
        detail = [t.tolist() for t in detail]
        for i, case in enumerate(test.cases):
            base = dict(mode=mode,policy=policy,seed=seed,trajectory=case["file"],start=case["start"],
                        re=case["re"],step=step,alpha=alpha)
            rows.append(with_rate(dict(base,**{k:v[i] for k,v in full.items()})))
            for k,(c,e,b,m) in enumerate(zip(*(v[i] for v in detail)),start=1):
                leads.append(with_rate(dict(base,lead=k,native_mse=b,mse=m,C=c,E=e,G=2*c-e)))

    record("Native",0,"zero",0,0.,torch.zeros_like(p))
    for mode in MODES:
        for seed in SEEDS:
            ck_predictions = {}
            for policy, row in frozen["selected"][mode][str(seed)].items():
                ck = row["sha256"]
                if ck not in ck_predictions:
                    head = load_model(out,mode,row,p.device)
                    ck_predictions[ck] = residual(head,inputs,scale).double()
                record(mode,seed,policy,row["step"],row["alpha"],ck_predictions[ck])
    assert torch.equal(test.p.double(),p) and tensor_sha(test.p)==test_p_sha
    assert lineage == sources()[1]
    metrics_names = tuple(k for k in rows[0] if k not in META and k!="correction_pct")
    by_seed = averages(rows,("mode","policy","seed","trajectory","re"),metrics_names)
    for r in by_seed:
        if r["mode"]=="R0" and r["policy"] in ("selected","final_raw","final_fit"):
            prev = previous_scores[r["policy"],r["seed"],r["trajectory"]]
            r0_error = max(r0_error,abs(r["mse"]-prev))
    assert r0_error < 2e-11, r0_error
    tracks = averages(by_seed,("mode","policy","trajectory","re"),metrics_names)
    counts = groups(tracks,("mode","policy"))
    names = sorted({r["trajectory"] for r in tracks})
    assert len(names)==12
    native = {r["trajectory"]:r["mse"] for r in counts["Native","zero"]}
    baseline = np.array([native[k] for k in names])
    idx = np.random.default_rng(42).integers(len(names),size=(10000,len(names)))
    summary = []
    for (mode,policy), sub in counts.items():
        values = {r["trajectory"]:r["mse"] for r in sub}
        assert set(values)==set(names)
        actual = np.array([values[k] for k in names])
        gains = 100*(1-actual[idx].mean(1)/baseline[idx].mean(1))
        lo,hi = np.quantile(gains,(.025,.975))
        summary.append(with_rate(dict(mode=mode,policy=policy,low95_pct=float(lo),high95_pct=float(hi),
                                      improved_trajectories=int((actual<baseline).sum()),trajectories=len(names),
                                      **{m:mean(r[m] for r in sub) for m in metrics_names})))
    regime = averages(tracks,("mode","policy","re"),metrics_names)
    regime_counts = groups(tracks,("mode","policy","re"))
    for r in regime:
        rr = regime_counts[r["mode"],r["policy"],r["re"]]
        r["improved_trajectories"] = sum(t["mse"]<t["native_mse"] for t in rr)
        r["trajectories"] = len(rr)
    seed_summary = averages(by_seed,("mode","policy","seed"),metrics_names)
    seed_counts = groups(by_seed,("mode","policy","seed"))
    for r in seed_summary:
        rr = seed_counts[r["mode"],r["policy"],r["seed"]]
        r["improved_trajectories"] = sum(t["mse"]<t["native_mse"] for t in rr)
        r["trajectories"] = len(rr)
    lead_names = ("native_mse","mse","C","E","G")
    lead_seed = averages(leads,("mode","policy","seed","trajectory","lead"),lead_names)
    lead_tracks = averages(lead_seed,("mode","policy","trajectory","lead"),lead_names)
    lead_summary = averages(lead_tracks,("mode","policy","lead"),lead_names)
    contrasts = []
    links = {"R1":("R0",),"R2":("R1",),"R3":("R1",),"R4":("R2","R3")}
    for mode, baselines in links.items():
        for policy in ("selected","step1000_raw","step1000_fit","final_raw","final_fit"):
            for comparator in baselines:
                if comparator=="R0" and policy.startswith("final"):
                    continue
                for metric in ("mse","C","E","G"):
                    tr = dict()
                    for arm in (mode,comparator):
                        index = {r["trajectory"]:r[metric] for r in counts[arm,policy]}
                        tr[arm]=np.array([index[k] for k in names])
                    a,b=tr[mode],tr[comparator]
                    dist=np.mean((a-b)[idx],axis=1)
                    lo,hi=np.quantile(dist,(.025,.975))
                    contrasts.append(dict(mode=mode,comparator=comparator,policy=policy,metric=metric,
                                          delta=float(np.mean(a-b)),low95=float(lo),high95=float(hi)))
    for filename,data in (("windows",rows),("lead_windows",leads),("trajectories",tracks),
                          ("seed_trajectories",by_seed),("regimes",regime),("seeds",seed_summary),
                          ("leads",lead_summary),("scores",summary),("contrasts",contrasts)):
        table(out/f"{filename}.csv",[with_rate(r) if filename!="contrasts" else r for r in data])
    assert len(rows)==1824 and len(leads)==36480 and len(summary)==26
    assert all(np.isfinite(r[k]) for r in rows for k in metrics_names)
    winner = frozen["winner"]
    interventions = {"R0": (r0_rows()[42,1000],1.)}
    if winner and winner!="R0":
        picked = frozen["selected"][winner]["42"]["selected"]
        interventions[winner] = (picked,picked["alpha"])
    for mode,(row,alpha) in interventions.items():
        head = load_model(out,mode,row,p.device)
        original = issue(p,residual(head,input_fields(test,mean_field,units),scale).double(),alpha)
        test.y = test.y.flip(0)
        assert not torch.equal(test.y.double(),y)
        perturbed = issue(p,residual(head,input_fields(test,mean_field,units),scale).double(),alpha)
        assert torch.equal(original,perturbed)
        test.y = test.y.flip(0)
    selected_gains = [r["correction_pct"] for r in seed_summary
                      if r["mode"]==winner and r["policy"]=="selected"]
    row = next((r for r in summary if r["mode"]==winner and r["policy"]=="selected"),None)
    stable = bool(row and min(selected_gains)>0 and row["low95_pct"]>0)
    dump(out/"verification.json",dict(passed=True,frozen_sha256=sha(frozen_path),
                                      test_p_sha256=test_p_sha,checked_r0_previous_max_mse_error=r0_error,
                                      zero_exact_native=True,future_intervention_bitwise_equal=True,
                                      source_weights_unchanged=True,max_decomposition_error=max_gap,
                                      future_intervention_methods=list(interventions),
                                      test_trajectories=len(names),windows=len(rows),lead_windows=len(leads),
                                      dev_selected_candidate=winner,test_seed_gains=selected_gains,
                                      stable_exploratory_positive=stable))
    print("TEST SCORED",winner,stable,"max decomposition error",max_gap,flush=True)
