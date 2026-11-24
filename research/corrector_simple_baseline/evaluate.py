"""Frozen scoring and trajectory-paired uncertainty on the new single-path P."""
import csv

import numpy as np
import torch

from research.corrector.run import dump, sha
from research.corrector_window_study.data import ROOT, WORKSPACE, read_json, tensor_sha

from .fit import STEPS, table
from .measure import average, correction_rate, errors, groups
from .model import fields, network, physical_residual


@torch.inference_mode()
def score(data):
    out = data.out
    frozen = read_json(out/"FROZEN.json")
    for name,digest in frozen["sources_sha256"].items():
        assert sha(out/name)==digest, name
    cfg = read_json(out/"config.json")
    sources = read_json(out/"sources.json")
    assert sources["experiment_protocol_sha256"]==sha(out/"PROTOCOL.md")
    assert cfg["correction_strength"]==1. and cfg["seed"]==42
    with (out/"training.csv").open() as stream:
        checkpoints = list(csv.DictReader(stream))
    assert tuple(int(r["step"]) for r in checkpoints)==STEPS
    for row in checkpoints:
        path = out/row["checkpoint"]
        assert sha(path)==row["sha256"], path
    selected = frozen["checkpoints"]
    lead_rows, rows = [], []
    max_identity = 0.
    plot_written = False
    for split_name in ("train","development","test"):
        split = data.split(split_name)
        point_digest = tensor_sha(split.p)
        model = network(split.p.device)
        inputs = fields(split,data.norm)
        predictions = {}
        for policy, row in selected.items():
            step = row["step"]
            if step not in predictions:
                path = out/row["checkpoint"]
                assert sha(path)==row["sha256"]
                model.load_state_dict(torch.load(path,map_location=split.p.device,weights_only=True))
                predictions[step] = physical_residual(model,inputs,cfg["physical_residual_rms"])
            residual = predictions[step]
            if policy=="native":
                assert torch.equal(split.p+residual,split.p) and not torch.count_nonzero(residual)
            stats, lead_stats, discrepancy = errors(residual,split.p,split.y,data.norm["std"])
            max_identity = max(max_identity,discrepancy)
            stats = {key:v.tolist() for key,v in stats.items()}
            lead_stats = {key:v.tolist() for key,v in lead_stats.items()}
            for i,case in enumerate(split.cases):
                meta = dict(split=split_name,policy=policy,step=step,trajectory=case["file"],
                            start=case["start"],re=case["re"])
                rows.append(dict(meta,**{k:v[i] for k,v in stats.items()}))
                for j in range(20):
                    lead_rows.append(dict(meta,lead=j+1,**{k:v[i][j] for k,v in lead_stats.items()}))
            if split_name=="development" and policy=="selected":
                case = cfg["plot"]
                i = next(i for i,c in enumerate(split.cases) if c["file"]==case["file"] and c["start"]==case["start"])
                t = case["lead"]-1
                true = (split.y[i,:,t]-split.p[i,:,t]).cpu().numpy()
                pred = residual[i,:,t].cpu().numpy()
                # The fields are small and make the Matplotlib panel independently reproducible.
                table(out/"fixed_development_field.csv",
                      [dict(channel=chan,y=y,x=x,truth=float(true[k,y,x]),
                            prediction=float(pred[k,y,x]),remaining=float(true[k,y,x]-pred[k,y,x]))
                       for k,chan in enumerate(("u","v")) for y in range(32) for x in range(64)])
                plot_written = True
            if split_name=="test" and policy=="final":
                old = split.y
                split.y = old.flip(0)
                try:
                    assert not torch.equal(split.y,old)
                    released = split.p+physical_residual(model,fields(split,data.norm),cfg["physical_residual_rms"])
                    assert torch.equal(released,split.p+residual)
                finally:
                    split.y = old
        assert tensor_sha(split.p)==point_digest
    assert plot_written
    for path,digest in ((ROOT/"reliability/checkpoints/current.pt",data.identity["packaged_checkpoint_sha256"]),
                        (WORKSPACE/"runs/physical20_augmented_dynamics_20261109/augmented_diffusion/model.pt",
                         data.identity["original_checkpoint_sha256"])):
        assert sha(path)==digest
    base = ("split","policy","step")
    meta_keys = (*base,"trajectory","start","re")
    metrics = tuple(k for k in rows[0] if k not in meta_keys)
    lead_metrics = tuple(k for k in lead_rows[0] if k not in (*meta_keys,"lead"))
    tracks = average(rows,(*base,"trajectory","re"),metrics)
    regimes = average(tracks,(*base,"re"),metrics)
    lead_tracks = average(lead_rows,(*base,"trajectory","lead"),lead_metrics)
    leads = average(lead_tracks,(*base,"lead"),lead_metrics)
    idx = np.random.default_rng(42).integers(12,size=(10000,12))
    scores = []
    for group, own in groups(tracks,base).items():
        own.sort(key=lambda r:r["trajectory"])
        old = np.array([r["native_mse"] for r in own])
        new = np.array([r["mse"] for r in own])
        n = int(np.sum(new<old))
        summary = next(iter(average(own,base,metrics)))
        lo, hi = None, None
        if group[0]=="test":
            assert len(old)==12
            lo,hi = np.quantile(100*(1-new[idx].mean(1)/old[idx].mean(1)),(.025,.975)).tolist()
        scores.append(dict(summary,improved_trajectories=n,trajectories=len(own),
                           improved_fraction=n/len(own),ci95_low_pct=lo,ci95_high_pct=hi))
    for r in regimes:
        own = [t for t in tracks if all(t[k]==r[k] for k in (*base,"re"))]
        r["improved_trajectories"] = sum(t["mse"]<t["native_mse"] for t in own)
        r["trajectories"] = len(own)
    for name, values in (("scores",scores),("leads",leads),("trajectories",tracks),("regimes",regimes),
                         ("windows",rows),("lead_windows",lead_rows)):
        table(out/f"{name}.csv",[correction_rate(r) for r in values])
    n = 4*(822+24+24)
    assert len(rows)==n and len(lead_rows)==20*n
    assert all(np.isfinite(r[k]) for r in rows for k in metrics)
    winner = next(r for r in scores if r["split"]=="test" and r["policy"]=="selected")
    dev = next(r for r in scores if r["split"]=="development" and r["policy"]=="selected")
    qualifies = bool(winner["step"] and dev["G"]>0 and winner["G"]>0 and winner["improved_fraction"]>.5)
    dump(out/"verification.json",dict(passed=True,frozen_sha256=sha(out/"FROZEN.json"),
                                      max_decomposition_error=max_identity,
                                      zero_exact_native=True,future_intervention_bitwise_equal=True,
                                      source_weights_unchanged=True,window_rows=n,lead_window_rows=20*n,
                                      selected_step=winner["step"],qualifies_for_constraints=qualifies))
    print("TEST SCORED",correction_rate(winner),"follow-up candidate",qualifies,flush=True)
