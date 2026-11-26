"""One trajectory-equal physical scoring path for all frozen evaluations."""
import csv
import gzip
from collections import defaultdict
from math import sqrt
from statistics import mean

import numpy as np
import torch

from research.corrector_residual_feedback.evaluate import average
from research.corrector_simple_baseline.fit import table
from research.corrector_simple_baseline.measure import groups
from research.corrector_spacetime_optimization.evaluate import independent_mse,rows_for_estimate

from .objectives import operator


def key(row,fields):
    return tuple(row[k] for k in fields)


def aggregate(rows,axes=()):
    keys = ("split","policy","arm",*axes)
    tracks = average(rows,(*keys,"trajectory","re"))
    return average(tracks,keys),tracks,average(tracks,(*keys,"re"))


def correction_rates(rows,axes,controls):
    fields = ("split","policy",*axes)
    lookup = {(key(r,fields),r["arm"]):r["mse"] for r in rows}
    for r in rows:
        ix,arm = key(r,fields),r["arm"]
        plain = controls.get((r["split"],arm),controls.get(arm))
        values = []
        for a in ("Native","S3",plain):
            score = lookup.get((ix,a))
            values.append(100*(1-r["mse"]/score) if score else None)
        r.update(dict(zip(("native_pct","s3_pct","own_pct"),values)))
    return rows


def paired(tracks,comparisons,extra=()):
    result = []
    identity = ("split","policy",*extra)
    for cell,rows in groups(tracks,identity).items():
        split,policy = cell[:2]
        data = {a:{r["trajectory"]:r["mse"] for r in subset} for (a,),subset in groups(rows,("arm",)).items()}
        order = sorted(data["Native"])
        assert all(sorted(r)==order for r in data.values())
        rng = np.random.default_rng(42)
        ix = rng.integers(len(order),size=(10000,len(order)))
        for group,a,b in comparisons(split,data):
            p,q = (np.array([data[arm][t] for t in order],dtype=np.float64) for arm in (a,b))
            values = 100*(1-p[ix].mean(1)/q[ix].mean(1))
            lo,hi = np.percentile(values,[2.5,97.5])
            result.append(dict(**dict(zip(identity,cell)),comparison=group,arm=a,control=b,
                               effect_pct=100*(1-p.mean()/q.mean()),lower_pct=lo,upper_pct=hi,
                               mse_reduction=(q-p).mean(),improved=int((p<q).sum()),trajectories=len(order)))
    return result


def track_columns(rows,identity,values):
    tracks = groups(rows,(*identity,"trajectory"))
    equal = [dict(zip((*identity,"trajectory"),k),**{c:mean(r[c] for r in v) for c in values}) for k,v in tracks.items()]
    return [dict(zip(identity,k),**{c:mean(r[c] for r in v) for c in values})
            for k,v in groups(equal,identity).items()]


class Scores:
    def __init__(self,region_thresholds):
        D = operator()
        self.eigen,self.basis = torch.linalg.eigh(D.T@D)
        self.region_thresholds = region_thresholds
        self.max_identity = 0.
        self.recomputed = []
        self.rows = defaultdict(list)
        self.modes = []
        self.regions = []

    def add(self,data,split,arm,policy,r,std,scale,independent=True):
        assert torch.isfinite(r).all()
        rows,error = rows_for_estimate(data,split,arm,policy,r,std,scale)
        self.max_identity = max(self.max_identity,error)
        if independent:
            self.recomputed.append(dict(split=split,policy=policy,arm=arm,mse=independent_mse(data,r)))
        for k,v in rows.items():
            self.rows[k].extend(v)
        target = data.y.double()-data.p.double()
        prediction = r.double()
        difference = prediction-target
        rmean,tmean = (v.mean((1,2,3,4)).tolist() for v in (prediction,target))
        for row,a,b in zip(rows["window"],rmean,tmean):
            row.update(residual_mean=a,target_mean=b)
        if arm=="Native":
            assert torch.equal(data.p+r,data.p)
        assert float((data.p.double()+target-data.y.double()).abs().max())<1e-14
        eig = self.basis.to(r.device)
        parts = [(x.movedim(2,-1)@eig).square().mean((2,3)) for x in (prediction,target,difference)]
        assert torch.allclose(parts[2].mean(-1),difference.square().mean((2,3,4)),atol=1e-16,rtol=1e-12)
        projected = [v.tolist() for v in parts]
        for i,case in enumerate(data.cases):
            for c,channel in enumerate(("u","v")):
                info = dict(split=split,policy=policy,arm=arm,trajectory=case["file"],channel=channel)
                for j in range(20):
                    self.modes.append(dict(info,mode=j,eigenvalue=float(self.eigen[j]),
                                           band="low" if j<6 else "middle" if j<13 else "high",
                                           predicted=projected[0][i][c][j],truth=projected[1][i][c][j],
                                           squared_error=projected[2][i][c][j]))
        # Compute masked per-window sums and report both coverage and physical contribution.
        for c,channel in enumerate(("u","v")):
            high = target[:,c].abs()>self.region_thresholds[c]
            for label,mask in (("bulk",~high),("tail",high)):
                terms = torch.stack([mask.double().flatten(1).mean(1),
                                     (prediction[:,c]*target[:,c]*mask).flatten(1).mean(1),
                                     (prediction[:,c].square()*mask).flatten(1).mean(1),
                                     (difference[:,c].square()*mask).flatten(1).mean(1)],1).tolist()
                for case,(share,C,U,error) in zip(data.cases,terms):
                    self.regions.append(dict(split=split,policy=policy,arm=arm,trajectory=case["file"],channel=channel,
                                             region=label,coverage=share,C=C,U=U,G=2*C-U,mse_contribution=error))

    def write(self,out,controls,comparisons):
        scores,tracks,regime = aggregate(self.rows["window"])
        lookup = {key(r,("split","policy","arm")):r["mse"] for r in scores}
        for item in self.recomputed:
            item["absolute_error"] = abs(lookup[key(item,("split","policy","arm"))]-item["mse"])
            assert item["absolute_error"]<1e-15
        for row in (*scores,*tracks,*regime):
            U,V,C = (row[k] for k in ("correction_energy","native_mse","C"))
            x,y = (row[k] for k in ("residual_mean","target_mean"))
            row.update(energy_ratio=U/V if V else None,global_cosine=C/sqrt(U*V) if U and V else None,
                       covariance=C-x*y,pearson=(C-x*y)/sqrt((U-x*x)*(V-y*y))
                       if U>x*x and V>y*y else None)
        tables = {"scores":(scores,()),"trajectories":(tracks,("trajectory",)),"regimes":(regime,("re",)),
                  "leads":(aggregate(self.rows["lead"],("lead",))[0],("lead",)),
                  "channels":(aggregate(self.rows["channel"],("channel",))[0],("channel",)),
                  "lead_channels":(aggregate(self.rows["lead_channel"],("lead","channel"))[0],("lead","channel"))}
        for name,(rows,axes) in tables.items():
            table(out/f"{name}.csv",correction_rates(rows,axes,controls))
        contrast = paired(tracks,comparisons)
        table(out/"contrasts.csv",contrast)
        table(out/"holdout_results.csv",[r for r in scores if r["split"].startswith("holdout_")])
        lead,lead_tracks,_ = aggregate(self.rows["lead"],("lead",))
        def primary(split,available):
            return [row for row in comparisons(split,available) if row[0].startswith("P")]
        lead_intervals = paired(lead_tracks,primary,("lead",))
        if lead_intervals:
            table(out/"lead_primary_contrasts.csv",lead_intervals)
        for k,rows in groups(lead,("split","policy","arm")).items():
            assert sorted(r["lead"] for r in rows)==list(range(1,21))
            assert abs(mean(r["mse"] for r in rows)-lookup[k])<1e-15
        for name,rows in (("windows",self.rows["window"]),("lead_windows",self.rows["lead"])):
            with gzip.open(out/f"{name}.csv.gz","wt",newline="") as f:
                writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        dims = ("split","policy","arm","channel")
        modes = track_columns(self.modes,(*dims,"mode","eigenvalue","band"),("predicted","truth","squared_error"))
        regions = track_columns(self.regions,(*dims,"region"),("coverage","C","U","G","mse_contribution"))
        table(out/"temporal_mode_scores.csv",modes)
        table(out/"spatial_region_scores.csv",regions)
        return dict(scores=len(scores),leads=len(lead),contrasts=len(contrast),mode_rows=len(modes),
                    identity_max_error=self.max_identity,independent_recomputation=self.recomputed)
