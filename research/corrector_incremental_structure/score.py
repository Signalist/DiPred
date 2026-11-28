"""Score total and incremental corrections in physical units."""
import csv
import gzip
from math import sqrt
from statistics import mean

import torch

from research.corrector_simple_baseline.fit import table
from research.corrector_simple_baseline.measure import groups
from research.corrector_spacetime_optimization.evaluate import AXES
from research.corrector_spacetime_validation.objectives import operator
from research.corrector_spacetime_validation.score import Scores,aggregate,correction_rates,key,paired,track_columns

REDUCTIONS = {"window":(1,2,3,4),"lead":(1,3,4),"channel":(2,3,4),"lead_channel":(3,4)}


class IncrementalScores(Scores):
    def __init__(self,basis,thresholds):
        super().__init__(thresholds)
        self.basis=basis
        d=operator()
        self.eigen=(d@basis).square().sum(0)
        self.delta_modes=[]
        self.delta_identity=0.

    def add_view(self,view,split,arm,policy,estimate,std):
        offsets={name:len(self.rows[name]) for name in AXES}
        self.add(view.data,split,arm,policy,estimate,std,view.scale)
        e0=view.target-view.current.double()
        delta=estimate.double()-view.current.double()
        error=view.target-estimate.double()
        tensors=(e0.square(),delta*e0,delta.square(),error.square())
        for dim,axes in REDUCTIONS.items():
            rows=self.rows[dim][offsets[dim]:]
            b,c,u,m=[v.mean(axes).reshape(-1) for v in tensors]
            g=2*c-u
            gap=float((b-m-g).abs().max())
            self.delta_identity=max(self.delta_identity,gap)
            assert gap<1e-15
            window_cos=c/(b*u).sqrt().clamp_min(1e-30)
            values=torch.stack((b,c,u,g,window_cos),dim=1).tolist()
            assert len(rows)==len(values)
            for row,v in zip(rows,values):
                row.update(dict(zip(("s3_mse","C_delta","U_delta","G_delta","delta_cosine"),v)))
        basis=self.basis.to(delta.device)
        energies=[(v.movedim(2,-1)@basis).square().mean((2,3)) for v in (delta,e0,error)]
        assert torch.allclose(energies[-1].mean(-1),error.square().mean((2,3,4)),atol=1e-16,rtol=1e-12)
        values=torch.stack(energies,dim=-1).tolist()
        for case,window in zip(view.data.cases,values):
            for channel,components in zip(("u","v"),window):
                for j,(delta_energy,remaining_energy,error_energy) in enumerate(components):
                    self.delta_modes.append(dict(split=split,arm=arm,policy=policy,trajectory=case["file"],
                                                 channel=channel,mode=j,delta_energy=delta_energy,
                                                 s3_error_energy=remaining_energy,final_error_energy=error_energy))

    def write(self,out,controls,comparisons,shard_raw=False):
        dims=("split","policy","arm")
        scores,trajectories,regimes=aggregate(self.rows["window"])
        leads,lead_tracks,_=aggregate(self.rows["lead"],("lead",))
        channels,_,_=aggregate(self.rows["channel"],("channel",))
        lead_channels,_,_=aggregate(self.rows["lead_channel"],("lead","channel"))
        tables=dict(scores=scores,trajectories=trajectories,regimes=regimes,leads=leads,
                    channels=channels,lead_channels=lead_channels)
        lookup={key(r,dims):r["mse"] for r in scores}
        for r in self.recomputed:
            r["absolute_error"]=abs(lookup[key(r,dims)]-r["mse"])
            assert r["absolute_error"]<1e-15
        for name,rows in tables.items():
            first=rows[0]
            axes=tuple(k for k in ("trajectory","re","lead","channel") if k in first)
            if name=="trajectories":
                axes=("trajectory",)
            for r in rows:
                u,b,c=r["U_delta"],r["s3_mse"],r["C_delta"]
                r["global_delta_cosine"]=c/sqrt(u*b) if u and b else None
                v,a,t=r["correction_energy"],r["native_mse"],r["C"]
                r["global_cosine"]=t/sqrt(v*a) if v and a else None
                mean_r,mean_t=r.get("residual_mean"),r.get("target_mean")
                r["energy_ratio"]=v/a if a else None
                if mean_r is not None:
                    covariance=t-mean_r*mean_t
                    r["covariance"]=covariance
                    r["pearson"]=covariance/sqrt((v-mean_r**2)*(a-mean_t**2)) if v>mean_r**2 and a>mean_t**2 else None
                assert abs(b-r["mse"]-r["G_delta"])<1e-15
            table(out/f"{name}.csv",correction_rates(rows,axes,controls))
        holdout=[r for r in scores if r["split"].startswith("holdout_")]
        if holdout:
            table(out/"holdout_results.csv",holdout)
        contrasts=paired(trajectories,comparisons)
        table(out/"contrasts.csv",contrasts)
        def primary(split,arms):
            return [r for r in comparisons(split,arms) if r[0].startswith("P")]
        lead_pairs=paired(lead_tracks,primary,("lead",))
        table(out/"lead_primary_contrasts.csv",lead_pairs)
        for cell,rows in groups(leads,dims).items():
            assert sorted(r["lead"] for r in rows)==list(range(1,21))
            assert abs(mean(r["mse"] for r in rows)-lookup[cell])<1e-15
        tracks=groups(trajectories,dims)
        by_re=groups(regimes,dims)
        risks=[]
        for identity,rows in tracks.items():
            gains=sorted((r["G_delta"] for r in rows if r["G_delta"]>0),reverse=True)
            worst=min(by_re[identity],key=lambda r:r["s3_pct"])
            risks.append(dict(zip(dims,identity),improved_tracks=len(gains),trajectories=len(rows),
                              worst_re=worst["re"],worst_re_s3_pct=worst["s3_pct"],
                              top_two_share_of_positive_gain=sum(gains[:2])/sum(gains) if gains else None))
        table(out/"trajectory_risks.csv",risks)
        modes=track_columns(self.delta_modes,("split","policy","arm","channel","mode"),
                            ("delta_energy","s3_error_energy","final_error_energy"))
        table(out/"incremental_mode_scores.csv",modes)
        spectral=track_columns(self.modes,(*dims,"channel","mode","eigenvalue","band"),
                               ("predicted","truth","squared_error"))
        table(out/"temporal_mode_scores.csv",spectral)
        spatial=track_columns(self.regions,(*dims,"channel","region"),("coverage","C","U","G","mse_contribution"))
        table(out/"spatial_region_scores.csv",spatial)
        for name,rows in (("windows",self.rows["window"]),("lead_windows",self.rows["lead"])):
            cohorts=groups(rows,("split",)) if shard_raw else {():rows}
            for partition,items in cohorts.items():
                filename="_".join((name,*partition))
                with gzip.open(out/f"{filename}.csv.gz","wt",newline="") as handle:
                    writer=csv.DictWriter(handle,fieldnames=list(items[0]),lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(items)
        return dict(scores=len(scores),leads=len(leads),contrasts=len(contrasts),mode_rows=len(spectral),
                    identity_max_error=self.max_identity,delta_identity_max_error=self.delta_identity,
                    independent_recomputation=self.recomputed)
