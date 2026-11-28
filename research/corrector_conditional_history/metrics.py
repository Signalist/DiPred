"""Paired trajectory summaries for identity, conditional gains and strata."""
import csv
import gzip
from collections import defaultdict
from statistics import mean

import numpy as np
import torch

from research.corrector_incremental_structure.score import REDUCTIONS
from research.corrector_incremental_structure.validation import read_rows
from research.corrector_simple_baseline.fit import table
from research.corrector_simple_baseline.measure import groups

from .diagnostics import observable


def comparisons(frozen):
    def requested(split, arms):
        pairs=[]
        for family in ("IB","ID"):
            for loss in ("MSE","TM1","TM2","EN"):
                normal=f"{family}_{loss}"
                fake=f"{family}_shuffled_{loss}"
                paired=f"{family}_pair_{loss}"
                missing=paired+"__missing"
                pairs.extend((("P1",normal,fake),("P3",normal,f"I0_{loss}"),("P4",paired,missing),
                              ("P5",paired,normal),("P6",missing,f"I0_{loss}"),("shuffled_vs_I0",fake,f"I0_{loss}")))
        pairs.append(("P7",frozen["selection"]["final"],"S3"))
        pairs.append(("P7",frozen["selection"]["selected"],"S3"))
        pairs.extend(("P8","ID_TM1_early",baseline) for baseline in ("ID_TM1","ID_MSE"))
        pairs.extend(("ID_vs_IB",f"ID_{loss}",f"IB_{loss}") for loss in ("MSE","TM1","TM2","EN"))
        return sorted({(tag,a,b) for tag,a,b in pairs if a in arms and b in arms and a!=b})
    return requested


def own_controls(records):
    return {arm:arm.split("_")[0]+"_MSE" for arm in records}


def conditional_rows(split,arm,policy,cases,target,full,missing,basis):
    change=full-missing
    e=target-missing
    corrected=target-full
    quantities=(change*e,change.square(),e.square(),corrected.square())
    channels=("u","v")
    axes={"window":[{}],"lead":[{"lead":i} for i in range(1,21)],"channel":[{"channel":c} for c in channels],
          "lead_channel":[dict(channel=c,lead=i) for c in channels for i in range(1,21)]}
    values={label:torch.stack([x.mean(dim).reshape(len(cases),-1) for x in quantities],dim=-1)
            for label,dim in REDUCTIONS.items()}
    basis=basis.to(target)
    d,b=(item.movedim(2,-1)@basis for item in (change,e))
    values["mode"]=torch.stack([x.mean((2,3)).reshape(len(cases),40)
                                for x in (d*b,d.square(),b.square(),(b-d).square())],dim=-1)
    axes["mode"]=[dict(channel=c,mode=i) for c in channels for i in range(20)]
    rows=[]
    for dimension,stat in values.items():
        inner,energy,baseline,now=stat.unbind(-1)
        gain=2*inner-energy
        assert float((gain-baseline+now).abs().max())<1e-15
        for case,measures in zip(cases,torch.cat((stat,gain[...,None]),dim=-1).tolist()):
            for pos,(c,u,m,f,g) in zip(axes[dimension],measures):
                rows.append(dict(split=split,arm=arm,policy=policy,dimension=dimension,trajectory=case["file"],
                   re=case["re"],start=case["start"],channel=pos.get("channel"),lead=pos.get("lead"),mode=pos.get("mode"),
                   C_H=c,U_H=u,G_H=g,mse_missing=m,mse_full=f))
    return rows


def sensitivity_rows(split,arm,cases,target,correct,wrong,can_move):
    e=lambda a:(a-target).square().mean((1,2,3,4)).tolist()
    return [dict(split=split,arm=arm,trajectory=c["file"],re=c["re"],start=c["start"],
                 eligible=eligible,correct_mse=a,wrong_mse=b)
            for c,eligible,a,b in zip(cases,can_move.tolist(),e(correct),e(wrong))]


def observations(split,view,thresholds,predictions):
    values=observable(view)
    axes=(1,2,3,4)
    errors={arm:(view.target-result).square().mean(axes) for arm,result in predictions.items()}
    contrasts={}
    for loss in ("MSE","EN"):
        left,right=f"ID_{loss}",f"IB_{loss}"
        if left in errors and right in errors:
            contrasts[f"ID_vs_IB_{loss}"]=errors[right]-errors[left]
    for arm,error in errors.items():
        if arm.endswith("__missing"):
            full=arm[:-9]
            assert full in errors
            contrasts[full]=error-errors[full]
    rows=[]
    for label,features in values.items():
        cut=features.new_tensor(thresholds[label])
        bins=torch.bucketize(features,cut).tolist()
        for comparison,gain in contrasts.items():
            for case,band,value in zip(view.data.cases,bins,gain.tolist()):
                rows.append(dict(split=split,feature=label,comparison=comparison,tercile=band+1,
                                 trajectory=case["file"],re=case["re"],gain=value))
    return rows


def paired_metrics(out, draws=10000):
    source=read_rows(out/"trajectories.csv")
    interactions=[]
    family_contrasts=[]
    for (split,policy), cohort in groups(source,("split","policy")).items():
        tracks=defaultdict(dict)
        effects={}
        for r in cohort:
            tracks[r["arm"]][r["trajectory"]]=float(r["mse"])
        for family in ("IB","ID"):
            for loss in ("MSE","EN"):
                correct,fake=f"{family}_{loss}",f"{family}_shuffled_{loss}"
                names=(correct,fake,correct+"__wrong",fake+"__wrong")
                if not all(n in tracks for n in names):
                    continue
                ids=sorted(tracks[correct])
                assert all(set(tracks[n])==set(ids) for n in names)
                regular,trained_wrong,inferred_wrong,both_wrong=(np.array([tracks[n][i] for i in ids]) for n in names)
                a,b=trained_wrong-regular,both_wrong-inferred_wrong
                effect=a-b
                effects[family,loss]=(ids,a,effect)
                samples=np.random.default_rng(42).integers(len(ids),size=(draws,len(ids)))
                low,high=np.quantile(effect[samples].mean(1),[.025,.975])
                interactions.append(dict(split=split,policy=policy,family=family,loss=loss,trajectories=len(ids),
                    correct_correct=float(regular.mean()),wrong_correct=float(trained_wrong.mean()),
                    correct_wrong=float(inferred_wrong.mean()),wrong_wrong=float(both_wrong.mean()),
                    gain_correct_eval=float(a.mean()),gain_wrong_eval=float(b.mean()),interaction=float(effect.mean()),
                    lower=float(low),upper=float(high)))
        for loss in ("MSE","EN"):
            if not all((family,loss) in effects for family in ("IB","ID")):
                continue
            ib_ids,ib_gain,ib_interaction=effects["IB",loss]
            id_ids,id_gain,id_interaction=effects["ID",loss]
            assert id_ids==ib_ids
            samples=np.random.default_rng(42).integers(len(id_ids),size=(draws,len(id_ids)))
            for metric,effect in (("correct_training_gain",id_gain-ib_gain),
                                  ("training_evaluation_interaction",id_interaction-ib_interaction)):
                low,high=np.quantile(effect[samples].mean(1),[.025,.975])
                family_contrasts.append(dict(split=split,policy=policy,loss=loss,metric=metric,
                    trajectories=len(id_ids),id_minus_ib=float(effect.mean()),lower=float(low),upper=float(high)))
    table(out/"history_pairing_interactions.csv",interactions)
    if family_contrasts:
        table(out/"history_pairing_family_contrasts.csv",family_contrasts)


def stratify(out, rows):
    output=[]
    for key,cohort in groups(rows,("split","feature","comparison","tercile")).items():
        tracks=groups(cohort,("trajectory",))
        values=[mean(r["gain"] for r in v) for v in tracks.values()]
        output.append(dict(zip(("split","feature","comparison","tercile"),key),
                           trajectories=len(values),windows=len(cohort),gain=mean(values),
                           improved=sum(x>0 for x in values)))
    table(out/"observable_strata.csv",output)


def summarize_gain(out, rows):
    with gzip.open(out/"history_gain_windows.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    names=("split","arm","policy","dimension","channel","lead","mode")
    metrics=("C_H","U_H","G_H","mse_missing","mse_full")
    track_rows=[dict(zip((*names,"trajectory","re"),identity),
                     **{k:mean(r[k] for r in items) for k in metrics})
                for identity,items in groups(rows,(*names,"trajectory","re")).items()]
    summary=[]
    for scope,extra in (("overall",()),("trajectory",("trajectory","re")),("regime",("re",))):
        fields=(*names,*extra)
        for identity,items in groups(track_rows,fields).items():
            result=dict(zip(fields,identity))
            values={k:mean(r[k] for r in items) for k in metrics}
            assert abs(values["G_H"]-values["mse_missing"]+values["mse_full"])<1e-15
            summary.append(dict(scope=scope,**{k:result.get(k) for k in (*names,"trajectory","re")},**values))
    table(out/"history_gain_decomposition.csv",summary)


def eligible_pairs(out, rows, draws=10000):
    """Repeat the four-cell analysis using only windows with legal alternative tracks."""
    subset=[r for r in rows if r["eligible"]]
    result=[]
    for (split,),cohort in groups(subset,("split",)).items():
        indexed={}
        for (arm,),items in groups(cohort,("arm",)).items():
            indexed[arm]={(v["trajectory"],v["start"]):v for v in items}
        for family in ("IB","ID"):
            for loss in ("MSE","EN"):
                correct,wrong=f"{family}_{loss}",f"{family}_shuffled_{loss}"
                if correct not in indexed or wrong not in indexed:
                    continue
                a,b=indexed[correct],indexed[wrong]
                shared=sorted(set(a)&set(b))
                by_track=defaultdict(list)
                for identity in shared:
                    by_track[identity[0]].append([a[identity]["correct_mse"],b[identity]["correct_mse"],
                                                  a[identity]["wrong_mse"],b[identity]["wrong_mse"]])
                values=np.stack([np.mean(v,axis=0) for v in by_track.values()])
                regular,fake,wrong_eval,both=values.T
                ix=np.random.default_rng(42).integers(len(values),size=(draws,len(values)))
                interaction=(fake-regular)-(both-wrong_eval)
                lo,hi=np.quantile(interaction[ix].mean(1),[.025,.975])
                result.append(dict(split=split,family=family,loss=loss,windows=len(shared),trajectories=len(values),
                    correct_correct=float(regular.mean()),wrong_correct=float(fake.mean()),
                    correct_wrong=float(wrong_eval.mean()),wrong_wrong=float(both.mean()),
                    interaction=float(interaction.mean()),lower=float(lo),upper=float(hi)))
    table(out/"eligible_history_pairing.csv",result)
