"""Descriptive stratified tables, original-trajectory pairs, and frozen-weight audit."""
from collections import defaultdict
from core import *

METRICS=["crps","energy","mse","smse","r_mse","r_smse","brier"]
CELL_METRICS=["crps","mse","smse"]
SOURCE_METRICS=["r_mse","r_smse"]


def average(rows,keys,metrics):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    return [dict(zip(keys,key),**{m:float(np.mean([float(r[m]) for r in values if r[m]!=""]))
                                 for m in metrics}) for key,values in groups.items()]


def pairs(rows,keys,metrics,contrasts):
    grouped=average(rows,[*keys,"trajectory","method"],metrics)
    groups=defaultdict(dict)
    for r in grouped:groups[tuple(r[k] for k in keys)][r["trajectory"],r["method"]]=r
    result=[]
    for values,index in groups.items():
        for left,right in contrasts:
            trajectories=sorted({t for t,m in index if m==left})
            if not trajectories:continue
            assert all((t,right) in index for t in trajectories)
            for metric in metrics:
                delta=[index[t,left][metric]-index[t,right][metric] for t in trajectories]
                stats=bootstrap(delta)
                result.append(dict(**dict(zip(keys,values)),contrast=left+" minus "+right,metric=metric,
                    mean_delta=stats["mean_delta"],ci95_low=stats["ci95"][0],ci95_high=stats["ci95"][1],
                    improved=stats["improved"],trajectories=stats["trajectories"]))
    return result


@torch.no_grad()
def ridge_diagnostics():
    """Describe all six predeclared fits without changing any chosen model."""
    operator,_=operators()
    base=torch.load(OUT/"weights/baseline.pt",map_location="cuda",weights_only=False)
    allrows=[];timings=[]
    import time
    for split in ("train","development"):
        data=load_data(split)
        true=encode_source(data.b.double(),operator)
        choices=[("R0",0,None)]
        choices += [(kind,reg,Ridge(torch.load(OUT/"weights"/f"{kind}_{reg}.pt",
                       map_location="cuda",weights_only=False),operator))
                     for kind in ("R1","D1") for reg in CFG["ridge"]]
        for kind,reg,model in choices:
            if model is None:q=base["bias_mean"].float().expand_as(data.b)
            else:q=model(data.x)
            error=encode_source(q.double(),operator)[:,1:]-true[:,1:]
            mse=error.square().mean((-1,-2))
            smse=(error/data.std).square().mean((-1,-2))
            for index,meta in enumerate(data.meta):
                for lead in range(19):
                    for channel,name in enumerate("uv"):
                        allrows.append(dict(split=split,**meta,kind=kind,ridge=reg,lead=lead+2,channel=name,
                                            r_mse=float(mse[index,lead,channel]),r_smse=float(smse[index,lead,channel])))
            if split=="development" and model is not None and reg==1.:
                torch.cuda.synchronize()
                before=time.perf_counter()
                for _ in range(3):model(data.x)
                torch.cuda.synchronize()
                timings.append(dict(kind=kind,ridge=reg,coefficients=88*40,fixed_mean_elements=20*2*32*64,
                    seconds_per_block=(time.perf_counter()-before)/(3*len(data.x)),repetitions=3))
    csvwrite(OUT/"ridge_source_detail.csv",allrows)
    for keys,name in [
        (["split","kind","ridge"],"ridge_source_summary"),
        (["split","kind","ridge","channel"],"ridge_source_channel"),
        (["split","kind","ridge","Re"],"ridge_source_Re"),
        (["split","kind","ridge","lead","channel"],"ridge_source_lead_channel")]:
        csvwrite(OUT/(name+".csv"),average(allrows,keys,SOURCE_METRICS))
    csvwrite(OUT/"inference_cost.csv",timings)


def main():
    configure()
    frozen=json.loads((OUT/"FROZEN.json").read_text())
    for name,expected in frozen["weights"].items():assert digest(OUT/"weights"/name)==expected
    scores=read_csv(OUT/"scores.csv")
    detail=read_csv(OUT/"detail.csv")
    keys=["split","trajectory","start","method","seed"]
    index={tuple(r[k] for k in keys):r for r in scores}
    assert len(index)==len(scores)
    count=10+(9 if frozen["stage_B"]["executed"] else 0)
    assert len(scores)==264*count and len(detail)==len(scores)*40
    cells=average(detail,keys,CELL_METRICS)
    for r in cells:
        original=index[tuple(r[k] for k in keys)]
        for metric in CELL_METRICS:
            assert abs(r[metric]-float(original[metric]))<CFG["numeric"]["old_score_atol"]
    # A selected zero correction is the identical method. Normalize only verified
    # aliases for aggregation, avoiding roundoff-only "improved trajectory" counts.
    audit=json.loads((OUT/"evaluation_audit.json").read_text())
    alias_error=0.
    native={(r["split"],r["trajectory"],r["start"]):r for r in scores if r["method"]=="native"}
    for r in scores:
        name=r["method"]
        alias=name=="zero" or name in ("R1","D1") and frozen["stage_A"]["selected"][name]["deployment"]=="native"
        if alias:
            entry=next(v for v in audit if v["split"]==r["split"] and str(v["start"])==r["start"])
            assert entry["issued_biases"][name+":0"]==entry["issued_biases"]["native:0"]
            base=native[r["split"],r["trajectory"],r["start"]]
            for metric in METRICS:
                error=abs(float(r[metric])-float(base[metric]))
                alias_error=max(alias_error,error)
                assert error<CFG["numeric"]["old_score_atol"]
                r[metric]=base[metric]
    native_cells={(r["split"],r["trajectory"],r["start"],r["lead"],r["channel"]):r
                  for r in detail if r["method"]=="native"}
    for r in detail:
        name=r["method"]
        alias=name=="zero" or name in ("R1","D1") and frozen["stage_A"]["selected"][name]["deployment"]=="native"
        if alias:
            base=native_cells[r["split"],r["trajectory"],r["start"],r["lead"],r["channel"]]
            for metric in CELL_METRICS+SOURCE_METRICS:
                if r[metric]!="":
                    assert abs(float(r[metric])-float(base[metric]))<CFG["numeric"]["old_score_atol"]
                    r[metric]=base[metric]
    devscores=read_csv(OUT/"development_scores_A.csv")
    devcells=read_csv(OUT/"development_detail_A.csv")
    csvwrite(OUT/"development_summary.csv",average(devscores,["method"],METRICS))
    csvwrite(OUT/"development_per_channel.csv",average(devcells,["method","channel"],CELL_METRICS))
    csvwrite(OUT/"development_source_Re.csv",average(devscores,["method","Re"],SOURCE_METRICS))
    for keys,name in [
        (["split","method"],"summary"),(["split","method","seed"],"per_seed"),
        (["split","method","Re"],"per_Re"),(["split","method","trajectory"],"per_trajectory"),
        (["split","method","start"],"per_window")]:
        csvwrite(OUT/(name+".csv"),average(scores,keys,METRICS))
    for keys,name in [
        (["split","method","channel"],"per_channel"),
        (["split","method","lead","channel"],"per_lead_channel"),
        (["split","method","Re","channel"],"per_Re_channel"),
        (["split","method","Re","lead","channel"],"per_Re_lead_channel")]:
        csvwrite(OUT/(name+".csv"),average(detail,keys,CELL_METRICS))
        csvwrite(OUT/("source_"+name+".csv"),
                 average([r for r in detail if r["r_smse"]!=""],keys,SOURCE_METRICS))
    source_gain_rows=[]
    for scope in [[],["Re"],["channel"],["lead"],["Re","lead","channel"]]:
        grouped=average([r for r in detail+devcells if r["r_smse"]!=""],
                        ["split","method",*scope],SOURCE_METRICS)
        reference={(r["split"],*(r[k] for k in scope)):r for r in grouped if r["method"]=="R0"}
        for row in grouped:
            base=reference[(row["split"],)+tuple(row[k] for k in scope)]
            source_gain_rows.append(dict(split=row["split"],method=row["method"],
                scope="overall" if not scope else "+".join(scope),
                subgroup="all" if not scope else "|".join(str(row[k]) for k in scope),
                r_mse=row["r_mse"],r_smse=row["r_smse"],
                raw_gain_percent=100*(1-row["r_mse"]/base["r_mse"]),
                standardized_gain_percent=100*(1-row["r_smse"]/base["r_smse"])))
    csvwrite(OUT/"source_gains.csv",source_gain_rows)
    contrasts=[("R1","D1"),("R1_ridge","D1_ridge"),("R1_ridge","R0"),("D1_ridge","R0"),
               ("R0","native"),("R1","native"),("D1","native"),("phase3","native"),
               ("R1_ridge","native"),("D1_ridge","native")]
    if frozen["stage_B"]["executed"]:
        contrasts += [("T_local","D"),("T_local","T_diag"),("D","native"),("T_local","native"),("T_diag","native")]
    csvwrite(OUT/"paired.csv",pairs(scores,["split"],METRICS,contrasts))
    csvwrite(OUT/"paired_Re.csv",pairs(scores,["split","Re"],METRICS,contrasts))
    csvwrite(OUT/"paired_channel.csv",pairs(detail,["split","channel"],CELL_METRICS,contrasts))
    trajectory_rows=average(scores,["split","trajectory","method"],METRICS)
    trajectory_index={(r["split"],r["trajectory"],r["method"]):r for r in trajectory_rows}
    differences=[]
    for split in ("test","extension"):
        for record in RECORDS["test"]:
            for left,right in contrasts:
                key=(split,record["file"],left)
                if key not in trajectory_index:continue
                reference=trajectory_index[split,record["file"],right]
                differences.append(dict(split=split,trajectory=record["file"],Re=record["group"],
                    contrast=left+" minus "+right,
                    **{metric:trajectory_index[key][metric]-reference[metric] for metric in METRICS}))
    csvwrite(OUT/"trajectory_differences.csv",differences)
    # Only finite, original-trajectory observations determine event counts.
    events=[]
    for split in ("test","extension"):
        selected=[r for r in native.values() if r["split"]==split]
        positive=[r for r in selected if int(r["event"])]
        events.append(dict(split=split,blocks=len(selected),trajectories=len({r["trajectory"] for r in selected}),
            positive_blocks=len(positive),positive_trajectories=len({r["trajectory"] for r in positive}),
            positive_cases=[dict(trajectory=r["trajectory"],start=int(r["start"])) for r in positive]))
    dump(OUT/"events_summary.json",events)
    for r in scores:
        assert all(np.isfinite(float(r[k])) for k in METRICS)
        assert abs(float(r["brier"])-(float(r["probability"])-int(r["event"]))**2)<1e-12
    invariance=read_csv(OUT/"invariance.csv")
    maximum={key:max(float(r[key]) for r in invariance) for key in invariance[0]
             if key.endswith("abs") or key.endswith("bound")}
    assert max(maximum.values())<=CFG["numeric"]["invariance_atol"]
    assert len(audit)==22
    for r in audit:
        assert digest(Path(r["cache"]))==r["cache_sha256"]
        assert r["frozen_selection_sha256"]==digest(OUT/"FROZEN.json")
    # Hashes and provenance of all reused training/development and release caches.
    cache_hashes={}
    for split in ("train","development"):
        for start in CFG["starts"]:
            path=EXTRA/f"{split}_{start}_native.pt"
            cache_hashes[str(path)]=digest(path)
        path=EXTRA/f"{split}_fields.pt"
        cache_hashes[str(path)]=digest(path)
    for r in audit:cache_hashes[r["cache"]]=r["cache_sha256"]
    dump(OUT/"cache_hashes.json",cache_hashes)
    ridge_diagnostics()
    dump(OUT/"verification.json",dict(passed=True,rows=len(scores),detail=len(detail),blocks=264,
         weights_verified=len(frozen["weights"]),neural_training_runs=frozen["stage_B"]["neural_runs"],
         cache_files=len(cache_hashes),invariance_maximum=maximum,zero_alias_roundoff_max=alias_error,
         config_sha256=CONFIG_HASH,frozen_sha256=digest(OUT/"FROZEN.json")))
    print("PASS fixed-model hashes, complete frozen evaluation, source diagnostics, trajectory statistics and invariance")
    print("zero alias maximum numerical score difference",alias_error)
    print("neural training runs",frozen["stage_B"]["neural_runs"])


if __name__=="__main__":
    main()
