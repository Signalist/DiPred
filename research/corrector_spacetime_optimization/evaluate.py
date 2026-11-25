"""Score all fitted arms at the development choice and at update 6000."""
import csv
import gzip
from io import TextIOWrapper
from collections import defaultdict
from statistics import mean

import numpy as np
import torch

from research.corrector.run import dump,sha
from research.corrector_consistency.model import COS_FLOOR,MASK_THRESHOLD,neighbors
from research.corrector_residual_feedback.data import PRIOR,guard
from research.corrector_residual_feedback.evaluate import average,measurements,time_model
from research.corrector_residual_feedback.model import predict
from research.corrector_residual_optimization.fit import load as load_s3,predict as predict_s3
from research.corrector_simple_baseline.fit import table
from research.corrector_simple_baseline.measure import groups
from research.corrector_window_study.data import ROOT,WORKSPACE,read_json,tensor_sha

from .data import BRANCHES,LAST
from .fit import checked_net

AXES = {"window":(),"lead":("lead",),"channel":("channel",),"lead_channel":("channel","lead")}
SPLITS = ("development","test")
POLICIES = ("selected","final")


def temporal_and_spatial(r,target,scale):
    """Physical difference energies and true-masked adjacent direction/cosine."""
    temporal = {}
    for order,prefix in ((1,"first"),(2,"second")):
        for name,field in (("predicted",r),("true",target)):
            temporal[f"{prefix}_{name}_energy"] = (field.diff(n=order,dim=2).square(),order)
    nr = (r/scale).norm(dim=1,keepdim=True)
    nt = (target/scale).norm(dim=1,keepdim=True)
    unit,truth = (r/scale)/nr.clamp_min(COS_FLOOR),(target/scale)/nt.clamp_min(COS_FLOOR)
    valid = nt>MASK_THRESHOLD
    shape = (len(r),r.shape[2])
    spatial = {key:r.new_zeros(shape) for key in (
        "adjacent_predicted_cosine","adjacent_true_cosine","adjacent_direction_match",
        "adjacent_predicted_magnitude_difference","adjacent_true_magnitude_difference",
        "adjacent_magnitude_match")}
    counts = r.new_zeros(shape)
    for (a,b),(c,d),(m,n),(p,q),(s,t) in zip(
            neighbors(unit),neighbors(truth),neighbors(valid),neighbors(nr),neighbors(nt)):
        mask = (m&n).squeeze(1)
        pred_cos = (a*b).sum(1).clamp(-1,1)
        true_cos = (c*d).sum(1).clamp(-1,1)
        pred_mag = scale*(p-q).squeeze(1)
        true_mag = scale*(s-t).squeeze(1)
        values = (pred_cos,true_cos,(pred_cos-true_cos).square(),pred_mag.abs(),
                  true_mag.abs(),(pred_mag-true_mag).square())
        counts += mask.sum((-2,-1))
        for key,value in zip(spatial,values):
            spatial[key] += (mask*value).sum((-2,-1))
    return temporal,spatial,counts


def rows_for_estimate(data,split,arm,policy,r,std,scale):
    details,gap = measurements(r,data.p,data.y,std)
    temporal,spatial,counts = temporal_and_spatial(r.double(),data.y.double()-data.p.double(),scale)
    result = {}
    for dim,axes in (("window",(1,2,3,4)),("lead",(1,3,4)),("channel",(2,3,4)),
                     ("lead_channel",(3,4))):
        terms = details[dim]
        result[dim] = []
        for key,(energy,order) in temporal.items():
            value = energy.mean(axes)
            if "lead" in AXES[dim]:
                value = torch.cat((value.new_full((*value.shape[:-1],order),float("nan")),value),dim=-1)
            terms[key] = value.tolist()
        if dim=="window":
            for key,value in spatial.items():
                terms[key] = (value.sum(1)/counts.sum(1).clamp_min(1)).tolist()
        if dim=="lead":
            for key,value in spatial.items():
                terms[key] = (value/counts.clamp_min(1)).tolist()
        indices = {
            "window":[((),{})],
            "lead":[((t,),dict(lead=t+1)) for t in range(20)],
            "channel":[((i,),dict(channel=c)) for i,c in enumerate(("u","v"))],
            "lead_channel":[((i,t),dict(channel=c,lead=t+1)) for i,c in enumerate(("u","v")) for t in range(20)],
        }[dim]
        for i,case in enumerate(data.cases):
            meta = dict(split=split,policy=policy,arm=arm,trajectory=case["file"],re=case["re"],start=case["start"])
            for coord,labels in indices:
                parts = {}
                for key,values in terms.items():
                    value = values[i]
                    for j in coord:
                        value = value[j]
                    if not np.isfinite(value):
                        assert (dim in ("lead","lead_channel") and key.startswith(("first_","second_"))
                                and labels["lead"]<=(1 if key.startswith("first_") else 2))
                        value = None
                    parts[key] = value
                result[dim].append(dict(meta,**labels,**parts))
    return result,gap


def comparators(arm):
    branch = arm.split("_")[0]
    return ("Native","S3",f"{branch}_MSE") if branch in BRANCHES else ("Native","S3")


def rates(rows,axes):
    keys = ("split","policy",*axes)
    lookup = {(tuple(r[k] for k in keys),r["arm"]):r["mse"] for r in rows
              if r["arm"] in ("Native","S3","B_MSE","E_MSE")}
    for r in rows:
        k = tuple(r[f] for f in keys)
        native,s3,*own = [lookup[k,ref] for ref in comparators(r["arm"])]
        assert abs(native-r["native_mse"])<1e-15
        r.update(native_pct=100*(1-r["mse"]/native),s3_pct=100*(1-r["mse"]/s3),
                 own_pct=100*(1-r["mse"]/own[0]) if own else None)
    return rows


def contrasts(tracks):
    result = []
    for (split,policy),rows in groups(tracks,("split","policy")).items():
        by_arm = {key[0]:{r["trajectory"]:r["mse"] for r in subset}
                  for key,subset in groups(rows,("arm",)).items()}
        order = sorted(by_arm["Native"])
        assert all(sorted(v)==order for v in by_arm.values())
        x = {name:np.array([v[f] for f in order],dtype=np.float64) for name,v in by_arm.items()}
        n = len(order)
        sampled = np.random.default_rng(42).integers(n,size=(10000,n))
        for arm,a in x.items():
            for control in comparators(arm):
                b = x[control]
                diff = b-a
                gain = 100*(1-a[sampled].mean(1)/b[sampled].mean(1))
                low,high = np.percentile(gain,[2.5,97.5])
                raw_low,raw_high = np.percentile(diff[sampled].mean(1),[2.5,97.5])
                result.append(dict(split=split,policy=policy,arm=arm,control=control,
                                   gain_pct=100*(1-a.mean()/b.mean()),lower_pct=low,upper_pct=high,
                                   mse_reduction=diff.mean(),lower_reduction=raw_low,upper_reduction=raw_high,
                                   improved=int((diff>0).sum()),trajectories=n))
    return result


def independent_mse(data,r):
    p,y,residual = (np.asarray(v.cpu(),dtype=np.float64) for v in (data.p,data.y,r))
    mse = np.mean(np.square(p+residual-y),axis=(1,2,3,4))
    tracks = defaultdict(list)
    for case,error in zip(data.cases,mse):
        tracks[case["file"]].append(error)
    return float(np.mean([np.mean(v) for v in tracks.values()]))


def retain_fields(fields,arm,r,data,case,i):
    t = case["lead"]-1
    actual = (data.y[i,:,t].double()-data.p[i,:,t].double()).cpu().numpy()
    value = r[i,:,t].double().cpu().numpy()
    fields["truth"] = actual
    fields[arm] = value
    fields[f"{arm}_remaining"] = actual-value


def save_fields(out,fields):
    shape = fields["truth"].shape
    table(out/"fixed_fields.csv",[
        dict(channel=name,y=y,x=x,**{k:float(v[c,y,x]) for k,v in fields.items()})
        for c,name in enumerate(("u","v")) for y in range(shape[1]) for x in range(shape[2])])


def results_tables(out,dimensions):
    filenames = {
        "window":("trajectories","regimes","scores"),
        "lead":("lead_trajectories","regime_leads","leads"),
        "channel":("channel_trajectories","channel_regimes","channels"),
        "lead_channel":("lead_channel_trajectories","lead_channel_regimes","lead_channels"),
    }
    scores = None
    for dim,rows in dimensions.items():
        axes = AXES[dim]
        identity = ("split","policy","arm",*axes)
        tracks = average(rows,(*identity,"trajectory","re"))
        regimes = average(tracks,(*identity,"re"))
        total = average(tracks,identity)
        if dim=="window":
            comparison = contrasts(tracks)
            table(out/"contrasts.csv",comparison)
            index = {(v["split"],v["policy"],v["arm"],v["control"]):v for v in comparison}
            for row in total:
                key = tuple(row[k] for k in identity)
                for control in comparators(row["arm"]):
                    v = index[(*key,control)]
                    suffix = "own" if control in ("B_MSE","E_MSE") else control.lower()
                    for k in ("improved","trajectories","lower_pct","upper_pct"):
                        row[f"{k}_{suffix}"] = v[k]
                for k in ("improved","trajectories","lower_pct","upper_pct"):
                    row.setdefault(f"{k}_own",None)
            scores = total
            table(out/"windows.csv",rates(rows,(*axes,"trajectory","start")))
        elif dim=="lead":
            with (out/"lead_windows.csv.gz").open("wb") as raw:
                with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as binary:
                    with TextIOWrapper(binary,newline="") as f:
                        rates(rows,(*axes,"trajectory","start"))
                        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
                        writer.writeheader()
                        writer.writerows(rows)
        for name,data,labels in zip(filenames[dim],(tracks,regimes,total),((*axes,"trajectory"),(*axes,"re"),axes)):
            table(out/f"{name}.csv",rates(data,labels))
    return scores


@torch.inference_mode()
def score(study):
    out = study.out
    frozen = guard(out)
    freeze_sha = sha(out/"FROZEN.json")
    choice = frozen["selections"]
    assert all(sha(out/path)==digest for path,digest in frozen["checkpoints"].items())
    prior = study.prior["reference"]
    source = read_json(out/"sources.json")
    config = read_json(out/"config.json")
    ref = load_s3(prior["config"],prior["selected"],PRIOR,study.norm["mean"].device)
    release = [choice["best_extension"]] if choice["best_extension"] else []
    finalists = {"Native","S3","B_MSE","E_MSE",*choice["best"].values(),*release}
    finalists.update(v["best"] for b in BRANCHES for v in choice["single"][b].values())
    finalists.update(choice["joint"][b]["best"] for b in BRANCHES)
    dimensions = {k:[] for k in AXES}
    gap,restoration = 0.,0.
    estimates,interventions,times,fields = [],[],[],{}
    for split in SPLITS:
        data,current,design = study.get(split)
        initial = tensor_sha(data.p)
        assert initial==study.guards[split]["current_native"]
        common = dict(Native=torch.zeros_like(data.p),S3=predict_s3(ref,current,study.cfg["external_rms"]))
        assert torch.equal(data.p+common["Native"],data.p)
        target = data.y.double()-data.p.double()
        restoration = max(restoration,float((data.p.double()+target-data.y.double()).abs().max()))
        case = config["case"]
        if split=="development":
            times.append(dict(arm="Native",parameters=0,ms_per_window_batch8=0.,train_seconds=0.))
            times.append(dict(arm="S3",parameters=sum(p.numel() for p in ref.parameters()),
                              ms_per_window_batch8=time_model(ref,current[:8]),train_seconds=None))
            i, = [i for i,c in enumerate(data.cases) if (c["file"],c["start"])==(case["file"],case["start"])]
        for arm in (*common,*choice["models"]):
            branch = arm.split("_")[0]
            fit = choice["models"].get(arm)
            last = None
            for policy in POLICIES:
                checkpoint = fit[policy] if fit else None
                if fit:
                    z = design[branch]
                    if last!=checkpoint["sha256"]:
                        net = checked_net(checkpoint,out,z.device)
                        r = predict(net,z,study.scale)
                    last = checkpoint["sha256"]
                    if split=="development" and policy=="selected":
                        times.append(dict(arm=arm,parameters=sum(p.numel() for p in net.parameters()),
                                          ms_per_window_batch8=time_model(net,z[:8]),
                                          train_seconds=fit["elapsed_seconds"]))
                else:
                    r = common[arm]
                if policy=="selected" and arm in finalists:
                    if split=="development":
                        retain_fields(fields,arm,r,data,case,i)
                    estimates.append(dict(split=split,arm=arm,mse=independent_mse(data,r)))
                    if split=="test":
                        saved = data.y
                        data.y = saved.flip(0)
                        try:
                            assert not torch.equal(data.y,saved)
                            intervened,inputs = study.inputs(data)
                            assert torch.equal(intervened,current)
                            assert all(torch.equal(inputs[b],design[b]) for b in BRANCHES)
                            if arm=="Native":
                                issued = data.p
                            elif arm=="S3":
                                issued = data.p+predict_s3(ref,intervened,study.cfg["external_rms"])
                            else:
                                issued = data.p+predict(net,inputs[branch],study.scale)
                            assert torch.equal(data.p+r,issued)
                        finally:
                            data.y = saved
                        interventions.append(arm)
                rows,error = rows_for_estimate(data,split,arm,policy,r,study.norm["std"],study.scale)
                gap = max(gap,error)
                for dim in AXES:
                    dimensions[dim].extend(rows[dim])
        assert tensor_sha(data.p)==initial
    scores = results_tables(out,dimensions)
    baseline = {(r["split"],r["policy"],r["arm"]):r["mse"] for r in scores}
    for item in estimates:
        item["absolute_error"] = abs(item["mse"]-baseline[item["split"],"selected",item["arm"]])
        assert item["absolute_error"]<1e-15
    history = {(r["split"],r["policy"],r["arm"]):float(r["mse"])
               for r in csv.DictReader((LAST/"scores.csv").open()) if r["split"] in SPLITS}
    historical = []
    for split in SPLITS:
        for policy in POLICIES:
            for old,new in (("Native","Native"),("S3","S3"),("B","B_MSE"),("E","E_MSE")):
                error = abs(history[split,policy,old]-baseline[split,policy,new])
                assert error<1e-12,(split,policy,new,error)
                historical.append(dict(split=split,policy=policy,arm=new,absolute_error=error))
    for path,name in ((ROOT/"reliability/checkpoints/current.pt","packaged_checkpoint_sha256"),
                      (WORKSPACE/"runs/physical20_augmented_dynamics_20261109/augmented_diffusion/model.pt",
                       "original_checkpoint_sha256")):
        assert sha(path)==source["backbone"][name]
    assert restoration<1e-14 and gap<1e-15 and set(interventions)==finalists
    assert sha(out/"FROZEN.json")==freeze_sha
    save_fields(out,fields)
    table(out/"inference_cost.csv",times)
    dump(out/"verification.json",dict(passed=True,freeze_sha256=freeze_sha,
                                      source_weights_unchanged=True,zero_exact=True,
                                      restoration_max_error=restoration,score_max_identity_error=gap,
                                      future_mutation_bitwise=sorted(interventions),numpy_recomputations=estimates,
                                      historical_baselines=historical,native_identity=study.guards,
                                      window_rows=len(dimensions["window"]),lead_rows=len(dimensions["lead"])))
    print("SCORED",len(dimensions["window"]),"window rows",flush=True)
