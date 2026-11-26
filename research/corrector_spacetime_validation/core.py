"""Strict reuse, fixed mechanism fits, development-only selection and freeze."""
import csv
from statistics import mean

import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.model import predict
from research.corrector_simple_baseline.fit import table
from research.corrector_window_study.data import read_json,tensor_sha
from research.corrector_spacetime_optimization.fit import config_hash

from .data import HISTORICAL,PREVIOUS,ROOT
from .fit import Fits,atomic_json,checkpoint_file,load_net
from .gradients import calibrate,monitor
from .objectives import KINDS,RHO,verify_math
from .run import s3_prefix


def located(name,record,root,kind,shared=None):
    def bind(row):
        return dict(row,path=str(checkpoint_file(root,row,shared).relative_to(ROOT)))
    return dict(name=name,config=record["config"],config_hash=record["config_hash"],
                selected=bind(record["selected"]),final=bind(record["final"]),training=[bind(r) for r in record["training"]],
                elapsed_seconds=record["elapsed_seconds"],kind=kind)


def legacy(study,name):
    old = read_json(PREVIOUS/f"run_{name}.json")
    b = old["config"]["branch"]
    cfg = old["config"]
    assert old["config_hash"]==config_hash(cfg)
    assert name in study.registry["selections"]["models"]
    assert all(old[k]==v for k,v in study.registry["selections"]["models"][name].items())
    assert cfg["scale"]==study.scale and cfg["seed"]==42 and cfg["updates"]==6000
    assert cfg["lr"]==.001 and cfg["weight_decay"]==1e-5
    assert cfg["schedule_sha256"]==tensor_sha(study.schedule)
    assert cfg["previous_train_input"]==study.old_runs[b]["config"]["train_input"]
    assert cfg["previous_dev_input"]==study.old_runs[b]["config"]["dev_input"]
    if cfg["weights"]:
        assert cfg["shared_sha256"]==sha(PREVIOUS/f"shared_{b}.pt") and cfg["first"]==1000
    assert all(sha(PREVIOUS/r["checkpoint"])==r["sha256"] for r in old["training"])
    return located(name,old,PREVIOUS,"reused")


def old_snapshot(study):
    """Freeze historic evidence by reference and only extract development rows before validation."""
    paths = (PREVIOUS/"FROZEN.json",PREVIOUS/"verification.json",PREVIOUS/"artifact_audit.json",PREVIOUS/"scores.csv",
             PREVIOUS/"leads.csv",PREVIOUS/"trajectories.csv",PREVIOUS/"regimes.csv")
    certificate = read_json(paths[1])
    assert certificate["freeze_sha256"]==sha(paths[0]) and certificate["passed"]
    assert read_json(paths[2])["freeze_sha256"]==sha(paths[0])
    assert all(sha(PREVIOUS/k)==v for k,v in read_json(paths[2])["scored_sha256"].items())
    return {str(f.relative_to(ROOT)):sha(f) for f in paths}


def track_mse(value,data):
    tracks = {}
    for case,score in zip(data.cases,value.tolist()):
        tracks.setdefault(case["file"],dict(re=case["re"],mse=[]))["mse"].append(score)
    return {k:dict(re=r["re"],mse=mean(r["mse"])) for k,r in tracks.items()}


def dev_table(records,dev,dz):
    baseline = track_mse((dev.y.double()-dev.p.double()).square().mean((1,2,3,4)),dev)
    result = {}
    for label,rr in records.items():
        config = rr["config"]
        z = dz[config["branch"]]
        result[label] = {}
        for policy in ("selected","final"):
            r = rr[policy]
            net = load_net(ROOT/r["path"],r["sha256"],config["branch"],z.device)
            residual = predict(net,z,config["scale"])
            ms = (dev.p.double()+residual.double()-dev.y.double()).square().mean((1,2,3,4))
            result[label][policy] = track_mse(ms,dev)
            assert abs(mean(v["mse"] for v in result[label][policy].values())-r["dev_mse"])<1e-12
    return baseline,result


def rank_rules(names,values,native,keys,plain):
    def stats(name):
        tracks = values[name]
        overall = mean(tracks[k]["mse"] for k in keys)
        regimes = {}
        for k in keys:
            regimes.setdefault(native[k]["re"],[]).append(k)
        ratio = [sum(tracks[k]["mse"] for k in g)/sum(native[k]["mse"] for k in g) for g in regimes.values()]
        return overall,mean(ratio),max(ratio)
    stats_by_name = {n:stats(n) for n in names}
    available = [n for n in names if stats_by_name[n][0]<=stats_by_name[plain][0]]
    return dict(A=min(names,key=lambda n:(stats_by_name[n][0],n)),
                B=min(names,key=lambda n:(stats_by_name[n][1],stats_by_name[n][0],n)),
                C=min(available,key=lambda n:(stats_by_name[n][2],stats_by_name[n][0],n)))


def sensitivity(out,native,dev,records):
    result,rows = {},[]
    all_keys = tuple(sorted(native))
    historical = {k:{} for k in ("selected","final")}
    with (PREVIOUS/"trajectories.csv").open() as handle:
        for r in csv.DictReader(handle):
            if r["split"]=="development" and r["arm"].startswith(("B_","E_")):
                historical[r["policy"]].setdefault(r["arm"],{})[r["trajectory"]] = dict(
                    re=float(r["re"]),mse=float(r["mse"]))
    re_values = sorted({p["re"] for p in native.values()})
    folds = [("trajectory",k,[k]) for k in all_keys]+[
        ("regime",re,[k for k in all_keys if native[k]["re"]==re]) for re in re_values]
    for policy in ("selected","final"):
        current = {n:dev[n][policy] for n in dev}
        for branch in ("B","E","S3"):
            for scope in ("core","expanded","historical_grid"):
                if scope=="historical_grid":
                    if branch=="S3":
                        continue
                    plain,values = f"{branch}_MSE",historical[policy]
                    names = sorted(n for n in values if n.startswith(branch+"_"))
                    assert len(names)==32
                    assert all(set(values[n])==set(all_keys) for n in names)
                else:
                    plain,values = f"{branch}_M0",current
                    names = sorted(n for n,r in records.items() if r["config"]["branch"]==branch and
                                   (scope=="expanded" or r["kind"]!="historical" and "_energy_match_" not in n))
                selected = rank_rules(names,values,native,all_keys,plain)
                if scope=="expanded":
                    result.setdefault(branch,{})[policy] = selected
                for kind,excluded,held in folds:
                    active = [k for k in all_keys if k not in held]
                    sub = rank_rules(names,values,native,active,plain)
                    for rule,n in sub.items():
                        p,a,b = (mean(metric[k]["mse"] for k in held)
                                 for metric in (values[n],values[plain],native))
                        rows.append(dict(branch=branch,scope=scope,policy=policy,rule=rule,excluded=excluded,
                                         kind=kind,picked=n,full_pick=selected[rule],held_mse=p,own_mse=a,native=b,
                                         held_vs_plain=100*(1-p/a),held_vs_native=100*(1-p/b),candidate_count=len(names)))
    table(out/"selection_sensitivity.csv",rows)
    atomic_json(out/"development_tracks.json",dict(native=native,models=dev))
    return result


def core(study,prepared):
    from .expansion import plan_periods
    train,dev,z,dz = prepared
    out = study.out
    s3,result = s3_prefix(study,prepared)
    records = {"S3_M0":located("S3_M0",result,out,"new")}
    old_references = old_snapshot(study)
    history = {}
    for name in HISTORICAL:
        r = legacy(study,name)
        history[name] = r
        records["historical_"+name] = dict(r,kind="historical")
    for b in ("B","E"):
        records[f"{b}_JOINT_original"] = legacy(study,f"{b}_JOINT")
    reused,search,energy_rows = {},[],[]
    # The historic S3 is the external seed and exact replay, with its own path kept in sources.json.
    for b in ("S3","B","E"):
        engine = s3 if b=="S3" else Fits(out,study.cache,train,z[b],study.scale,study.schedule,study.initial,dev,dz[b])
        shared = (out if b=="S3" else PREVIOUS)/f"shared_{b}.pt"
        if b!="S3":
            records[f"{b}_M0"] = legacy(study,f"{b}_MSE")
        plain = records[f"{b}_M0"]
        at1000 = next(r for r in plain["training"] if r["step"]==1000)
        net = load_net(ROOT/at1000["path"],at1000["sha256"],b,train.p.device)
        terms = ("energy","match","match2") if b=="S3" else ("energy","velocity","acceleration","match","match2")
        coeff,norm = calibrate(net,z[b],engine.labels,train.cases,terms)

        def fit(tag,kind,weights,coordinate=None,reuse=None):
            n = f"{b}_{tag}"
            if reuse:
                record = legacy(study,reuse)
                assert record["config"]["weights"].keys()==weights.keys()
                assert all(abs(record["config"]["weights"][k]-v)<1e-12 for k,v in weights.items())
                reused[n] = dict(run=reuse,configuration=record["config"],checkpoint=record["final"]["sha256"],
                                 prior_freeze=study.registry["bindings"][f"run_{reuse}.json"],
                                 optimizer_source=record["config"].get("shared_sha256"))
            else:
                record = located(n,engine.fit(n,b,weights,shared=shared,origin=kind),out,"new",shared=shared)
            records[n] = record
            chosen,end = record["selected"],record["final"]
            search.append(dict(model=n,source=record["name"],branch=b,objective=kind,coordinate=coordinate,
                               weights=str(record["config"]["weights"]),selected_step=chosen["step"],
                               selected_mse=chosen["dev_mse"],final_mse=end["dev_mse"],
                               train_U_selected=chosen["train_E"],train_U_final=end["train_E"]))
            table(out/"mechanism_search.csv",search)
            return n

        ratios = (.3,) if b=="S3" else RHO
        by_kind = {}
        for kind,term in dict(EN="energy",**KINDS).items():
            if b=="S3" and kind not in ("EN","TM1","TM2"):
                continue
            by_kind[kind] = []
            for rho in ratios:
                reuse = None
                if b!="S3":
                    if kind in ("TS1","TS2","TM1"):
                        reuse = f"{b}_{dict(TS1='VEL',TS2='ACC',TM1='MATCH')[kind]}_rho_{rho}"
                    elif kind=="TM2" and rho==.3:
                        reuse = f"{b}_ext_match2"
                by_kind[kind].append(fit(f"{kind}_{rho}",kind,{term:coeff[term][rho]},rho,reuse))
        atomic_json(out/f"core_map_{b}.json",by_kind)

        if b!="S3":
            full = legacy(study,f"{b}_JOINT_tuned")
            records[f"{b}_J5"] = full
            w = full["config"]["weights"]
            assert set(w)=={"direction","magnitude","velocity","acceleration","match"}
            fit("J4","J4",{k:v for k,v in w.items() if k!="match"})
            if b=="E":
                anchor = history["E_ext_match2"]["config"]["weights"]["match2"]
                fit("DIAG","DIAG",{"diag":anchor})
                fit("PERM","PERM",{"perm":anchor})
            best = lambda names:min(names,key=lambda n:(records[n]["selected"]["dev_mse"],n))
            target = records[best(by_kind["TM2"])]["final"]["train_E"]
            energy_arms = by_kind["EN"][:]
            for attempt in range(2):
                close = min(energy_arms,key=lambda n:abs(records[n]["final"]["train_E"]/target-1))
                if abs(records[close]["final"]["train_E"]/target-1)<=.05:
                    break
                points = [(0.,plain["final"]["train_E"])]+[
                    (records[n]["config"]["weights"]["energy"],records[n]["final"]["train_E"]) for n in energy_arms]
                pairs = [(x,y) for x in points for y in points if x[0]<y[0] and x[1]>y[1] and x[1]>=target>=y[1]]
                if pairs:
                    (x,u),(y,v) = min(pairs,key=lambda p:p[1][0]-p[0][0])
                    weight = x+(y-x)*(u-target)/(u-v)
                else:
                    x,u = min(points,key=lambda p:abs(p[1]-target))
                    weight = 2*x if u>target else x/2
                    if weight==0:
                        weight = min(k for k,_ in points if k>0)/4
                if not all(abs(weight-x)>1e-9 for x,_ in points) or not 0<weight<=100:
                    break
                tag = f"energy_match_{attempt+1}"
                atomic_json(out/f"{b}_{tag}_plan.json",dict(weight=weight,training_U=points,target=target))
                energy_arms.append(fit(tag,"EN_train_U",{"energy":weight}))
            for policy in ("selected","final"):
                target_id = best(by_kind["TM2"])
                target_u = records[target_id][policy]["train_E"]
                for n in energy_arms:
                    u = records[n][policy]["train_E"]
                    energy_rows.append(dict(branch=b,policy=policy,model=n,target=target_id,train_U=u,target_U=target_u,
                                            relative_gap=u/target_u-1))
        atomic_json(out/f"calibration_{b}.json",dict(norms=norm,weights=coeff))
    table(out/"energy_matching.csv",energy_rows)
    atomic_json(out/"reuse_audit.json",dict(historical_evidence=old_references,reused=reused,
                                           shared={b:sha((out if b=="S3" else PREVIOUS)/f"shared_{b}.pt") for b in ("B","E","S3")},
                                           layouts=dict(B="NCDHW",E="NCDHW",S3="channels_last_3d")))
    gradient,clipping,batches = monitor(ROOT,records,train,z)
    table(out/"gradient_diagnostics.csv",gradient)
    table(out/"gradient_clipping.csv",clipping)
    atomic_json(out/"gradient_batches.json",[[train.cases[i] for i in indices] for indices in batches])
    native,dev_records = dev_table(records,dev,dz)
    selected_rules = sensitivity(out,native,dev_records,records)
    choices = {}
    for b in ("B","E","S3"):
        mapping = read_json(out/f"core_map_{b}.json")
        choices[b] = {k:min(v,key=lambda n:(records[n]["selected"]["dev_mse"],n)) for k,v in mapping.items()}
        choices[b]["M0"] = f"{b}_M0"
        if b!="S3":
            for policy in ("selected","final"):
                match = min((r for r in energy_rows if r["branch"]==b and r["policy"]==policy),
                            key=lambda r:(abs(r["relative_gap"]),r["model"]))
                choices[b][f"EN_train_U_{policy}"] = match["model"]
    outer = {"B_MSE":choices["B"]["M0"],"B_ACC":"historical_B_ACC_raw_1","B_MAG":"historical_B_MAG_rho_0.1",
             "B_TM2":choices["B"]["TM2"],"E_MSE":choices["E"]["M0"],"E_EN":choices["E"]["EN"],
             "E_TM1":choices["E"]["TM1"],"E_TM2":choices["E"]["TM2"]}
    global_pick = {p:min((results[p]["A"] for results in selected_rules.values()),
                         key=lambda n:(records[n][p]["dev_mse"],n)) for p in ("selected","final")}
    selection = dict(choices=choices,rules=selected_rules,outer=outer,global_rule_A=global_pick,
                     primary=dict(P1=[choices["E"]["TM2"],choices["E"]["M0"]],
                                  P2=[choices["E"]["TM2"],choices["E"]["EN"]],
                                  P2_U_selected=[choices["E"]["TM2"],choices["E"]["EN_train_U_selected"]],
                                  P2_U_final=[choices["E"]["TM2"],choices["E"]["EN_train_U_final"]],
                                  P3=[choices["E"]["TM2"],choices["E"]["TM1"]]))
    new_count = sum(1 for p in (out/"fits").glob("*.json"))
    assert new_count<=32
    atomic_json(out/"selections.json",selection)
    atomic_json(out/"period_plan.json",plan_periods(study))
    atomic_json(out/"math_verification.json",verify_math())
    all_csv = ["mechanism_search.csv","gradient_diagnostics.csv","gradient_clipping.csv","selection_sensitivity.csv",
               "energy_matching.csv"]
    files = ["PROTOCOL.md","hypotheses.json","experiment_plan.json","data_roles.json","sources.json","reuse_audit.json",
             "shared_S3.pt","triplet_manifest.json","outer_split_manifest.json","selections.json","period_plan.json",
             "math_verification.json","native_train.json","native_development.json","gradient_batches.json",
             "development_tracks.json",*all_csv]
    files.extend(f"core_map_{b}.json" for b in ("S3","B","E"))
    weights = {r["path"]:r["sha256"] for v in records.values() for r in v["training"]}
    assert all(sha(ROOT/p)==digest for p,digest in weights.items())
    atomic_json(out/"FROZEN.json",dict(prior_test_manifest_blob=study.registry["prior_test_manifest_blob"],
                                      bindings={p:sha(out/p) for p in files},global_checkpoints=weights,
                                      core_new_fits=new_count,selection=selection,models=records,
                                      validation=study.plan["verification"]))
    print("FROZEN",selection,flush=True)
