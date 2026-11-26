"""Run the frozen evaluation cohorts and collect the signed physical evidence."""
import csv
from statistics import mean

import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.data import guard
from research.corrector_residual_feedback.evaluate import time_model
from research.corrector_residual_feedback.model import base,predict
from research.corrector_residual_optimization.fit import load as load_s3
from research.corrector_simple_baseline.fit import table
from research.corrector_window_study.data import read_json,tensor_sha

from .data import HISTORICAL,PREVIOUS,ROOT,S3_ROOT
from .expansion import feedback_inputs,period_data,rollout
from .fit import atomic_json,load_net
from .outer import outer_models
from .score import Scores


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def comparisons(frozen):
    def choose(split,models):
        pairs = []
        if split in ("development","historical_test","period","period_new"):
            for label,(model,control) in frozen["selection"]["primary"].items():
                if {model,control}<=set(models):
                    pairs.append((label,model,control))
            for b in ("B","E","S3"):
                for arm in frozen["selection"]["choices"][b].values():
                    plain = f"{b}_M0"
                    if arm!=plain and {arm,plain}<=set(models):
                        pairs.append(("S1",arm,plain))
            for a,b in (("E_DIAG","historical_E_ext_match2"),("E_PERM","historical_E_ext_match2"),
                        ("B_J4","B_J5"),("E_J4","E_J5"),("E_J4","E_JOINT_original"),
                        ("B_J4","B_JOINT_original")):
                if {a,b}<=set(models):
                    pairs.append(("S2" if a.startswith(("E_DIAG","E_PERM")) else "S3",b,a))
            for branch,policies in frozen["selection"]["rules"].items():
                for policy,rules in policies.items():
                    for rule,name in rules.items():
                        plain = f"{branch}_M0"
                        if {name,plain}<=set(models):
                            pairs.append((f"S4_{policy}_{rule}",name,plain))
            for name in HISTORICAL:
                model = f"historical_{name}"
                if model in models:
                    pairs.append(("historical_vs_own",model,f"{name[0]}_M0"))
        if split.startswith("holdout_"):
            pairs.extend(("S5",m,m[0]+"_MSE") for m in models if m.startswith(("B_","E_")) and not m.endswith("_MSE"))
        if split in ("period","period_new"):
            for e in ("M0","EN","TM1","TM2"):
                a = f"self_{e}"
                b = frozen["selection"]["choices"]["E"][e]
                pairs.append(("S6",a,b))
        return list(dict.fromkeys(pairs))

    return choose


def table_fit(out,frozen):
    rows = {}
    for arm,r in frozen["models"].items():
        rows[arm] = [dict(row,run=arm,source=r["name"],origin=r["kind"]) for row in r["training"]]
    for file in sorted((out/"fits").glob("outer_*.json")):
        item = read_json(file)
        rows[file.stem] = [dict(r,path=str((out/r["checkpoint"]).relative_to(ROOT)),source=file.stem,
                                origin="outer") for r in item["training"]]
    full = [r for group in rows.values() for r in group]
    keys = list(dict.fromkeys(k for r in full for k in r))
    table(out/"training.csv",[{key:r.get(key) for key in keys} for r in full])


def history_review(out,root):
    old = read_rows(PREVIOUS/"scores.csv")
    tracks = read_rows(PREVIOUS/"trajectories.csv")
    history = {r["arm"]:{} for r in old}
    for r in old:
        m,C,U,G,native = (float(r[k]) for k in ("mse","C","correction_energy","G","native_mse"))
        assert abs(G-(2*C-U))<1e-15 and abs(G-(native-m))<1e-15
        history[r["arm"]][r["split"],r["policy"]] = m,C,U,G
    details = {"B":{}, "E":{}}
    for row in tracks:
        a = row["arm"]
        if a in ("B_MSE","E_MSE"):
            details[a[0]][row["split"],row["policy"],row["trajectory"]] = float(row["mse"])
    reductions = {}
    for row in tracks:
        b = row["arm"][0]
        if b not in details:
            continue
        key = row["split"],row["policy"],row["trajectory"]
        reductions.setdefault((row["arm"],*key[:2]),[]).append((key[-1],float(row["re"]),
                                                               details[b][key]-float(row["mse"])))
    result = []
    for r in old:
        a,s,p = r["arm"],r["split"],r["policy"]
        m,C,U,G = history[a][s,p]
        if a in ("Native","S3"):
            selected_step = 6000 if a=="S3" else 0
        else:
            selected_step = root["selections"]["models"][a][p]["step"]
        d = reductions.get((a,s,p),[])
        positives = sorted((v for _,_,v in d if v>0),reverse=True)
        concentration = sum(positives[:2])/sum(positives) if positives else None
        result.append(dict(split=s,policy=p,arm=a,step=selected_step,shared_prefix=0<selected_step<=1000,
                           mse=m,C=C,U=U,G=G,positive_tracks=len(positives),top2_share_positive_gain=concentration))
    table(out/"historical_review.csv",result)


def fixed_fields(out,fields):
    table(out/"fixed_fields.csv",[
        dict(channel=("u","v")[c],y=y,x=x,**{k:float(v[c,y,x]) for k,v in fields.items()})
        for c in range(2) for y in range(32) for x in range(64)])


@torch.inference_mode()
def score_validation(study,train,outer,frozen):
    out = study.out
    target = (train.y.double()-train.p.double()).abs()[:,:,::2,::2,::2]
    thresholds = [float(torch.quantile(target[:,c].contiguous().flatten(),.95)) for c in (0,1)]
    atomic_json(out/"regional_thresholds.json",dict(source="training target absolute 95% within channel",
                                                   stride=[2,2,2],thresholds=thresholds))
    book = Scores(thresholds)
    models = frozen["models"]
    controls = {a:f'{r["config"]["branch"]}_M0' for a,r in models.items()}
    controls.update(S3="S3_M0")
    controls.update({f"self_{k}":"E_M0" for k in ("M0","EN","TM1","TM2")})
    tests,future,fields,speed = [],[],{},[]
    for label,cohort,estimates in outer:
        for name,est in estimates.items():
            split = f"holdout_{label}"
            assert cohort.cases
            if name.startswith(("B_","E_")):
                controls[split,name] = name[0]+"_MSE"
            book.add(cohort,split,name,"final",est,study.norm["std"],1.)
    reference = load_s3(study.reference["config"],study.reference["selected"],S3_ROOT,train.p.device)
    old = {(r["split"],r["policy"],r["arm"]):float(r["mse"]) for r in read_rows(PREVIOUS/"scores.csv")}
    for split,label in (("development","development"),("test","historical_test")):
        data,features,z = study.get(split)
        z["S3"] = features
        signature = tensor_sha(data.p)
        common = {"Native":torch.zeros_like(data.p),"S3":predict(reference,features,study.s3_scale)}
        if split=="development":
            case = study.plan["fixed_case"]
            idx, = [i for i,c in enumerate(data.cases) if (c["file"],c["start"])==(case["file"],case["start"])]
            lead = case["lead"]-1
            fields["true"] = (data.y[idx,:,lead].double()-data.p[idx,:,lead].double()).cpu()
            speed.append(dict(model="S3",params=sum(p.numel() for p in reference.parameters()),
                              ms_per_window=time_model(reference,features[:8]),training_seconds=study.reference["elapsed_seconds"]))
        wanted = set(frozen["selection"]["choices"]["B"].values())|set(frozen["selection"]["choices"]["E"].values())
        for arm in (*common,*models):
            for policy in ("selected","final"):
                if arm in common:
                    est = common[arm]
                    scale = study.scale
                else:
                    info = models[arm]
                    cfg,row = info["config"],info[policy]
                    b,scale = cfg["branch"],cfg["scale"]
                    net = load_net(ROOT/row["path"],row["sha256"],b,data.p.device)
                    est = predict(net,z[b],scale)
                    if policy=="selected":
                        if split=="development":
                            speed.append(dict(model=arm,params=sum(p.numel() for p in net.parameters()),
                                              ms_per_window=time_model(net,z[b][:8]),training_seconds=info["elapsed_seconds"]))
                        if split=="test":
                            y = data.y
                            data.y = y.flip(0)
                            try:
                                f,zd = study.inputs(data)
                                a = f if b=="S3" else zd[b]
                                assert torch.equal(est,predict(net,a,scale))
                            finally:
                                data.y = y
                            future.append(dict(split=label,model=arm,bitwise=True))
                idx0 = len(book.recomputed)
                book.add(data,label,arm,policy,est,study.norm["std"],scale)
                if arm in common or arm.startswith("historical_"):
                    name = arm.replace("historical_","",1)
                    gap = abs(book.recomputed[idx0]["mse"]-old[split,policy,name])
                    assert gap<1e-12
                    tests.append(dict(split=label,policy=policy,model=name,absolute_error=gap))
                if split=="development" and policy=="selected" and (arm in wanted or arm=="S3"):
                    fields[arm] = est[idx,:,lead].cpu()
        assert tensor_sha(data.p)==signature

    feedback = []
    extended,evidence = period_data(study,data)
    history = base(extended.previous_x,extended.previous_p,study.norm)
    fixed_history = predict(reference,history,study.s3_scale)
    frame0 = base(extended.x,extended.p,study.norm)
    true = extended.x.double()-extended.previous_p.double()
    period_arms = {"Native":torch.zeros_like(extended.p),"S3":predict(reference,frame0,study.s3_scale)}
    active = {arm for values in frozen["selection"]["choices"].values() for arm in values.values()}
    active.update(a for branch in frozen["selection"]["rules"].values() for v in branch.values() for a in v.values())
    new_inds = [i for i,c in enumerate(extended.cases) if c["beyond_prior_window"]]
    from .outer import subset
    more = subset(extended,new_inds)
    for arm in sorted(active):
        item = models[arm]
        cfg,row = item["config"],item["final"]
        b,scale = cfg["branch"],cfg["scale"]
        z = frame0 if b=="S3" else feedback_inputs(frame0,true,fixed_history,b,scale)
        net = load_net(ROOT/row["path"],row["sha256"],b,extended.p.device)
        estimate = period_arms[arm] = predict(net,z,scale)
        for kind in ("M0","EN","TM1","TM2"):
            if frozen["selection"]["choices"]["E"][kind]!=arm:
                continue
            self_est,states = rollout(extended,study.norm,net,fixed_history,scale)
            period_arms[f"self_{kind}"] = self_est
            original = extended.y
            extended.y = original.flip(0)
            try:
                patched,_ = rollout(extended,study.norm,net,fixed_history,scale)
                assert torch.equal(patched,self_est)
            finally:
                extended.y = original
            assert all(s["issue_frame"]<s["score_frame"] for s in states)
            native,regular,responsive = ((extended.p.double()+r.double()-original.double()).square().mean((1,2,3,4)).tolist()
                                         for r in (torch.zeros_like(self_est),estimate,self_est))
            for i,state in enumerate(states):
                start = state["start"]
                if state["prior_source"]=="fixed_reset":
                    assert torch.equal(self_est[i],estimate[i]) or torch.allclose(self_est[i],estimate[i],atol=1e-7,rtol=1e-6)
                feedback.append(dict(model=arm,mode=kind,chain_index=1+(start-400)//20,re=extended.cases[i]["re"],
                                     **state,native_mse=native[i],fixed_mse=regular[i],self_mse=responsive[i],
                                     truth_residual_sha256=tensor_sha(original[i].double()-extended.p[i].double()),
                                     perturbation_unchanged=True))
    for arm,est in period_arms.items():
        book.add(extended,"period",arm,"final",est,study.norm["std"],study.scale)
        if new_inds:
            book.add(more,"period_new",arm,"final",est[new_inds],study.norm["std"],study.scale)
    atomic_json(out/"period_provenance.json",evidence)
    table(out/"feedback_rollout.csv",feedback)
    table(out/"inference_cost.csv",speed)
    fixed_fields(out,fields)
    history_review(out,study.registry)
    result = book.write(out,controls,comparisons(frozen))
    interaction(out,read_rows(out/"trajectories.csv"),frozen)
    return dict(result,freeze_sha256=sha(out/"FROZEN.json"),historical_reproduced=tests,
                future_invariance=future,period_windows=len(extended.cases),new_period_windows=len(new_inds),
                feedback_windows=len(feedback),period_native_sha256=tensor_sha(extended.p))


def interaction(out,rows,frozen):
    # The old positive MAG single arm is an immutable, matched design control.
    # Its scores were bound before validation, without adding a new selected candidate.
    prior = guard(PREVIOUS)
    evidence = read_json(out/"reuse_audit.json")["historical_evidence"]
    historical_tracks = PREVIOUS/"trajectories.csv"
    for path in (PREVIOUS/"FROZEN.json",historical_tracks):
        assert sha(path)==evidence[str(path.relative_to(ROOT))]
    models = frozen["models"]
    mag_name = "E_MAG_rho_0.1"
    old_names = ("E_MSE",mag_name,"E_MATCH_raw_1","E_JOINT_top2")
    new_names = dict(zip(("E_M0","historical_E_MATCH_raw_1","historical_E_JOINT_top2"),
                         (old_names[0],old_names[2],old_names[3])))
    plain,mag,match,joint = (prior["selections"]["models"][n]["config"] for n in old_names)
    assert not plain["weights"] and {**mag["weights"],**match["weights"]}==joint["weights"]
    assert mag["shared_sha256"]==match["shared_sha256"]==joint["shared_sha256"]
    for current,old in new_names.items():
        assert models[current]["config"]==prior["selections"]["models"][old]["config"]
        assert all(models[current][p]["sha256"]==prior["selections"]["models"][old][p]["sha256"]
                   for p in ("selected","final"))
    identity = ("split","policy","trajectory","re")
    old_tracks = {}
    for r in read_rows(historical_tracks):
        if r["arm"] in old_names:
            key = tuple(r[k] for k in identity)
            old_tracks.setdefault(key,{})[r["arm"]] = float(r["mse"])
    tracks = {}
    for r in rows:
        arm,split = r["arm"],r["split"]
        if arm not in new_names or split not in ("development","historical_test"):
            continue
        key = tuple(r[k] for k in identity)
        previous = ("test" if split=="historical_test" else split,*key[1:])
        old = old_tracks[previous]
        value = float(r["mse"])
        assert abs(old[new_names[arm]]-value)<1e-15
        track = tracks.setdefault(key,{mag_name:old[mag_name]})
        assert track[mag_name]==old[mag_name]
        track[new_names[arm]] = value
    values = []
    assert len(tracks)==48
    for key,v in sorted(tracks.items()):
        a,b,c,d = (v[n] for n in old_names)
        values.append(dict(zip(identity,key),plain=a,mag=b,match=c,joint=d,mag_source=mag_name,
                           interaction_mse_improvement=b+c-d-a))
    table(out/"joint_interaction.csv",values)


def validate(study,figures):
    out = study.out
    frozen = guard(out)
    assert all(sha(ROOT/path)==digest for path,digest in frozen["global_checkpoints"].items())
    jobs,lineage = outer_models(study)
    train,current,design = study.get("train")
    del current,design
    results = score_validation(study,train,jobs,frozen)
    table_fit(out,frozen)
    atomic_json(out/"verification.json",dict(results,passed=True))
    from .plot import plot
    from .report import report
    from .audit import audit
    plot(out,figures)
    report(out,figures)
    audit(study,figures)
