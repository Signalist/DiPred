"""All core searches use B/E-specific shared MSE, with strict budget accounting."""
import json
from statistics import mean

from research.corrector.run import dump,sha
from research.corrector_simple_baseline.fit import table

from .data import BRANCHES
from .diagnostics import calibration,norms,probe_indices,statistics,vectors,weighted_statistics
from .fit import START,checked_net
from .loss import LOSSES,NAMES
from research.corrector_residual_feedback.model import predict

CORE = 60
ORIGINAL = {k:.1 for k in LOSSES[:-1]}


def ranked(runner,names):
    return min(names,key=lambda n:(runner.runs[n]["selected"]["dev_mse"],n))


def development_support(runner,name):
    run = runner.runs[name]
    b = run["config"]["branch"]
    plain = runner.runs[f"{b}_MSE"]["selected"]
    chosen = run["selected"]
    target = runner.dev.y.double()-runner.dev.p.double()
    def correction(row):
        net = checked_net(row,runner.out,runner.dev.p.device)
        return predict(net,runner.dz[b],runner.scale).double()
    better = (target-correction(plain)).square().mean((1,2,3,4))-(target-correction(chosen)).square().mean((1,2,3,4))
    from research.corrector_predictability.alpha import trajectory_indices
    groups = trajectory_indices(runner.dev.cases)
    return dict(gain_pct=100*(1-chosen["dev_mse"]/plain["dev_mse"]),
                improved=int(sum(float(better[ix].mean())>0 for ix in groups.values())))


class Search:
    def __init__(self,runner):
        self.r = runner
        self.out = runner.out
        self.unique = {(b,()):f"{b}_MSE" for b in BRANCHES}
        self.rows = []
        self.joint = []
        self.fixed = {b:{"MSE":f"{b}_MSE"} for b in BRANCHES}
        self.single = {b:{} for b in BRANCHES}
        self.combined = {b:{} for b in BRANCHES}

    def trial(self,branch,tag,weights,stage,loss="",coordinate=None):
        clean = {k:float(v) for k,v in weights.items() if v}
        assert all(0.<v<=100. for v in clean.values())
        key = branch,tuple(sorted(clean.items()))
        found = self.unique.get(key)
        if found is None:
            count = sum(v["config"]["stage"]!="extension" for v in self.r.runs.values())
            ext = len(self.r.runs)-count
            assert (ext<20 if stage=="extension" else count<CORE)
            found = self.r.train_arm(branch,tag,clean,stage)
            self.unique[key] = found
        result = self.r.runs[found]["selected"]
        row = dict(branch=branch,requested=tag,run=found,stage=stage,loss=loss,coordinate=coordinate,
                   weights=json.dumps(clean,sort_keys=True),step=result["step"],dev_mse=result["dev_mse"],
                   gain_native_pct=100*(1-result["dev_mse"]/result["dev_native"]))
        (self.joint if stage=="joint" else self.rows).append(row)
        table(self.out/("joint_search.csv" if stage=="joint" else "loss_search.csv"),
              self.joint if stage=="joint" else self.rows)
        return found

    def fixed_arms(self):
        for b in BRANCHES:
            for tag,name in NAMES.items():
                self.fixed[b][tag] = self.trial(b,tag,{name:.1},"fixed",name,.1)
            self.fixed[b]["JOINT"] = self.trial(b,"JOINT",ORIGINAL,"fixed")
        dump(self.out/"fixed_arms.json",self.fixed)

    def search_single(self,probes):
        raw,balanced = {},{}
        for b in BRANCHES:
            prior = f"{b}_MSE"
            ratio = norms(probes[b])
            for tag,loss in NAMES.items():
                raw[b,loss] = {0.:self.trial(b,f"{tag}_raw_zero",{},"raw",loss,0.),
                               .1:self.trial(b,f"{tag}_raw_point1",{loss:.1},"raw",loss,.1)}
                for v in (.01,1.):
                    raw[b,loss][v] = self.trial(b,f"{tag}_raw_{v:g}",{loss:v},"raw",loss,v)
                balanced[b,loss] = {0.:prior}
                for rho in (.1,.3):
                    weight,_,status = calibration(ratio[loss],ratio["mse"],rho)
                    if status=="ok":
                        balanced[b,loss][rho] = self.trial(b,f"{tag}_rho_{rho:g}",{loss:weight},"balanced",loss,rho)
        # Reserve every potentially distinct joint arm, even when optional boundary points qualify.
        room = max(0,CORE-len(self.r.runs)-6)
        requests = []
        for key,tried in raw.items():
            b,loss = key
            winner = ranked(self.r,tried.values())
            v = next(k for k,name in tried.items() if name==winner)
            if v in (.01,1.) and winner!=f"{b}_MSE":
                a = self.r.runs[winner]["selected"]["dev_mse"]
                base = self.r.runs[f"{b}_MSE"]["selected"]["dev_mse"]
                requests.append((1-a/base,b,loss,v))
        for _,b,loss,v in sorted(requests,reverse=True)[:room]:
            new = .001 if v==.01 else 10.
            raw[b,loss][new] = self.trial(b,f"{loss}_edge_{new:g}",{loss:new},"raw_boundary",loss,new)
        for key,values in raw.items():
            b,loss = key
            best_raw = ranked(self.r,values.values())
            best_balanced = ranked(self.r,balanced[key].values())
            best = ranked(self.r,(best_raw,best_balanced))
            nonzero = ranked(self.r,(n for n in (*values.values(),*balanced[key].values()) if n!=f"{b}_MSE"))
            self.single[b][loss] = dict(raw=best_raw,balanced=best_balanced,best=best,nonzero=nonzero,
                                      weight=self.r.runs[best]["config"]["weights"].get(loss,0.))
        return dict(eligible=len(requests),executed=min(room,len(requests)))

    def joint_arms(self,probes):
        for b in BRANCHES:
            best = self.single[b]
            tuned = {k:v["weight"] for k,v in best.items() if v["weight"]}
            keys = sorted(best,key=lambda k:(self.r.runs[best[k]["nonzero"]]["selected"]["dev_mse"],k))[:2]
            top2 = {k:self.r.runs[best[k]["nonzero"]]["config"]["weights"][k] for k in keys}
            basis = tuned if len(tuned)>=2 else top2
            world = probes[b]
            base = norms(world)["mse"]
            joint = mean(float(sum(batch[k]["gradient"]*v for k,v in basis.items()).norm())
                         for batch in world)
            # Keep individual weights capped and report the achieved joint ratio.
            factor = .3*base/(joint+1e-12) if joint>max(1e-8*base,1e-12) else 0.
            adj = {k:min(100.,v*factor) for k,v in basis.items()}
            achieved = (mean(float(sum(batch[k]["gradient"]*v for k,v in adj.items()).norm())
                             for batch in world)/base) if adj else 0.
            self.combined[b] = dict(original=self.trial(b,"JOINT_original",ORIGINAL,"joint"),
                                    tuned=self.trial(b,"JOINT_tuned",tuned,"joint"),
                                    balanced=self.trial(b,"JOINT_balanced",adj,"joint","joint_gradient_ratio",achieved),
                                    top2=self.trial(b,"JOINT_top2",top2,"joint"))
            self.combined[b]["best"] = ranked(self.r,self.combined[b].values())
        return self.combined


def execute(runner):
    r,out = runner,runner.out
    search = Search(runner)
    search.fixed_arms()
    probe = probe_indices(r.train.cases)
    dump(out/"gradient_batches.json",dict(indices=probe,cases=[[r.train.cases[i] for i in ids] for ids in probe]))
    diagnostics,conflicts,probes = [],[],{}
    for b in BRANCHES:
        plain = r.runs[f"{b}_MSE"]
        for step in (1000,3000,6000):
            row = next(x for x in plain["training"] if x["step"]==step)
            batch = vectors(r,b,row,probe)
            if step==START:
                probes[b] = batch
            d,c = statistics(b,f"{b}_MSE",step,batch)
            diagnostics.extend(d)
            conflicts.extend(c)
        for tag in ("DIR","VEL","ACC","MATCH","JOINT"):
            name = search.fixed[b][tag]
            for step in (2000,4000,6000):
                row = next(x for x in r.runs[name]["training"] if x["step"]==step)
                d,c = statistics(b,name,step,vectors(r,b,row,probe))
                diagnostics.extend(d)
                conflicts.extend(c)
    edge = search.search_single(probes)
    joints = search.joint_arms(probes)
    core_count = len(r.runs)
    assert core_count<=CORE
    core = {b:ranked(r,(f"{b}_MSE",*search.fixed[b].values(),*(x["best"] for x in search.single[b].values()),
                          *joints[b].values())) for b in BRANCHES}

    plan = []
    for b in BRANCHES:
        for loss,control,hypothesis in (
            ("relation","direction","保留真实相邻方向关系可能优于强制近邻同向"),
            ("match2","acceleration","恢复真实二阶差分可能优于抑制预测二阶变化")):
            plan.append(dict(branch=b,loss=loss,control=search.single[b][control]["best"],hypothesis=hypothesis,rho=.3))
        joint_name = joints[b]["best"]
        support = development_support(r,joint_name)
        w = r.runs[joint_name]["config"]["weights"]
        best_single = ranked(r,(x["best"] for x in search.single[b].values()))
        if len(w)>=2 and support["gain_pct"]>.5 and support["improved"]>=8 and ranked(r,(joint_name,best_single))==joint_name:
            plan.extend(dict(branch=b,loss=f"omit_{key}",control=joint_name,
                             hypothesis="固定联合配置中移除此项以验证独立贡献",
                             weights={k:v for k,v in w.items() if k!=key}) for key in w)
    dump(out/"exploration_plan.json",plan)
    extensions_used = []
    for item in plan:
        b,loss = item["branch"],item["loss"]
        if "weights" in item:
            weights = item["weights"]
        else:
            baseline = r.runs[f"{b}_MSE"]
            start = next(x for x in baseline["training"] if x["step"]==START)
            grad = norms(vectors(r,b,start,probe,(loss,)))
            coefficient,clipped,status = calibration(grad[loss],grad["mse"],item["rho"])
            weights = {loss:coefficient}
            if status!="ok":
                extensions_used.append(dict(item,executed=False,status=status))
                continue
            item = dict(item,clipped=clipped,calibrated_weight=coefficient)
        candidate = search.trial(b,f"ext_{loss}",weights,"extension",loss)
        support = development_support(r,candidate)
        extensions_used.append(dict(item,executed=True,run=candidate,**support))
        dump(out/"exploration_results.json",extensions_used)
    # Measure the chosen weighted gradients again under their own training paths.
    weighted = []
    for name,item in r.runs.items():
        cfg = item["config"]
        if cfg["stage"] not in ("fixed","raw","raw_boundary","balanced","joint","extension"):
            continue
        branch,weights = cfg["branch"],cfg["weights"]
        for step in (2000,4000,6000):
            row = next(x for x in item["training"] if x["step"]==step)
            batch = vectors(r,branch,row,probe,tuple(weights))
            weighted.extend(weighted_statistics(branch,name,step,batch,weights))
    table(out/"gradient_diagnostics.csv",diagnostics)
    table(out/"gradient_conflicts.csv",conflicts)
    table(out/"weighted_gradients.csv",weighted)
    dumped_extensions = [e["run"] for e in extensions_used if e["executed"]]
    choice = {b:ranked(r,[core[b],*(n for n in dumped_extensions if r.runs[n]["config"]["branch"]==b)])
              for b in BRANCHES}
    best_extended = ranked(r,dumped_extensions) if dumped_extensions else None
    # The global recommendation may retain the verified S3 when all alternatives are worse.
    external = r.study.prior["reference"]["selected"]["dev_mse"]
    overall = ranked(r,choice.values())
    if external<=r.runs[overall]["selected"]["dev_mse"]:
        overall = "S3"
    compact = {n:{k:v[k] for k in ("config","config_hash","selected","final","elapsed_seconds")} for n,v in r.runs.items()}
    selected = dict(fixed=search.fixed,single=search.single,joint=joints,core=core,
                    best=choice,global_best=overall,best_extension=best_extended,models=compact)
    dump(out/"selections.json",selected)
    dump(out/"exploration_results.json",extensions_used)
    omitted = {
        "extra_seeds_and_multinoise":"outside fixed study design",
        "direction_alignment_and_magnitude_relation":"prioritize adjacent directional relation and both temporal orders",
        "boundary_formula":"algebraically equal to first-lead MSE",
        "alternative_history_layout_or_self_feedback":"changes the input regime of the core comparison",
        "auxiliary_schedules_and_S3_refit":"would require additional matched fits",
        "edge_Re_and_internal_holdout":"current fixed cohort and historical reference used",
        "rho_0.01_and_1.0":"prespecified 0.1 and 0.3 are the core search",
    }
    dump(out/"stage_decisions.json",dict(core_fits=core_count,extension_fits=len(r.runs)-core_count,
                                         boundary=edge,not_executed=omitted,extra_data_used=False,
                                         expert_feedback_rollout=False,internal_holdout_reused=False,
                                         batch_probes=len(probe)))
    files = ["PROTOCOL.md","triplet_manifest.json","config.json","sources.json","selections.json","training.csv",
             "baseline_replay.json","gradient_batches.json","gradient_diagnostics.csv","gradient_conflicts.csv",
             "joint_search.csv","loss_search.csv","weighted_gradients.csv","stage_decisions.json","exploration_plan.json","exploration_results.json",
             "native_train.json","native_development.json","shared_B.pt","shared_E.pt"]
    files.extend(f"run_{name}.json" for name in r.runs)
    frozen = dict(prior_test_manifest_blob=r.study.prior["prior_test_manifest_blob"],
                  bindings={k:sha(out/k) for k in files},selections=selected,seed=42,
                  checkpoints={str(p.relative_to(out)):sha(p) for p in sorted((out/"checkpoints").glob("*.pt"))},
                  rule="minimum equal-trajectory development MSE, nonzero, ties earliest step")
    dump(out/"FROZEN.json",frozen)
    print("FROZEN BEFORE TEST",choice,overall,flush=True)
