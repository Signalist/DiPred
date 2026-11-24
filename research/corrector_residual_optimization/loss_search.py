"""Conditional fixed, calibrated, and joint searches on the existing objectives."""
from statistics import mean

import torch

from research.corrector_simple_baseline.fit import table

from .fit import load
from .model import LOSSES, components

ORIGINAL = {k:.1 for k in LOSSES[:-1]}
EPS = 1e-12
RATIO_FLOOR = 1e-8


def calibration(raw,base,rho):
    if raw<=RATIO_FLOOR*base:
        return 0.,False,"uncalibratable"
    proposed = rho*base/(raw+EPS)
    return min(proposed,100.),proposed>100,"finite"


def gradient_diagnostics(exp,base):
    run = exp.runs[base]
    cfg = run["config"]
    net = load(cfg,run["selected"],exp.out,exp.train.p.device).eval()
    x = exp.fields(exp.train,cfg["input"])
    target = (exp.train.y-exp.train.p)/exp.scale
    params = tuple(net.parameters())
    vectors, values = [], []
    rows = []
    for batch,ix in enumerate(exp.schedule[:3].cuda(),1):
        loss = components(net(x[ix]),target[ix],LOSSES)
        p = {name:torch.autograd.grad(value,params,retain_graph=name!="match")
             for name,value in loss.items()}
        assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for group in p.values() for v in group)
        vectors.append({k:torch.cat([v.flatten() for v in seq]) for k,seq in p.items()})
        values.append({k:float(t.detach()) for k,t in loss.items()})
        for name in p:
            raw = vectors[-1][name]
            primary = vectors[-1]["mse"]
            for layer,j in enumerate((0,2,4),1):
                raw_layer = torch.cat([v.flatten() for v in p[name][j:j+2]])
                values[-1][f"{name}_layer{layer}"] = float(raw_layer.norm())
            values[-1][f"{name}_cosine"] = float((raw@primary)/(raw.norm()*primary.norm()).clamp_min(EPS))
    norms = {k:mean(float(v[k].norm()) for v in vectors) for k in ("mse",*LOSSES)}
    assert norms["mse"]>0
    for batch,(grad,value) in enumerate(zip(vectors,values),1):
        for name in grad:
            raw = float(grad[name].norm())
            rates = {tag:calibration(norms[name],norms["mse"],rho)
                     for tag,rho in (("0.1",.1),("0.3",.3))}
            rows.append(dict(batch=batch,loss=name,raw_loss=value[name],gradient_l2=raw,
                             ratio_to_mse=raw/float(grad["mse"].norm()),cosine_to_mse=value[f"{name}_cosine"],
                             layer_1=value[f"{name}_layer1"],layer_2=value[f"{name}_layer2"],
                             layer_3=value[f"{name}_layer3"],fixed_0_1_gradient=.1*raw,
                             calibrated_rho_0_1_lambda=rates["0.1"][0],calibrated_rho_0_1_gradient=rates["0.1"][0]*raw,
                             calibrated_rho_0_3_lambda=rates["0.3"][0],calibrated_rho_0_3_gradient=rates["0.3"][0]*raw,
                             clipped_rho_0_1=rates["0.1"][1],clipped_rho_0_3=rates["0.3"][1],
                             calibration_status=rates["0.1"][2]))
    table(exp.out/"gradient_diagnostics.csv",rows)
    return norms,vectors


def search(exp,base,diagnostics):
    norms,vectors = diagnostics
    config = exp.runs[base]["config"]
    seen = {():base}
    search_rows = []
    def fit(name,weights,stage,coordinate=None):
        weights = {k:v for k,v in weights.items() if v}
        key = tuple(sorted(weights.items()))
        if key not in seen:
            seen[key] = exp.fit(name,dict(config,aux=weights),"loss_"+stage)
        run = seen[key]
        row = exp.runs[run]["selected"]
        search_rows.append(dict(candidate=name,run=run,stage=stage,coordinate=coordinate,weights=str(weights),
                                step=row["step"],dev_mse=row["dev_mse"],
                                dev_gain_pct=100*row["dev_G"]/row["dev_native"]))
        return run

    fixed = {"A":base}
    for letter, name in zip("BCDEF",LOSSES):
        fixed[letter] = fit(f"fixed_{letter}",{name:.1},"fixed",.1)
    fixed["G"] = fit("G_original",ORIGINAL,"joint")
    single, balance, chosen = {},{},{}
    for letter,name in zip("BCDEF",LOSSES):
        runs = {0.:base,.1:fixed[letter]}
        for rate in (.01,1.):
            runs[rate] = fit(f"{name}_{rate:g}",{name:rate},"raw",rate)
        picked = exp.best(runs.values())
        best_rate = next(k for k,v in runs.items() if v==picked)
        if best_rate in (.01,1.):
            rate = .001 if best_rate==.01 else 10.
            runs[rate] = fit(f"{name}_{rate:g}",{name:rate},"raw",rate)
        single[name] = exp.best(runs.values())
        if calibration(norms[name],norms["mse"],.1)[2]=="uncalibratable":
            balance[name] = base
        else:
            matched = {0.:base}
            for ratio in (.1,.3):
                rate,_,_ = calibration(norms[name],norms["mse"],ratio)
                matched[ratio] = fit(f"{name}_rho_{ratio:g}",{name:rate},"balanced",ratio)
            favorite = exp.best(matched.values())
            ratio = next(k for k,v in matched.items() if v==favorite)
            if ratio in (.1,.3):
                boundary = .01 if ratio==.1 else 1.
                rate,_,_ = calibration(norms[name],norms["mse"],boundary)
                matched[boundary] = fit(f"{name}_rho_{boundary:g}",{name:rate},"balanced",boundary)
            balance[name] = exp.best(matched.values())
        chosen[name] = exp.best((single[name],balance[name]))
    tuned = {k:exp.runs[run]["config"]["aux"].get(k,0.)
             for k,run in chosen.items() if run!=base}
    joint = {"G_original":fixed["G"]}
    joint["G_tuned"] = fit("G_tuned",tuned,"joint")
    if tuned:
        combined = mean(float(sum(a*v[k] for k,a in tuned.items()).norm()) for v in vectors)
        gain,_,_ = calibration(combined,norms["mse"],.3)
        adjusted = {k:min(gain*v,100.) for k,v in tuned.items()}
        achieved = mean(float(sum(a*v[k] for k,a in adjusted.items()).norm()) for v in vectors)/norms["mse"]
        joint["G_balanced"] = fit("G_balanced",adjusted,"joint",achieved)
    else:
        joint["G_balanced"] = base
    leading = sorted(tuned,key=lambda k:exp.runs[chosen[k]]["selected"]["dev_mse"])[:2]
    joint["G_top2"] = fit("G_top2",{k:tuned[k] for k in leading},"joint")
    table(exp.out/"loss_search.csv",search_rows)
    assert sum(v["stage"].startswith("loss_") for v in exp.runs.values())<=39
    return dict(fixed=fixed,raw=single,balanced=balance,tuned=chosen,joint=joint,
                best_joint=exp.best([base,*joint.values()]),best=exp.best([base,*chosen.values(),*joint.values()]))
