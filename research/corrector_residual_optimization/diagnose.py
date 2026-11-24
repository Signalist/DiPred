"""Training-only residual statistics, analytic controls, and input interventions."""
from types import SimpleNamespace

import torch

from research.corrector.run import dump, sha
from research.corrector_simple_baseline.fit import table
from research.corrector_window_study.data import read_json

from .data import SIMPLE, subset
from .fit import load, track_weights
from .model import inputs


def predictors(data):
    last = data.x[:,:,-1:].expand_as(data.p)
    trend = (data.x[:,:,-1:]-data.x[:,:,-2:-1]).expand_as(data.p)
    return torch.cat((data.p,last,trend),dim=1)


@torch.inference_mode()
def analytic(train,out):
    w = track_weights(train)
    true = train.y.double()-train.p.double()
    avg = (true*w.view(-1,1,1,1,1)).sum(0)
    variance = ((true-avg).square()*w.view(-1,1,1,1,1)).sum(0)
    spatial_mean = avg.mean(1)
    spatial_var = (variance+avg.square()).mean(1)-spatial_mean.square()
    table(out/"spatial_residual_stats.csv",[
        dict(channel=label,y=y,x=x,mean=float(spatial_mean[c,y,x]),variance=float(spatial_var[c,y,x]))
        for c,label in enumerate(("u","v")) for y in range(32) for x in range(64)])
    table(out/"lead_residual_stats.csv",[
        dict(channel=label,lead=t+1,mean=float(avg[c,t].mean()),mean_square=float((variance+avg.square())[c,t].mean()))
        for c,label in enumerate(("u","v")) for t in range(20)])
    valid_re = sorted({c["re"] for c in train.cases})
    records = []
    for re in valid_re:
        ix = [i for i,r in enumerate(train.cases) if r["re"]==re]
        local = subset(train,ix)
        weights = track_weights(local)
        r = true[ix]
        for c,label in enumerate(("u","v")):
            for t in range(20):
                x = r[:,c,t]
                records.append(dict(re=re,channel=label,lead=t+1,
                                    mean=float((weights*x.mean((1,2))).sum()),
                                    mean_square=float((weights*x.square().mean((1,2))).sum())))
    table(out/"regime_residual_stats.csv",records)
    def correlation(a,b):
        def expect(t):
            return (w*t.reshape(len(w),-1).mean(1)).sum()
        ma,mb = expect(a),expect(b)
        return float((expect(a*b)-ma*mb)/((expect(a*a)-ma*ma)*(expect(b*b)-mb*mb)).sqrt())
    features = predictors(train)
    train_stats = {}
    for c,label in enumerate(("u","v")):
        r = true[:,c]
        train_stats[label] = dict(mean=float(avg[c].mean()),rms=float((variance+avg.square())[c].mean().sqrt()),
                                  x_lag1=correlation(r[...,:-1],r[...,1:]),
                                  y_lag1=correlation(r[...,:-1,:],r[...,1:,:]),
                                  t_lag1=correlation(r[:,:-1],r[:,1:]),
                                  P=correlation(r,features[:,c]),last=correlation(r,features[:,c+2]),
                                  trend=correlation(r,features[:,c+4]))
    gram = torch.zeros((20,7,7),dtype=torch.float64,device=true.device)
    rhs = torch.zeros((20,7,2),dtype=torch.float64,device=true.device)
    for i in range(0,len(true),16):
        f = features[i:i+16].double()
        f = torch.cat((torch.ones_like(f[:,:1]),f),dim=1)
        weighted = w[i:i+16].sqrt().view(1,-1,1,1,1)
        fx = (f.permute(2,0,3,4,1)*weighted).reshape(20,-1,7)
        dy = (true[i:i+16].permute(2,0,3,4,1)*weighted).reshape(20,-1,2)
        gram += fx.transpose(1,2)@fx
        rhs += fx.transpose(1,2)@dy
    coef = torch.linalg.solve(gram,rhs)
    assert torch.isfinite(coef).all()
    dump(out/"linear_controls.json",dict(fit_split="train",bias=avg.mean((2,3)).tolist(),
                                       weights=coef.tolist(),gram_condition=torch.linalg.cond(gram).tolist()))
    dump(out/"training_residual_stats.json",train_stats)
    return train_stats


@torch.inference_mode()
def analytic_prediction(data,controls,policy):
    if policy=="B1":
        return data.p.new_tensor(controls["bias"]).view(1,2,20,1,1).expand_as(data.p)
    features = predictors(data)
    coeff = features.new_tensor(controls["weights"])
    return torch.einsum("bctij,tco->botij",features,coeff[:,1:])+coeff[:,0].T.reshape(1,2,20,1,1)


def residual_invariants(split,out):
    proof = read_json(SIMPLE/"diagnostic_audit.json")
    assert proof["passed"]
    p,y = split.p[:4].double(),split.y[:4].double()
    truth = y-p
    assert torch.equal(p,p+torch.zeros_like(p))
    assert (p+truth-y).abs().max()<1e-15
    delta = truth*.27+.0003
    assert abs(float((p+delta-y).square().mean()-(delta-truth).square().mean()))<1e-16
    dump(out/"diagnostic_invariants.json",dict(passed=True,zero_native=True,truth_recovers_y=True,
                                               mse_equivalence=True,prior_sha256=sha(SIMPLE/"diagnostic_audit.json"),
                                               synthetic=proof["results"]))


@torch.inference_mode()
def dependency(train,norm,scale,out,names,runs):
    fixed = subset(train,list(range(4)))
    rows = []
    for name in names:
        run = runs[name]
        conf = run["config"]
        net = load(conf,run["selected"],out,fixed.p.device).eval()
        base = net(inputs(fixed,norm,conf["input"]))*scale
        responses = []
        for frame in range(20):
            x = fixed.x.clone()
            x[:,:,frame] += .1*norm["std"][:,:,0]
            modified = SimpleNamespace(x=x,p=fixed.p)
            d = (net(inputs(modified,norm,conf["input"]))*scale-base).double()
            rms = d.square().mean((0,1,3,4)).sqrt().tolist()
            responses.append(rms)
            rows.extend(dict(run=name,history_frame=frame+1,lead=lead+1,response_rms=v)
                        for lead,v in enumerate(rms))
        if name=="S0":
            assert all(responses[h][t]==0 for h in range(20) for t in range(20) if abs(h-t)>3)
    table(out/"input_dependency.csv",rows)
