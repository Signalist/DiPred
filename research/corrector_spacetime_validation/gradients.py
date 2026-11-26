"""Parameter-gradient probes on fixed, regime-stratified training batches."""
from collections import defaultdict
from statistics import mean

import torch

from research.corrector_spacetime_optimization.diagnostics import calibration,cosine,probe_indices

from .fit import load_net
from .objectives import terms


def monitor_indices(cases):
    strata = defaultdict(list)
    for i,c in enumerate(cases):
        strata[c["re"]].append(i)
    clusters = [strata[k] for k in sorted(strata)]
    return [[g[(7*b+3*j)%len(g)] for j in range(8)
             for g in [clusters[(b*3+j)%len(clusters)]]] for b in range(8)]


def vectors(net,z,y,batches,names):
    params = tuple(net.parameters())
    outputs = []
    for ids in batches:
        value = terms(net(z[ids]),y[ids],names)
        grads = {}
        for i,(name,loss) in enumerate(value.items()):
            layers = torch.autograd.grad(loss,params,retain_graph=i+1<len(value))
            vector = torch.cat([p.flatten() for p in layers])
            assert torch.isfinite(loss) and torch.isfinite(vector).all()
            grads[name] = (float(loss.detach()),vector,[float(torch.cat([v.flatten() for v in layers[j:j+2]]).norm())
                                                       for j in (0,2,4)])
        outputs.append(grads)
    return outputs


def calibrate(net,z,y,cases,names):
    probes = vectors(net,z,y,probe_indices(cases),names)
    norms = {k:mean(float(p[k][1].norm()) for p in probes) for k in probes[0]}
    results = {}
    for name in names:
        results[name] = {rho:calibration(norms[name],norms["mse"],rho)[0] for rho in (.1,.3)}
        assert all(v>0 for v in results[name].values()),name
    return results,norms


def monitor(root,records,train,z,steps=(1000,2000,4000,6000)):
    batches = monitor_indices(train.cases)
    rows,clipping = [],[]
    done = set()
    for model in records.values():
        job = model["config"]
        branch = job["branch"]
        weights = job["weights"]
        label = model["name"]
        if label in done:
            continue
        done.add(label)
        objective_names = tuple(weights)
        for step in steps:
            row = next(r for r in model["training"] if r["step"]==step)
            net = load_net(root/row["path"],row["sha256"],branch,train.p.device)
            estimates = vectors(net,z[branch],(train.y-train.p)/job["scale"],batches,objective_names)
            for b,pack in enumerate(estimates):
                mse = pack["mse"][1]
                total = mse+sum(pack[k][1]*w for k,w in weights.items())
                preclip = float(total.norm())
                coefficient = min(1.,1./(preclip+1e-6))
                clipping.append(dict(model=label,branch=branch,step=step,batch=b,preclip=preclip,
                                     postclip=coefficient*preclip,clip_factor=coefficient,clipped=preclip>1,
                                     total_to_mse_cosine=cosine(total,mse),mse_norm=float(mse.norm())))
                for term,(value,vector,layers) in pack.items():
                    w = weights.get(term,1.)
                    for other,(_,against,_) in pack.items():
                        rows.append(dict(model=label,branch=branch,step=step,batch=b,term=term,other=other,
                                         raw_loss=value,weight=w,gradient_norm=float(vector.norm()),
                                         weighted_norm=float(vector.norm())*w,postclip_norm=float(vector.norm())*w*coefficient,
                                         cosine=cosine(vector,against),conv1=layers[0],conv2=layers[1],conv3=layers[2]))
    return rows,clipping,batches
