"""FP32 gradients on a frozen regime-stratified probe schedule."""
from collections import defaultdict
from statistics import mean

import torch

from .fit import checked_net
from .loss import LOSSES,extensions

EPS = 1e-12


def probe_indices(cases):
    regimes = defaultdict(list)
    for i,case in enumerate(cases):
        regimes[case["re"]].append(i)
    ids = sorted(regimes)
    return [[regimes[ids[(j+3*b)%len(ids)]][(7*j+11*b)%len(regimes[ids[(j+3*b)%len(ids)]])]
             for j in range(8)] for b in range(3)]


def vectors(runner,branch,row,indices,losses=LOSSES):
    net = checked_net(row,runner.out,runner.train.p.device).eval()
    params = tuple(net.parameters())
    batches = []
    for ids in indices:
        prediction = net(runner.z[branch][ids])
        values = extensions(prediction,runner.labels[ids],losses)
        grads = {}
        for i,(name,value) in enumerate(values.items()):
            grad = torch.autograd.grad(value,params,retain_graph=i+1<len(values))
            g = torch.cat([v.flatten() for v in grad])
            assert g.dtype==torch.float32 and torch.isfinite(g).all() and torch.isfinite(value)
            raw = float(g.norm())
            grads[name] = dict(loss=float(value.detach()),gradient=g,norm=raw,
                               near_zero_fraction=float((g.abs()<=EPS).float().mean()),
                               layers=[float(torch.cat([v.flatten() for v in grad[j:j+2]]).norm())
                                       for j in (0,2,4)])
        batches.append(grads)
    return batches


def calibration(norm,base,rho):
    if norm<=max(base*1e-8,EPS):
        return 0.,False,"uncalibratable"
    raw = rho*base/(norm+EPS)
    return min(raw,100.),raw>100.,"ok"


def norms(batches):
    return {name:mean(b[name]["norm"] for b in batches) for name in batches[0]}


def cosine(a,b):
    return float((a@b)/(a.norm()*b.norm()).clamp_min(EPS))


def statistics(branch,run,step,batches):
    computed = []
    conflicts = []
    averaged = norms(batches)
    base = averaged["mse"]
    for i,batch in enumerate(batches,1):
        mse = batch["mse"]["gradient"]
        for loss,value in batch.items():
            g = value["gradient"]
            raw = value["norm"]
            fixed_total = mse+.1*g
            total = float(fixed_total.norm())
            ratio = min(1.,1./(total+1e-6))
            low,c1,status = calibration(averaged[loss],base,.1)
            high,c3,_ = calibration(averaged[loss],base,.3)
            computed.append(dict(branch=branch,run=run,step=step,batch=i,loss=loss,
                                 raw_loss=value["loss"],gradient_l2=raw,near_zero_fraction=value["near_zero_fraction"],
                                 layer1=value["layers"][0],layer2=value["layers"][1],layer3=value["layers"][2],
                                 ratio_to_mse=raw/max(batch["mse"]["norm"],EPS),cosine_to_mse=cosine(g,mse),
                                 fixed_0_1_gradient=.1*raw,fixed_total_preclip=total,
                                 fixed_total_postclip=total*ratio,fixed_aux_postclip=.1*raw*ratio,
                                 rho_0_1_weight=low,rho_0_3_weight=high,rho_0_1_norm=low*raw,
                                 rho_0_3_norm=high*raw,clipped_0_1=c1,clipped_0_3=c3,status=status))
            for other,v in batch.items():
                conflicts.append(dict(branch=branch,run=run,step=step,batch=i,loss=loss,other=other,
                                      cosine=cosine(g,v["gradient"])))
    return computed,conflicts


def weighted_statistics(branch,run,step,batches,weights):
    rows = []
    for i,batch in enumerate(batches,1):
        mse = batch["mse"]["gradient"]
        joint = sum(batch[k]["gradient"]*weight for k,weight in weights.items())
        pre = float((mse+joint).norm())
        factor = min(1.,1./(pre+1e-6))
        for loss,value in batch.items():
            weight = weights.get(loss,1.)
            norm = value["norm"]*weight
            rows.append(dict(branch=branch,run=run,step=step,batch=i,loss=loss,weight=weight,
                             weighted_l2=norm,weighted_l2_after_clip=norm*factor,
                             weighted_ratio=norm/max(float(mse.norm()),EPS),
                             auxiliary_ratio=float(joint.norm())/max(float(mse.norm()),EPS),
                             auxiliary_cosine=cosine(joint,mse),preclip=pre,postclip=pre*factor))
    return rows
