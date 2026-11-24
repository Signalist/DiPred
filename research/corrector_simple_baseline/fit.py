"""One fixed-seed training run, with preflight diagnostics and development selection."""
import csv

import torch

from research.corrector.run import dump, sha
from research.corrector_baseline_recovery.diagnostics import update
from research.corrector_baseline_recovery.fit import sampling
from research.corrector_window_study.data import tensor_sha

from .measure import errors, weighted
from .model import fields, network, physical_residual

SEED = 42
STEPS = (0,50,100,250,500,750,1000,1500,2000)


def table(path,rows):
    with path.open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def optimization(net,updates):
    opt = torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-5)
    return opt,torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=updates-1,eta_min=1e-6)


def preflight(split,norm,scale,out):
    x,p = (v[:4] for v in fields(split,norm))
    physical = split.p[:4].double()
    y = split.y[:4].double()
    truth = y-physical
    assert torch.equal(physical+torch.zeros_like(physical),physical)
    torch.testing.assert_close(physical+truth,y,atol=1e-15,rtol=0.)
    probe = truth*.4+.001
    mse_gap = float(((physical+probe-y).square().mean()-(probe-truth).square().mean()).abs())
    assert mse_gap<1e-16
    for normalized,raw in ((x,split.x),(p,split.p)):
        torch.testing.assert_close(normalized*norm["std"]+norm["mean"],raw[:4],atol=2e-7,rtol=0.)
    amplitudes = physical.new_tensor([.008,-.004]).float().view(1,2,1,1,1)
    scales = physical.new_tensor([.08,-.16]).float().view(1,2,1,1,1)
    targets = {"fixed_bias":amplitudes.expand_as(p),"channel_scaling":(split.p[:4]-norm["mean"])*scales,
               "real_windows":(split.y[:4]-split.p[:4])}
    curves, results = [], []
    for mode,label in targets.items():
        torch.manual_seed(SEED)
        net = network(p.device)
        opt,sched = optimization(net,1000)
        score = []
        for step in range(1001):
            if step in STEPS[:7]:
                with torch.no_grad():
                    mse = float((net(x,p)*scale-label).square().mean())
                score.append(mse)
                curves.append(dict(diagnostic=mode,step=step,mse=mse))
            if step<1000:
                loss,grad = update(net,opt,(x,p),label/scale)
                sched.step()
        assert torch.isfinite(torch.stack((loss,grad))).all()
        reduction = 100*(1-score[-1]/score[0])
        row = dict(diagnostic=mode,initial_mse=score[0],final_mse=score[-1],improvement_pct=reduction)
        results.append(row)
        print("DIAGNOSTIC",row,flush=True)
    passed = all(r["improvement_pct"]>(0 if r["diagnostic"]=="real_windows" else 90) for r in results)
    dump(out/"diagnostic_audit.json",dict(passed=passed,zero_exact=True,truth_recovers_Y=True,
                                         normalized_roundtrip=True,residual_vs_forecast_mse_error=mse_gap,
                                         fixed_train_cases=split.cases[:4],results=results))
    table(out/"diagnostic_curves.csv",curves)
    assert passed


def train(train,dev,norm,scale,out):
    field = {k:fields(s,norm) for k,s in (("train",train),("development",dev))}
    plan = sampling(train,SEED,False)
    schedule_sha = tensor_sha(plan)
    plan = plan.cuda()
    torch.manual_seed(SEED)
    net = network(train.x.device)
    assert sum(t.numel() for t in net.parameters())==3042
    opt,sched = optimization(net,2000)
    checkpoint_dir = out/"checkpoints"
    checkpoint_dir.mkdir(exist_ok=False)
    labels = (train.y-train.p)/scale
    rows, updates = [], []
    step_lr = opt.param_groups[0]["lr"]
    for step in range(2001):
        if step in STEPS:
            path = checkpoint_dir/f"simple_{step}.pt"
            torch.save({k:v.cpu() for k,v in net.state_dict().items()},path)
            batch_loss,grad = torch.stack(updates).mean(0).tolist() if updates else (None,None)
            row = dict(step=step,checkpoint=str(path.relative_to(out)),sha256=sha(path),lr=step_lr,
                       scaled_batch_objective=batch_loss,gradient_norm=grad,schedule_sha256=schedule_sha)
            assert not updates or torch.isfinite(torch.tensor([batch_loss,grad])).all()
            for key,split in (("train",train),("development",dev)):
                r = physical_residual(net,field[key],scale)
                if step==0:
                    assert not torch.count_nonzero(r) and torch.equal(split.p+r,split.p)
                scores, _, mismatch = errors(r,split.p,split.y,norm["std"])
                for name in ("mse","native_mse","C","E","G","cosine"):
                    row[f"{key}_{name}"] = float(weighted(scores[name],split.cases))
                row[f"{key}_decomposition_error"] = mismatch
            rows.append(row)
            table(out/"training.csv",rows)
            print("CHECKPOINT",step,"train",row["train_mse"],"dev",row["development_mse"],flush=True)
            updates.clear()
        if step<2000:
            net.train()
            i = plan[step]
            step_lr = opt.param_groups[0]["lr"]
            loss,grad = update(net,opt,tuple(v[i] for v in field["train"]),labels[i])
            updates.append(torch.stack((loss,grad)))
            sched.step()
    choose = lambda r:(r["development_mse"],r["step"])
    return {"native":rows[0],"selected":min(rows,key=choose),
            "best_nonzero":min((r for r in rows if r["development_E"]>0),key=choose),"final":rows[-1]}
