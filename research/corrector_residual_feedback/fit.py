"""Restartable fixed-budget residual-MSE training with a common batch schedule."""
import csv
import random
from time import perf_counter

import numpy as np
import torch

from research.corrector.run import dump,sha
from research.corrector_residual_optimization.fit import checks, track_weights
from research.corrector_simple_baseline.fit import table
from research.corrector_window_study.data import read_json,tensor_sha

from .data import write_tensor
from .model import predict, seeded

STEPS = (0,50,100,250,500,750,1000,1500,2000,3000,4000,5000,6000)


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def state_digest(weights):
    return {name:tensor_sha(v) for name,v in weights.items()}


def seed():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)


def fit(name,train,train_input,dev,dev_input,scale,initial,schedule,out,cache):
    summary = out/f"fit_{name}.json"
    journal = cache/f"fit_{name}.pt"
    sched_hash = tensor_sha(schedule)
    config = dict(name=name,seed=42,updates=6000,channels=train_input.shape[1],scale=scale,
                  schedule_sha256=sched_hash,train_P=tensor_sha(train.p),
                  train_input=tensor_sha(train_input),train_Y=tensor_sha(train.y),
                  dev_input=tensor_sha(dev_input) if dev_input is not None else None,
                  initialization=state_digest(initial),lr=1e-3,weight_decay=1e-5)
    if summary.exists():
        prior = read_json(summary)
        assert prior["config"]==config
        for row in prior["training"]:
            assert sha(out/row["checkpoint"])==row["sha256"]
        journal.unlink(missing_ok=True)
        print("REUSED",name,flush=True)
        return prior

    seed()
    device = train.p.device
    net = seeded(train_input.shape[1],initial,device)
    joint_init = state_digest(net.state_dict())
    opt = torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5999,eta_min=1e-6)
    schedule = schedule.to(device)
    weights = track_weights(train),track_weights(dev) if dev is not None else None
    labels = (train.y-train.p)/scale
    rows,stats = [],[]
    previous_time = 0.
    first = 0
    started = perf_counter()
    lr = 1e-3
    if journal.exists():
        state = torch.load(journal,map_location="cpu",weights_only=False)
        assert state["config"]==config and state["initial_digest"]==joint_init
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type=="cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        first,lr,rows,previous_time = (state[k] for k in ("step","lr","rows","elapsed"))
        stats = [v.to(device) for v in state["stats"]]
        print("RESUMED",name,first,flush=True)
    for step in range(first,6001):
        checkpoint = step in STEPS
        if checkpoint and not (rows and rows[-1]["step"]==step):
            dest = out/"checkpoints"/f"{name}_{step}.pt"
            current = {k:v.detach().cpu() for k,v in net.state_dict().items()}
            if dest.exists():
                past = torch.load(dest,map_location="cpu",weights_only=True)
                assert state_digest(past)==state_digest(current)
            else:
                write_tensor(dest,current)
            output = torch.stack(stats) if stats else None
            if output is not None:
                assert torch.isfinite(output).all()
            loss,grad = output.mean(0).tolist() if output is not None else (None,None)
            row = dict(run=name,step=step,checkpoint=str(dest.relative_to(out)),sha256=sha(dest),
                       schedule_sha256=sched_hash,lr=lr,loss=loss,gradient_norm=grad)
            for key,data,z,w in zip(("train","dev"),(train,dev),(train_input,dev_input),weights):
                if data is not None:
                    r = predict(net,z,scale)
                    if step==0:
                        assert not torch.count_nonzero(r) and torch.equal(data.p+r,data.p)
                    metric,decomp = checks(r,data,w)
                    row.update({f"{key}_{k}":v for k,v in metric.items()})
                    row[f"{key}_identity_error"] = decomp
                else:
                    row.update({f"{key}_{k}":None for k in ("mse","native","C","E","G","cosine","identity_error")})
            rows.append(row)
            stats.clear()
            print("STEP",name,step,row["train_mse"],row["dev_mse"],flush=True)
        if step%100==0 or checkpoint:
            write_tensor(journal,dict(config=config,initial_digest=joint_init,step=step,lr=lr,rows=rows,
                                      model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                                      python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                                      torch_rng=torch.get_rng_state(),
                                      cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else [],
                                      elapsed=previous_time+perf_counter()-started,
                                      stats=torch.stack(stats).cpu() if stats else []))
        if step<6000:
            ix = schedule[step]
            lr = opt.param_groups[0]["lr"]
            net.train()
            opt.zero_grad(set_to_none=True)
            value = (net(train_input[ix])-labels[ix]).square().mean()
            value.backward()
            gradient = torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            sched.step()
            stats.append(torch.stack((value.detach(),gradient.detach())))
    picked = min((r for r in rows[1:] if r["dev_E"] and r["dev_E"]>0),
                 key=lambda r:(r["dev_mse"],r["step"]),default=rows[-1])
    result = dict(config=config,selected=picked,final=rows[-1],training=rows,initial_digest=joint_init,
                  elapsed_seconds=previous_time+perf_counter()-started)
    dump(summary,result)
    journal.unlink(missing_ok=True)
    print("FINISHED",name,picked["step"],picked["dev_mse"],result["elapsed_seconds"],flush=True)
    return result


def ledger(out):
    rows = [row for file in sorted(out.glob("fit_*.json")) for row in read_json(file)["training"]]
    if rows:
        table(out/"training.csv",rows)
