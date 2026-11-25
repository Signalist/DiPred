"""Replay MSE and branch once from its full optimizer state at step 1000."""
from hashlib import sha256
import json
import random
from time import perf_counter

import numpy as np
import torch

from research.corrector.run import dump,sha
from research.corrector_residual_feedback.data import write_tensor,OUT as LAST
from research.corrector_residual_feedback.fit import STEPS,seed,state_digest
from research.corrector_residual_feedback.model import model,predict
from research.corrector_residual_feedback.run import references
from research.corrector_residual_feedback.model import seeded
from research.corrector_residual_optimization.fit import checks,track_weights
from research.corrector_simple_baseline.fit import table
from research.corrector_window_study.data import read_json,tensor_sha

from .loss import objective

START = 1000


def config_hash(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def random_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore(net,opt,sched,state):
    net.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state_all(rng["cuda"])


def checked_net(row,out,device):
    path = out/row["checkpoint"]
    assert sha(path)==row["sha256"],path
    net = model(12,device)
    net.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    return net


class Runner:
    def __init__(self,study,train,dev,z_train,z_dev):
        self.study = study
        self.out,self.cache = study.out,study.cache
        self.train,self.dev = train,dev
        self.z,self.dz = z_train,z_dev
        self.schedule = study.schedule.to(train.p.device)
        self.scale = study.scale
        self.labels = (train.y-train.p)/self.scale
        self.plan_hash = tensor_sha(study.schedule)
        self.w = track_weights(train),track_weights(dev)
        self.initial = references()[2]
        self.runs = {}
        self.identity = self.out/"checkpoints"
        self.identity.mkdir(exist_ok=True)

    def train_arm(self,branch,arm,weights,stage):
        name = f"{branch}_{arm}"
        if name in self.runs:
            assert self.runs[name]["config"]["weights"]==weights
            return name
        first = 0 if arm=="MSE" else START
        shared = self.out/f"shared_{branch}.pt"
        cfg = dict(branch=branch,arm=arm,weights=weights,stage=stage,seed=42,updates=6000,
                   first=first,shared_sha256=sha(shared) if first else None,
                   schedule_sha256=self.plan_hash,scale=self.scale,lr=.001,weight_decay=1e-5,
                   previous_train_input=self.study.old_runs[branch]["config"]["train_input"],
                   previous_dev_input=self.study.old_runs[branch]["config"]["dev_input"])
        digest = config_hash(cfg)
        manifest = self.out/f"run_{name}.json"
        journal = self.cache/f"{name}.pt"
        if manifest.exists():
            record = read_json(manifest)
            assert record["config_hash"]==digest and record["config"]==cfg
            assert all(sha(self.out/r["checkpoint"])==r["sha256"] for r in record["training"])
            self.runs[name] = record
            print("REUSED",name,flush=True)
            return name
        seed()
        net = seeded(12,self.initial,self.train.p.device)
        opt = torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5999,eta_min=1e-6)
        rows,stats = [],[]
        lr = .001
        started = perf_counter()
        elapsed = 0.
        if journal.exists():
            state = torch.load(journal,map_location="cpu",weights_only=False)
            assert state["config_hash"]==digest
            restore(net,opt,sched,state)
            first,lr,rows,elapsed = (state[k] for k in ("step","lr","rows","elapsed"))
            stats = [v.to(self.train.p.device) for v in state["stats"]]
            print("RESUMED",name,first,flush=True)
        elif first:
            state = torch.load(shared,map_location="cpu",weights_only=False)
            assert state["step"]==START and state["branch"]==branch and state["schedule_sha256"]==self.plan_hash
            assert state["config_hash"]==self.runs[f"{branch}_MSE"]["config_hash"]
            assert state["rows"][-1]["sha256"]==sha(self.out/state["rows"][-1]["checkpoint"])
            restore(net,opt,sched,state)
            rows = [dict(r,run=name,phase="shared_MSE") for r in state["rows"]]
            lr = state["lr"]
        for step in range(first,6001):
            checkpoint = step in STEPS
            if checkpoint and not (rows and rows[-1]["step"]==step):
                dest = self.identity/f"{name}_{step}.pt"
                current = {k:v.detach().cpu() for k,v in net.state_dict().items()}
                if dest.exists():
                    prev = torch.load(dest,map_location="cpu",weights_only=True)
                    assert state_digest(prev)==state_digest(current)
                else:
                    write_tensor(dest,current)
                measured = torch.stack(stats) if stats else None
                if measured is not None:
                    assert torch.isfinite(measured).all()
                loss,mse,grad,post = measured.mean(0).tolist() if measured is not None else (None,)*4
                row = dict(run=name,step=step,checkpoint=str(dest.relative_to(self.out)),sha256=sha(dest),
                           lr=lr,phase="auxiliary" if weights else "MSE",loss=loss,
                           batch_mse=mse,gradient_norm=grad,gradient_after_clip=post)
                for key,data,z,w in zip(("train","dev"),(self.train,self.dev),(self.z[branch],self.dz[branch]),self.w):
                    r = predict(net,z,self.scale)
                    if step==0:
                        assert not torch.count_nonzero(r) and torch.equal(data.p+r,data.p)
                    metric,error = checks(r,data,w)
                    row.update({f"{key}_{k}":v for k,v in metric.items()})
                    row[f"{key}_identity_error"]=error
                rows.append(row)
                stats.clear()
                print("STEP",name,step,row["train_mse"],row["dev_mse"],flush=True)
            if step%100==0 or checkpoint:
                state = dict(config_hash=digest,step=step,lr=lr,rows=rows,model=net.state_dict(),
                             optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=random_state(),
                             elapsed=elapsed+perf_counter()-started,branch=branch,schedule_sha256=self.plan_hash,
                             stats=torch.stack(stats).cpu() if stats else [])
                if arm=="MSE" and step==START and not shared.exists():
                    write_tensor(shared,state)
                write_tensor(journal,state)
            if step<6000:
                ix = self.schedule[step]
                lr = opt.param_groups[0]["lr"]
                net.train()
                opt.zero_grad(set_to_none=True)
                r = net(self.z[branch][ix])
                target = self.labels[ix]
                total = objective(r,target,weights)
                mse = total.detach() if not weights else (r.detach()-target).square().mean()
                total.backward()
                grad = torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
                sched.step()
                stats.append(torch.stack((total.detach(),mse,grad.detach(),grad.detach().clamp_max(1.))))
        pick = min((r for r in rows if r["step"] and r["dev_E"]>0),key=lambda r:(r["dev_mse"],r["step"]))
        record = dict(config=cfg,config_hash=digest,selected=pick,final=rows[-1],training=rows,
                      elapsed_seconds=elapsed+perf_counter()-started)
        dump(manifest,record)
        journal.unlink(missing_ok=True)
        self.runs[name] = record
        self.ledger()
        print("FINISHED",name,pick["step"],pick["dev_mse"],record["elapsed_seconds"],flush=True)
        return name

    def ledger(self):
        rows = [r for f in sorted(self.out.glob("run_*.json")) for r in read_json(f)["training"]]
        dest = self.out/"training.tmp"
        table(dest,rows)
        dest.replace(self.out/"training.csv")

    def verify_mse(self,branch):
        current = self.runs[f"{branch}_MSE"]
        prior = self.study.old_runs[branch]
        result = []
        for a,b in zip(current["training"],prior["training"]):
            assert a["step"]==b["step"]
            assert sha(LAST/b["checkpoint"])==b["sha256"]
            a_weights = torch.load(self.out/a["checkpoint"],map_location="cpu",weights_only=True)
            b_weights = torch.load(LAST/b["checkpoint"],map_location="cpu",weights_only=True)
            max_error = max(float((v-b_weights[k]).abs().max()) for k,v in a_weights.items())
            test_error = abs(a["dev_mse"]-b["dev_mse"])
            assert max_error<=1e-7 and test_error<=1e-12
            result.append(dict(step=a["step"],checkpoint_max_error=max_error,
                               dev_mse_max_error=test_error,previous_sha256=b["sha256"]))
        assert current["selected"]["step"]==prior["selected"]["step"]
        shared = self.out/f"shared_{branch}.pt"
        state = torch.load(shared,map_location="cpu",weights_only=False)
        assert state["step"]==state["scheduler"]["last_epoch"]==START
        assert state["config_hash"]==current["config_hash"] and state["schedule_sha256"]==self.plan_hash
        assert all(int(v["step"])==START for v in state["optimizer"]["state"].values())
        assert state_digest(state["model"])==state_digest(torch.load(self.out/state["rows"][-1]["checkpoint"],
                                                                    map_location="cpu",weights_only=True))
        return dict(shared_state_sha256=sha(shared),replay=result)
