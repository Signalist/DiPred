"""Frozen checkpoint bookkeeping and one shared seed42 batch schedule."""
import csv
import re
from pathlib import Path
from time import perf_counter

import torch

from research.corrector.run import sha
from research.corrector_predictability.alpha import moments, trajectory_indices
from research.corrector_simple_baseline.fit import SEED, STEPS, table
from research.corrector_window_study.data import tensor_sha

from .model import components, inputs, network

LONG = (*STEPS,3000,4000,5000,6000)
DEFAULT = dict(width=8,lr=3e-4,weight_decay=1e-5,updates=2000,aux={})


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def plan(split,updates=6000):
    tracks = tuple(trajectory_indices(split.cases).values())
    gen = torch.Generator().manual_seed(SEED)
    track = torch.randint(len(tracks),(updates,8),generator=gen).tolist()
    pos = torch.rand(updates,8,generator=gen).tolist()
    return torch.tensor([[tracks[i][int(z*len(tracks[i]))] for i,z in zip(a,b)] for a,b in zip(track,pos)])


def shared_plan(split):
    # Preserve the exact preceding study's 2000 updates for every short run.
    return torch.cat((plan(split,2000),plan(split,6000)[2000:]))


def track_weights(data):
    weights = torch.empty(len(data.cases),dtype=torch.float64,device=data.p.device)
    tracks = trajectory_indices(data.cases)
    for ix in tracks.values():
        weights[ix] = 1/(len(tracks)*len(ix))
    return weights


@torch.inference_mode()
def predict(net,x,scale):
    net.eval()
    return torch.cat([net(chunk)*scale for chunk in x.split(8)])


def checks(residual,data,weights):
    r = residual.double()
    tgt = data.y.double()-data.p.double()
    c,e,b = moments(r,tgt)
    m = (data.p.double()+r-data.y.double()).square().mean((1,2,3,4))
    g = 2*c-e
    gap = float((b-m-g).abs().max())
    assert gap<1e-15
    values = dict(mse=m,native=b,C=c,E=e,G=g,cosine=c/(b*e).sqrt().clamp_min(1e-30))
    return {k:float((v*weights).sum()) for k,v in values.items()},gap


def load(config,row,out,device):
    file = Path(row["external_checkpoint"]) if row["external_checkpoint"] else out/row["checkpoint"]
    assert sha(file)==row["sha256"],file
    net = network(config,device)
    net.load_state_dict(torch.load(file,map_location=device,weights_only=True))
    return net


class Experiment:
    def __init__(self,train,dev,norm,scale,out,schedule,namespace=""):
        self.train,self.dev,self.norm,self.scale = train,dev,norm,scale
        self.out,self.namespace = out,namespace
        self.schedule = schedule
        self.weights = track_weights(train),track_weights(dev) if dev is not None else None
        self.rows = []
        self.runs = {}
        self.measurements = {}
        self.ledger = out/f"{namespace}training.csv"
        self.pending = {}
        if self.ledger.exists():
            text_fields = ("run","checkpoint","external_checkpoint","sha256","schedule_sha256")
            for raw in read(self.ledger):
                row = {k:(v if k in text_fields else int(v) if k=="step" else float(v) if v else None)
                       for k,v in raw.items()}
                self.pending.setdefault(row["run"],[]).append(row)
        self.saved_times = {}
        log = out/"train_run.log"
        if log.exists():
            for run,seconds in re.findall(r"^FINISHED (\S+) \S+ \S+ ([0-9.]+)s$",
                                         log.read_text(),re.MULTILINE):
                self.saved_times[run] = float(seconds)
        self.progress = out/".cache"
        self.progress.mkdir(exist_ok=True)

    def fields(self,data,mode):
        key = id(data),mode
        if key not in self.measurements:
            self.measurements[key] = inputs(data,self.norm,mode)
        return self.measurements[key]

    def capture(self,net,config,stats):
        for name,split,w in (("train",self.train,self.weights[0]),("dev",self.dev,self.weights[1])):
            if split is not None:
                r = predict(net,self.fields(split,config["input"]),self.scale)
                row, gap = checks(r,split,w)
                stats.update({f"{name}_{k}":v for k,v in row.items()})
                stats[f"{name}_identity_error"] = gap
            else:
                stats.update({f"{name}_{k}":None for k in ("mse","native","C","E","G","cosine","identity_error")})
        return stats

    def replay(self,previous):
        config = dict(DEFAULT,input="S0")
        sched = tensor_sha(self.schedule[:2000])
        if self.restore("S0",config,"input",STEPS,sched):
            return "S0"
        rows = []

        for old in read(previous/"training.csv"):
            assert old["schedule_sha256"]==sched
            row = dict(run="S0",step=int(old["step"]),checkpoint=old["checkpoint"],
                       external_checkpoint=str(previous/old["checkpoint"]),sha256=old["sha256"],
                       schedule_sha256=sched,lr=float(old["lr"]),batch_loss=old["scaled_batch_objective"],
                       gradient_norm=old["gradient_norm"])
            net = load(config,row,self.out,self.train.p.device)
            self.capture(net,config,row)
            for a,b in (("train","train"),("dev","development")):
                for metric in ("mse","C","E","G"):
                    assert abs(float(old[b+"_"+metric])-row[a+"_"+metric])<2e-13
            rows.append(row)
        return self.register("S0",config,rows,"input",0.)

    def restore(self,name,config,stage,steps,schedule_hash):
        rows = self.pending.get(name)
        if rows is None:
            return False
        assert [r["step"] for r in rows]==sorted(steps)
        for row in rows:
            file = Path(row["external_checkpoint"]) if row["external_checkpoint"] else self.out/row["checkpoint"]
            assert sha(file)==row["sha256"] and row["schedule_sha256"]==schedule_hash
        self.pending.pop(name)
        self.register(name,config,rows,stage,self.saved_times.get(name,0.),restored=True)
        return True

    def register(self,name,config,rows,stage,elapsed,restored=False):
        picked = min((r for r in rows if r["dev_E"] and r["dev_E"]>0),
                     key=lambda r:(r["dev_mse"],r["step"]),default=rows[-1])
        self.runs[name] = dict(config=config,selected=picked,stage=stage,elapsed_seconds=elapsed)
        self.rows.extend(rows)
        ledger = self.ledger.with_suffix(".tmp")
        table(ledger,[*self.rows,*(r for group in self.pending.values() for r in group)])
        ledger.replace(self.ledger)
        status = "REUSED" if restored else "FINISHED"
        print(status,name,picked["step"],picked["dev_mse"],f"{elapsed:.2f}s",flush=True)
        return name

    def fit(self,name,config,stage,stop=None,record_steps=None):
        assert name not in self.runs
        start = perf_counter()
        total = config["updates"]
        stop = total if stop is None else stop
        sampler = self.schedule[:stop]
        sched_sha = tensor_sha(sampler)
        steps = {s for s in (LONG if record_steps is None else record_steps) if s<=stop}
        assert stop in steps and 0 in steps
        journal = self.progress/f"{name}.pt"
        if self.restore(name,config,stage,steps,sched_sha):
            journal.unlink(missing_ok=True)
            return name
        torch.manual_seed(SEED)
        net = network(config,self.train.p.device)
        x = self.fields(self.train,config["input"])
        labels = (self.train.y-self.train.p)/self.scale
        sampler = sampler.to(self.train.p.device)
        opt = torch.optim.AdamW(net.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total-1,eta_min=1e-6)
        rows,stats = [],[]
        lr = config["lr"]
        first = 0
        prior_elapsed = 0.
        signature = dict(config=config,schedule_sha256=sched_sha,steps=sorted(steps),scale=self.scale,
                         normalization={k:v.flatten().tolist() for k,v in self.norm.items()})
        if journal.exists():
            state = torch.load(journal,map_location="cpu",weights_only=False)
            assert state["signature"]==signature
            net.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
            first,lr,prior_elapsed = (state[k] for k in ("step","lr","elapsed"))
            rows = state["rows"]
            stats = [v.to(self.train.p.device) for v in state["stats"]]
            print("RESUMED",name,first,flush=True)
        for step in range(first,stop+1):
            if step in steps and not (rows and rows[-1]["step"]==step):
                dest = self.out/"checkpoints"/f"{name}_{step}.pt"
                weights = {k:v.detach().cpu() for k,v in net.state_dict().items()}
                if dest.exists():
                    saved = torch.load(dest,map_location="cpu",weights_only=True)
                    assert saved.keys()==weights.keys() and all(torch.equal(saved[k],v) for k,v in weights.items())
                else:
                    tmp = dest.with_suffix(".tmp")
                    torch.save(weights,tmp)
                    tmp.replace(dest)
                if stats:
                    stacked = torch.stack(stats)
                    assert torch.isfinite(stacked).all()
                    loss,grad = stacked.mean(0).tolist()
                else:
                    loss,grad = None,None
                row = dict(run=name,step=step,checkpoint=str(dest.relative_to(self.out)),external_checkpoint="",
                           sha256=sha(dest),schedule_sha256=sched_sha,lr=lr,batch_loss=loss,gradient_norm=grad)
                rows.append(self.capture(net,config,row))
                stats.clear()
                print("STEP",name,step,row["train_mse"],row["dev_mse"],flush=True)
            if step%100==0 or step in steps:
                tmp = journal.with_suffix(".tmp")
                torch.save(dict(signature=signature,model=net.state_dict(),optimizer=opt.state_dict(),
                                scheduler=sched.state_dict(),step=step,lr=lr,rows=rows,
                                stats=torch.stack(stats).cpu() if stats else [],
                                elapsed=prior_elapsed+perf_counter()-start),tmp)
                tmp.replace(journal)
            if step<stop:
                net.train()
                i = sampler[step]
                lr = opt.param_groups[0]["lr"]
                opt.zero_grad(set_to_none=True)
                values = components(net(x[i]),labels[i],tuple(config["aux"]))
                loss = values["mse"]+sum(a*values[k] for k,a in config["aux"].items())
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
                sched.step()
                stats.append(torch.stack((loss.detach(),grad.detach())))
        result = self.register(name,config,rows,stage,prior_elapsed+perf_counter()-start)
        journal.unlink(missing_ok=True)
        return result

    def best(self,names):
        return min(names,key=lambda k:(self.runs[k]["selected"]["dev_mse"],k))
