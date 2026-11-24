"""Matched AdamW runs and verification of the unchanged R0 reference."""
import csv
from functools import lru_cache

import torch

from research.corrector.run import sha
from research.corrector_predictability.alpha import moments, optimum, trajectory_indices
from research.corrector_predictability.data import OUT as PREVIOUS
from research.corrector_predictability.fit import schedule as old_schedule
from research.corrector_window_study.data import read_json, tensor_sha

from .diagnostics import update
from .model import input_fields, network, residual

SEEDS = (42,43,44)
STEPS = (0,50,100,250,500,750,1000,1500,2000)
KINDS = {"R0":"Point","R1":"Point","R2":"Block","R3":"Point","R4":"Block"}
MODES = tuple(KINDS)


def sampling(data, seed, base):
    tracks = tuple(trajectory_indices(data.cases).values())
    rand = torch.Generator().manual_seed(seed + 100000 if base else seed)
    n = 1000 if base else 2000
    track = torch.randint(len(tracks),(n,8),generator=rand).tolist()
    u = torch.rand(n,8,generator=rand).tolist()
    result = torch.tensor([[tracks[i][int(z*len(tracks[i]))] for i,z in zip(a,b)] for a,b in zip(track,u)])
    return torch.cat((old_schedule(data,seed),result)) if base else result


@lru_cache(None)
def r0_rows():
    with (PREVIOUS/"training.csv").open() as file:
        return {(int(r["seed"]),int(r["step"])):r for r in csv.DictReader(file) if r["mode"]=="Expanded"}


def load_model(out, mode, row, device):
    model = network(KINDS[mode],device)
    path = (PREVIOUS if mode=="R0" else out)/row["checkpoint"]
    assert sha(path)==row["sha256"], path
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    return model.eval()


def weights(data):
    tracks = trajectory_indices(data.cases)
    w = data.p.new_empty(len(data.cases),dtype=torch.float64)
    for ix in tracks.values():
        w[ix] = 1/(len(ix)*len(tracks))
    return w


def decomposition(pred, truth, w):
    c,e,native = moments(pred,truth)
    return tuple(float((a*w).sum()) for a in (c,e,native))


class Experiment:
    def __init__(self, train, development, norm, out, base):
        self.train, self.dev, self.out, self.base = train, development, out, base
        mean, units, self.scale = norm
        self.train_fields = input_fields(train,mean,units)
        self.dev_fields = input_fields(development,mean,units)
        self.train_target = (train.y-train.p)/self.scale
        self.train_truth = train.y.double()-train.p.double()
        self.dev_truth = development.y.double()-development.p.double()
        self.train_weights = weights(train)
        self.dev_weights = weights(development)

    def measure(self, model, mode, seed, step, checkpoint, digest, plan_hash, lr, loss, gradient):
        tr = residual(model,self.train_fields,self.scale)
        de = residual(model,self.dev_fields,self.scale)
        tc,te,tn = decomposition(tr,self.train_truth,self.train_weights)
        dc,de,dn = decomposition(de,self.dev_truth,self.dev_weights)
        train_window_raw = float((tr.double()-self.train_truth).square().mean())
        a = optimum(dc,de)
        row = dict(mode=mode,seed=seed,step=step,checkpoint=checkpoint,sha256=digest,schedule_sha256=plan_hash,
                   lr=lr,batch_objective=loss,gradient_norm=gradient,
                   train_mse=tn+te-2*tc,train_window_mse=train_window_raw,train_native_mse=tn,
                   train_C=tc,train_E=te,train_G=2*tc-te,
                   dev_mse=dn+de-2*dc,dev_native_mse=dn,dev_C=dc,dev_E=de,dev_G=2*dc-de,
                   dev_alpha=a,dev_fitted_mse=dn+a*a*de-2*a*dc)
        print("CHECKPOINT",mode,seed,step,"train",row["train_mse"],"dev",row["dev_mse"],"alpha",a,
              "dev_cal",row["dev_fitted_mse"],flush=True)
        return row

    def run(self, mode, seed):
        if mode=="R0":
            rows = []
            for step in STEPS[:7]:
                old = r0_rows()[seed,step]
                assert old["schedule_sha256"] == tensor_sha(old_schedule(self.train,seed))
                model = load_model(self.out,mode,old,self.train.x.device)
                row = self.measure(model,mode,seed,step,old["checkpoint"],old["sha256"],old["schedule_sha256"],
                                   .001,old["train_loss_mse"],old["train_gradient_norm"])
                for k1,k2 in (("train_window_mse","training_mse"),("dev_mse","dev_raw_mse"),
                              ("dev_fitted_mse","dev_fitted_mse"),("dev_C","dev_raw_inner"),("dev_E","dev_raw_energy")):
                    assert abs(row[k1]-float(old[k2])) < 2e-11, (k1, row[k1],old[k2])
                rows.append(row)
            return rows
        torch.manual_seed(seed)
        model = network(KINDS[mode],self.train.x.device)
        sampler = sampling(self.train,seed,self.base)
        plan_hash = tensor_sha(sampler)
        plan = sampler.cuda()
        opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=1999,eta_min=1e-6)
        rows, stats = [], []
        current_lr = opt.param_groups[0]["lr"]
        for step in range(2001):
            if step in STEPS:
                path = self.out/"checkpoints"/f"{mode}_{seed}_{step}.pt"
                assert not path.exists()
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
                loss, gradient = torch.stack(stats).mean(0).tolist() if stats else (None,None)
                row = self.measure(model,mode,seed,step,str(path.relative_to(self.out)),sha(path),
                                   plan_hash,current_lr,loss,gradient)
                assert not stats or torch.isfinite(torch.tensor([loss,gradient])).all()
                rows.append(row)
                stats.clear()
            if step<2000:
                model.train()
                ix = plan[step]
                current_lr = opt.param_groups[0]["lr"]
                loss,grad = update(model,opt,tuple(v[ix] for v in self.train_fields),self.train_target[ix])
                stats.append(torch.stack((loss,grad)))
                sched.step()
        return rows


def choices(rows):
    def choice(row, strategy):
        a = {"zero":0.,"raw":1.,"fit":row["dev_alpha"]}[strategy]
        mse = {"zero":row["dev_native_mse"],"raw":row["dev_mse"],"fit":row["dev_fitted_mse"]}[strategy]
        return dict(step=row["step"],checkpoint=row["checkpoint"],sha256=row["sha256"],alpha=a,
                    strategy=strategy,dev_mse=mse,dev_gain_pct=100*(1-mse/row["dev_native_mse"]))
    selected = min((choice(r,s) for r in rows for s in ("fit","raw","zero")),
                   key=lambda c:(c["dev_mse"],c["step"],c["alpha"]))
    final = rows[-1]
    at1000 = next(r for r in rows if r["step"]==1000)
    if rows[0]["mode"]=="R0":
        seed = rows[0]["seed"]
        prior = read_json(PREVIOUS/"FROZEN.json")["training_candidates"]["Expanded"][SEEDS.index(seed)]["selected"]
        assert selected["step"]==prior["step"] and selected["sha256"]==prior["sha256"]
        assert abs(selected["alpha"]-prior["alpha"]) < 1e-8
        # Preserve the exact development-frozen scalar from the original selection.
        selected.update(alpha=prior["alpha"], dev_mse=prior["development_mse"],
                        dev_gain_pct=prior["development_improvement_pct"])
    return dict(selected=selected,step1000_raw=choice(at1000,"raw"),step1000_fit=choice(at1000,"fit"),
                final_raw=choice(final,"raw"),final_fit=choice(final,"fit"))
