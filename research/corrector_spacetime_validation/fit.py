"""Resumable paired fits with a byte-bound common optimizer prefix."""
from shutil import copyfile
from time import perf_counter

import torch

from research.corrector.run import dump,sha
from research.corrector_residual_feedback.data import write_tensor
from research.corrector_residual_feedback.fit import STEPS,seed,state_digest
from research.corrector_residual_feedback.model import model,predict,seeded
from research.corrector_residual_optimization.fit import checks,track_weights
from research.corrector_residual_optimization.model import network
from research.corrector_spacetime_optimization.fit import START,config_hash,random_state,restore
from research.corrector_window_study.data import read_json,tensor_sha

from .objectives import objective

UPDATES = 6000


def make_net(branch,device,layout=None):
    if branch=="S3":
        return model(6,device) if layout=="NCDHW" else network(dict(input="S3",width=32),device)
    return model(12,device)


def load_net(path,digest,branch,device,layout=None):
    assert sha(path)==digest,path
    net = make_net(branch,device,layout)
    net.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    return net


def checkpoint_file(out,row,shared=None):
    """The borrowed MSE prefix is stored beside its originating optimizer state."""
    folder = shared.parent if row.get("phase")=="shared_MSE" and shared is not None else out
    return folder/row["checkpoint"]


def atomic_json(path,value):
    tmp = path.with_suffix(".tmp")
    dump(tmp,value)
    tmp.replace(path)


class Fits:
    def __init__(self,out,cache,train,design,scale,schedule,initial,dev=None,dev_design=None,reference_layout=False):
        self.out,self.cache = out,cache
        self.train,self.z,self.scale = train,design,scale
        # Inputs remain in the original layouts; only the S3 forward changes its layout.
        self.dev,self.dz = dev,dev_design
        self.device = train.p.device
        self.plan = schedule.to(self.device)
        self.plan_hash = tensor_sha(schedule)
        self.initial = initial
        self.reference_layout = reference_layout
        self.labels = (train.y-train.p)/scale
        self.identities = dict(train=tensor_sha(design),target=tensor_sha(self.labels),
                               development=tensor_sha(dev_design) if dev_design is not None else None)
        self.weights = (track_weights(train),track_weights(dev) if dev is not None else None)
        (out/"checkpoints").mkdir(parents=True,exist_ok=True)
        (out/"fits").mkdir(exist_ok=True)
        cache.mkdir(parents=True,exist_ok=True)

    def fit(self,name,branch,weights,shared=None,save_shared=None,origin=None,archive_states=False):
        weights = {k:float(v) for k,v in weights.items() if v}
        assert all(0<v<=100 for v in weights.values())
        cfg = dict(name=name,branch=branch,weights=weights,seed=42,updates=UPDATES,scale=self.scale,
                   inputs=self.identities,schedule=self.plan_hash,lr=.001,weight_decay=1e-5,
                   clip=1.,batch=8,lr_end=1e-6,steps=list(STEPS),first=START if shared else 0,
                   layout="channels_last_3d" if branch=="S3" and not self.reference_layout else "NCDHW",
                   shared=sha(shared) if shared else None,origin=origin)
        if archive_states:
            assert shared is None
            cfg["archive_states"]=True
        digest = config_hash(cfg)
        meta = self.out/"fits"/f"{name}.json"
        journal = self.cache/f"{name}.pt"
        if meta.exists():
            result = read_json(meta)
            assert result["config"]==cfg and result["config_hash"]==digest
            for row in result["training"]:
                assert sha(checkpoint_file(self.out,row,shared))==row["sha256"]
            if archive_states:
                assert all(sha(self.out/r["path"])==r["sha256"] for r in result["states"])
            return result
        seed()
        if branch=="S3" and not self.reference_layout:
            net = make_net(branch,self.device)
        else:
            net = seeded(6 if branch=="S3" else 12,self.initial,self.device)
        opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=UPDATES-1,eta_min=1e-6)
        start,elapsed,rows,stats = 0,0.,[],[]
        started = perf_counter()
        if journal.exists():
            state = torch.load(journal,map_location="cpu",weights_only=False)
            assert state["config_hash"]==digest
            restore(net,opt,sched,state)
            start,elapsed,rows = (state[k] for k in ("step","elapsed","rows"))
            stats = [v.to(self.device) for v in state["stats"]]
            print("RESUMED",name,start,flush=True)
        elif shared:
            state = torch.load(shared,map_location="cpu",weights_only=False)
            assert state["step"]==state["scheduler"]["last_epoch"]==START
            assert all(int(v["step"])==START for v in state["optimizer"]["state"].values())
            assert state["schedule_sha256"]==self.plan_hash and state["branch"]==branch
            restore(net,opt,sched,state)
            start = START
            rows = [dict(row,run=name,phase="shared_MSE") for row in state["rows"]]
        for step in range(start,UPDATES+1):
            mark = step in STEPS
            if mark and not (rows and rows[-1]["step"]==step):
                dest = self.out/"checkpoints"/f"{name}_{step}.pt"
                cpu = {k:v.detach().cpu() for k,v in net.state_dict().items()}
                if dest.exists():
                    assert state_digest(torch.load(dest,map_location="cpu",weights_only=True))==state_digest(cpu)
                else:
                    write_tensor(dest,cpu)
                if stats:
                    measured = torch.stack(stats)
                    assert torch.isfinite(measured).all()
                    loss,mse,grad,post,clip_rate = measured.mean(0).tolist()
                else:
                    loss,mse,grad,post,clip_rate = (None,)*5
                row = dict(run=name,step=step,checkpoint=str(dest.relative_to(self.out)),sha256=sha(dest),
                           lr=opt.param_groups[0]["lr"],phase="auxiliary" if weights else "MSE",
                           loss=loss,batch_mse=mse,gradient_norm=grad,gradient_after_clip=post,clip_rate=clip_rate)
                for key,data,z,w in zip(("train","dev"),(self.train,self.dev),(self.z,self.dz),self.weights):
                    if data is None:
                        continue
                    estimate = predict(net,z,self.scale)
                    assert torch.isfinite(estimate).all()
                    if step==0:
                        assert not torch.count_nonzero(estimate) and torch.equal(data.p+estimate,data.p)
                    metrics,error = checks(estimate,data,w)
                    row.update({f"{key}_{k}":v for k,v in metrics.items()})
                    row[f"{key}_identity_error"] = error
                rows.append(row)
                stats.clear()
                print("STEP",name,step,row["train_mse"],row.get("dev_mse"),flush=True)
            if step%100==0 or mark:
                state = dict(config_hash=digest,model=net.state_dict(),optimizer=opt.state_dict(),
                             scheduler=sched.state_dict(),rng=random_state(),rows=rows,step=step,
                             branch=branch,schedule_sha256=self.plan_hash,lr=opt.param_groups[0]["lr"],
                             elapsed=elapsed+perf_counter()-started,stats=torch.stack(stats).cpu() if stats else [])
                if step==START and save_shared and not save_shared.exists():
                    write_tensor(save_shared,state)
                write_tensor(journal,state)
                if mark and archive_states:
                    snapshot=self.out/"training_states"/f"{name}_{step}.pt"
                    snapshot.parent.mkdir(exist_ok=True)
                    if not snapshot.exists():
                        staged=snapshot.with_suffix(".tmp")
                        copyfile(journal,staged)
                        staged.replace(snapshot)
            if step==UPDATES:
                break
            ix = self.plan[step]
            net.train()
            opt.zero_grad(set_to_none=True)
            r = net(self.z[ix])
            y = self.labels[ix]
            total = objective(r,y,weights)
            total.backward()
            g = torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            sched.step()
            stats.append(torch.stack((total.detach(),(r.detach()-y).square().mean(),g.detach(),
                                      g.detach().clamp_max(1.),(g>1).float())))
        picked = (min((r for r in rows if r["step"] and r["dev_E"]>0),
                      key=lambda r:(r["dev_mse"],r["step"])) if self.dev is not None else rows[-1])
        result = dict(config=cfg,config_hash=digest,training=rows,selected=picked,final=rows[-1],
                      elapsed_seconds=elapsed+perf_counter()-started)
        if archive_states:
            states=[self.out/"training_states"/f"{name}_{r['step']}.pt" for r in rows]
            result["states"]=[dict(step=r["step"],path=str(p.relative_to(self.out)),sha256=sha(p))
                              for r,p in zip(rows,states)]
        atomic_json(meta,result)
        journal.unlink(missing_ok=True)
        print("FINISHED",name,picked["step"],picked.get("dev_mse"),result["elapsed_seconds"],flush=True)
        return result
