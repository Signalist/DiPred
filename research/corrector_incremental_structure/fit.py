"""Checkpointed single-window and paired sequence training."""
from shutil import copyfile
from time import perf_counter

import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.data import write_tensor
from research.corrector_residual_feedback.fit import STEPS,seed,state_digest
from research.corrector_residual_optimization.fit import checks,track_weights
from research.corrector_spacetime_optimization.fit import START,config_hash,random_state,restore
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_window_study.data import read_json,tensor_sha

from .model import Head,inputs,residual,train_loss

UPDATES=6000


@torch.inference_mode()
def predict(net,view,cfg,history_ix=None,history_available=True):
    net.eval()
    idx=torch.arange(len(view.data.cases),device=view.current.device)
    batches=[]
    for i in idx.split(8):
        donor=None if history_ix is None else history_ix[i]
        features=inputs(view,i,cfg,history_ix=donor,history_available=history_available)
        batches.append(residual(view,i,cfg,net(features)))
    return torch.cat(batches)


def open_head(record,row,out,basis,device):
    path=out/row["checkpoint"]
    assert sha(path)==row["sha256"]
    head=Head(record["config"]["model"],basis,device)
    head.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    return head


class Fits:
    def __init__(self,out,cache,train,dev,schedule,basis):
        self.out,self.cache=out,cache
        self.train,self.dev=train,dev
        self.schedule=schedule
        self.basis=basis
        self.device=train.current.device
        self.w=track_weights(train.data),track_weights(dev.data) if dev else None
        self.identity=dict(native=tensor_sha(train.data.p),features=tensor_sha(train.base),
                           current=tensor_sha(train.current),previous=tensor_sha(train.historical),
                           observed=tensor_sha(train.true),target=tensor_sha(train.target),
                           development=dev.identity if dev else None)
        for path in (out/"checkpoints",out/"structured_fits",out/"training_states",cache):
            path.mkdir(parents=True,exist_ok=True)

    def fit(self,name,cfg,shared=None,save_shared=None,sequence_plan=None,replayed_prefix=None,shared_step=START,donor_plan=None,end_step=None):
        terminal=UPDATES if end_step is None else end_step
        assert terminal in STEPS and (terminal==UPDATES or shared is None)
        schedule=self.schedule if sequence_plan is None else sequence_plan
        assert (schedule.ndim==3)==bool(cfg["sequence"])
        assert schedule.shape==(UPDATES,8,3) if cfg["sequence"] else schedule.shape==(UPDATES,8)
        assert shared_step in STEPS and (shared_step == START or shared is not None)
        plan_hash=tensor_sha(schedule)
        reference_hash=tensor_sha(self.schedule)
        job=dict(name=name,model=cfg,inputs=self.identity,scale=self.train.scale,seed=42,updates=UPDATES,
                 batch=8,steps=list(STEPS),first=shared_step if shared else 0,
                 lr=.001,lr_end=1e-6,weight_decay=1e-5,clip=1.,layout="NCDHW",schedule=plan_hash,
                 prefix_schedule=reference_hash,basis=tensor_sha(self.basis),shared=sha(shared) if shared else None)
        if terminal!=UPDATES:
            job["end_step"]=terminal
        architecture = {k:cfg[k] for k in ("history","rank","mix","family")}
        if "history_fields" in cfg:
            architecture["history_fields"] = cfg["history_fields"]
        if cfg.get("paired_history",False):
            architecture["paired_history"]=True
        if donor_plan is not None:
            assert not cfg["sequence"] and not cfg.get("paired_history",False)
            assert donor_plan.shape==schedule.shape and donor_plan.dtype==torch.long
            assert int(donor_plan.min())>=0 and int(donor_plan.max())<len(self.train.data.cases)
            architecture["donor_batch_sha256"]=job["donor_batch_sha256"]=tensor_sha(donor_plan)
        if replayed_prefix is not None:
            assert shared is not None and self.dev is not None
            job["prefix_development_replay"]=config_hash(replayed_prefix)
        ids=config_hash(job)
        path=self.out/"structured_fits"/f"{name}.json"
        journal=self.cache/f"structured_{name}.pt"
        if path.is_file():
            old=read_json(path)
            assert old["config_hash"]==ids and old["config"]==job
            assert all(sha(self.out/r["checkpoint"])==r["sha256"] for r in old["training"])
            assert all(r["sha256"]==sha(self.out/r["path"]) for r in old["states"])
            return old
        seed()
        net=Head(cfg,self.basis,self.device)
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=UPDATES-1,eta_min=1e-6)
        rows,stats=[],[]
        start,elapsed=0,0.
        started=perf_counter()
        if journal.exists():
            state=torch.load(journal,map_location="cpu",weights_only=False)
            assert state["config_hash"]==ids
            restore(net,opt,sch,state)
            start,elapsed,rows=state["step"],state["elapsed"],state["rows"]
            stats=[x.to(self.device) for x in state["stats"]]
        elif shared:
            state=torch.load(shared,map_location="cpu",weights_only=False)
            assert state["step"]==state["scheduler"]["last_epoch"]==shared_step
            assert all(int(a["step"])==shared_step for a in state["optimizer"]["state"].values())
            prefix_identity = dict(state["identity"])
            if replayed_prefix is not None:
                assert prefix_identity.pop("development")==replayed_prefix["original_development"]
                prefix_identity["development"]=self.identity["development"]
            assert prefix_identity==self.identity and state["schedule"]==reference_hash
            assert state["architecture"]==architecture
            restore(net,opt,sch,state)
            start=shared_step
            rows=[dict(row,run=name,phase="shared_MSE") for row in state["rows"]]
            if replayed_prefix is not None:
                for row in rows:
                    row.update(replayed_prefix["metrics"][str(row["step"])])
        ix=schedule.to(self.device)
        donor_ids=donor_plan.to(self.device) if donor_plan is not None else None
        for step in range(start,terminal+1):
            mark=step in STEPS
            if mark and not (rows and rows[-1]["step"]==step):
                dest=self.out/"checkpoints"/f"{name}_{step}.pt"
                weights={k:v.detach().cpu() for k,v in net.state_dict().items()}
                if dest.exists():
                    assert state_digest(torch.load(dest,map_location="cpu",weights_only=True))==state_digest(weights)
                else:
                    write_tensor(dest,weights)
                measured=torch.stack(stats) if stats else None
                if measured is not None:
                    assert torch.isfinite(measured).all()
                values=measured.mean(0).tolist() if measured is not None else [None]*4
                row=dict(run=name,step=step,checkpoint=str(dest.relative_to(self.out)),sha256=sha(dest),
                         phase="sequence" if cfg["sequence"] else "paired",lr=opt.param_groups[0]["lr"],
                         **dict(zip(("loss","grad","postclip","clip_frequency"),values)))
                for label,view,weight in zip(("train","dev"),(self.train,self.dev),self.w):
                    if view is None:
                        continue
                    output=predict(net,view,cfg)
                    assert torch.isfinite(output).all()
                    if step==0:
                        goal=view.current.double() if cfg["family"]=="I" else torch.zeros_like(output)
                        assert torch.equal(output,goal)
                    metrics,error=checks(output,view.data,weight)
                    row.update({f"{label}_{k}":v for k,v in metrics.items()})
                    if cfg.get("paired_history",False):
                        missing=predict(net,view,cfg,history_available=False)
                        assert torch.isfinite(missing).all()
                        row[f"{label}_missing_mse"]=checks(missing,view.data,weight)[0]["mse"]
                    delta=output-view.current.double()
                    remaining=view.target-view.current.double()
                    c,u=(float((x.mean((1,2,3,4))*weight).sum()) for x in (delta*remaining,delta.square()))
                    row.update({f"{label}_identity_error":error,f"{label}_C_delta":c,
                                f"{label}_U_delta":u,f"{label}_G_delta":2*c-u})
                rows.append(row)
                stats.clear()
                print("STEP",name,step,row["train_mse"],row.get("dev_mse"),flush=True)
            if mark or step%100==0:
                state=dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),rng=random_state(),
                           step=step,rows=rows,stats=torch.stack(stats).cpu() if stats else [],
                           elapsed=elapsed+perf_counter()-started,config_hash=ids,identity=self.identity,
                           schedule=plan_hash,architecture=architecture)
                if step==START and save_shared and not save_shared.exists():
                    write_tensor(save_shared,state)
                write_tensor(journal,state)
                if mark and rows[-1]["phase"]!="shared_MSE":
                    snapshot=self.out/"training_states"/f"{name}_{step}.pt"
                    if not snapshot.exists():
                        partial=snapshot.with_suffix(".tmp")
                        copyfile(journal,partial)
                        partial.replace(snapshot)
            if step==terminal:
                break
            net.train()
            opt.zero_grad(set_to_none=True)
            donor=None if donor_ids is None else donor_ids[step]
            value=train_loss(net,self.train,ix[step],cfg,history_ix=donor)
            value.backward()
            grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            sch.step()
            stats.append(torch.stack((value.detach(),grad.detach(),grad.detach().clamp_max(1.),(grad>1.).float())))
        selected=min(rows,key=lambda r:(r["dev_mse"],r["step"])) if self.dev else rows[-1]
        tail=terminal-shared_step if shared else terminal
        sequence=3 if sequence_plan is not None else 1+int(cfg.get("paired_history",False))
        snapshots=[]
        for r in rows:
            checkpoint=self.out/"training_states"/r["checkpoint"].split("/")[-1]
            snapshots.append(dict(step=r["step"],path=str(checkpoint.relative_to(self.out)),sha256=sha(checkpoint),
                                  borrowed=r["phase"]=="shared_MSE"))
        result=dict(config=job,config_hash=ids,training=rows,selected=selected,final=rows[-1],
                    elapsed_seconds=elapsed+perf_counter()-started,
                    states=snapshots,
                    cost=dict(own_updates=tail,forward_training_calls=tail*sequence,
                              backward_calls=tail,supervised_windows=tail*sequence*8,
                              borrowed_updates=shared_step if shared else 0))
        if cfg["history"] and cfg["mix"]!="original":
            result["mapping"]=net.mapping().detach().cpu().tolist()
        atomic_json(path,result)
        journal.unlink(missing_ok=True)
        print("FINISHED",name,selected["step"],selected.get("dev_mse"),result["elapsed_seconds"],flush=True)
        return result
