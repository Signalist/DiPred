"""Development-side causal designs and immutable selection."""
import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.data import ROOT,guard,write_tensor
from research.corrector_residual_feedback.fit import state_digest
from research.corrector_simple_baseline.fit import table
from research.corrector_spacetime_validation.fit import atomic_json,checkpoint_file
from research.corrector_spacetime_validation.gradients import monitor_indices
from research.corrector_window_study.data import read_json,tensor_sha

from .diagnostics import calibrate,intervention,matched_e_rows,oracle,relationships
from .fit import Fits,open_head
from .model import inputs,labels,spec,temporal_basis,verify_basis
from .run import OPTIMIZATION,VALIDATION
from .teachers import chain_plan,check_chain_plan,main_bundle

HISTORY = ("B_MSE","E_MSE","B_ACC_raw_1","E_ext_match2","E_MATCH_raw_1")


def address(name,record,folder,kind,shared=None):
    def wrap(r):
        return dict(r,path=str(checkpoint_file(folder,r,shared).relative_to(ROOT)))
    return dict(record,name=name,kind=kind,training=[wrap(r) for r in record["training"]],
                selected=wrap(record["selected"]),final=wrap(record["final"]))


def best(models,names):
    return min(names,key=lambda arm:(models[arm]["selected"]["dev_mse"],models[arm]["selected"]["step"],arm))


def core(study,train,dev,train_base,dev_base,direct,z,dz,calibration_states):
    out=study.out
    x=main_bundle(study,train,train_base,"train")
    y=main_bundle(study,dev,dev_base,"development")
    q_plan=chain_plan(train.cases)
    chain_evidence=check_chain_plan(x,q_plan)
    basis=temporal_basis()
    basis_path=out/"temporal_basis.pt"
    if basis_path.exists():
        assert torch.equal(torch.load(basis_path,map_location="cpu",weights_only=True),basis)
    else:
        write_tensor(basis_path,basis)
    atomic_json(out/"temporal_basis.json",dict(sha256=sha(basis_path),tensor_sha256=tensor_sha(basis),
                                             sign="constant positive; linear increases; other peak positive",
                                             math=verify_basis(basis)))
    models={}
    ecal={k:address(k,r,out,"calibrated",OPTIMIZATION/"shared_E.pt") for k,r in direct.items()}
    models.update(ecal)
    old=guard(OPTIMIZATION)
    for name in HISTORY:
        r=read_json(OPTIMIZATION/f"run_{name}.json")
        assert all(old["checkpoints"][v["checkpoint"]]==v["sha256"] for v in r["training"])
        models["old_"+name]=address("old_"+name,r,OPTIMIZATION,"historical")
    prior=guard(VALIDATION)
    for name in ("E_EN_0.1","E_EN_0.3","E_DIAG","E_PERM"):
        models["old_"+name]=dict(prior["models"][name],name="old_"+name,kind="historical")
    engine=Fits(out,study.cache,x,y,study.schedule,basis)

    def fit(name,config,**kw):
        models[name]=address(name,engine.fit(name,config,**kw),out,"structured",kw.get("shared"))
        return name

    for h,suffix in enumerate(("0","B","E")):
        for family in ("I","D"):
            tag=family+suffix
            prefix=out/f"shared_{tag}.pt" if tag in ("IB","IE") else None
            fit(tag,spec(family,h),save_shared=prefix)
            paired="D"+suffix
            if family=="D":
                a,b=(models[n]["training"][0]["sha256"] for n in ("I"+suffix,paired))
                state=[torch.load(ROOT/models[n]["training"][0]["path"],map_location="cpu",weights_only=True)
                       for n in ("I"+suffix,paired)]
                assert state_digest(state[0])==state_digest(state[1]),(a,b)
    probes=monitor_indices(train.cases)
    auxiliary={}
    for h,branch in ((1,"IB"),(2,"IE")):
        prefix=out/f"shared_{branch}.pt"
        record=models[branch]
        row=next(r for r in record["training"] if r["step"]==1000)
        net=open_head(record,row,out,basis,train.p.device)
        cfg=record["config"]["model"]
        # Probe batches are small; the assembled features are never linked to labels.
        ids=sorted({i for batch in probes for i in batch})
        positions={v:i for i,v in enumerate(ids)}
        subset=[[positions[i] for i in a] for a in probes]
        coeff,norm=calibrate(net,inputs(x,ids,cfg),labels(x,ids,cfg),subset,("energy","match","match2"),.3)
        atomic_json(out/f"{branch}_calibration.json",dict(coefficients=coeff,norms=norm,probe_indices=probes))
        for loss,w in coeff.items():
            fit(f"{branch}_{loss}",spec("I",h,{loss:w}),shared=prefix)
        auxiliary[branch]=best(models,[f"{branch}_{k}" for k in coeff])

    chosen=best(models,("I0","IB","IE"))
    represented=[]
    for family in ("I","D"):
        name=family+chosen[1:]
        plain=models[name]["config"]["model"]
        for rank in (4,8):
            represented.append(fit(f"{name}_K{rank}",dict(plain,rank=rank)))
    atomic_json(out/"representation_selection.json",dict(incremental=chosen,direct="D"+chosen[1:],rank20="reused"))
    maps={}
    interventions=[]
    for kind in ("original","last","reverse","lag","dual","free"):
        name="IB" if kind=="original" else fit(f"IB_{kind}",spec("I",1,mix=kind))
        record=models[name]
        mappings=[]
        for policy in ("selected","final"):
            net=open_head(record,record[policy],out,basis,dev.p.device)
            rows,weights=intervention(net,y,record["config"]["model"],name)
            interventions.extend(dict(r,policy=policy) for r in rows)
            mappings.extend(dict(r,policy=policy) for r in weights)
        maps[kind]=mappings
    table(out/"time_maps.csv",[r for kind in maps.values() for r in kind])
    table(out/"history_interventions.csv",interventions)
    table(out/"history_relationships.csv",relationships(x))
    table(out/"oracle_modes.csv",oracle(x,basis))

    q_choice=auxiliary["IE"]
    if models[q_choice]["selected"]["dev_mse"]>=models["IE"]["selected"]["dev_mse"]:
        q_choice="IE_match2"
    seq_path=out/"sequence_schedule.pt"
    if seq_path.is_file():
        assert torch.equal(torch.load(seq_path,weights_only=True,map_location="cpu"),q_plan)
    else:
        write_tensor(seq_path,q_plan)
    atomic_json(out/"feedback_training_selection.json",dict(auxiliary=q_choice,schedule=tensor_sha(q_plan),
                                                           chain_length=3,**chain_evidence))
    q=[]
    for arm,tag in (("IE","MSE"),(q_choice,"aux")):
        cfg=models[arm]["config"]["model"]
        for mode in ("ref","self"):
            q.append(fit(f"Q_{tag}_{mode}",dict(cfg,sequence=mode),shared=out/"shared_IE.pt",sequence_plan=q_plan))

    c=read_json(out/"E_calibration.json")
    c["scale"]=study.scale
    gradients,leads=matched_e_rows(ROOT,ecal,train,dev,z["E"],dz["E"],c)
    table(out/"matched_gradients.csv",gradients)
    table(out/"matched_gradient_leads.csv",leads)
    active=[name for name,r in models.items() if r["kind"] in ("structured","calibrated")]
    new=best(models,active)
    zero=models["I0"]["training"][0]["dev_mse"]
    overall=new if models[new]["selected"]["dev_mse"]<zero else "S3"
    aux=best(models,auxiliary.values())
    chrono=best(models,[n for n in represented if n.startswith("I")]+[n for n in models if n.startswith("IB_") and
                models[n]["config"].get("model",{}).get("mix")!="original"])
    outer=list(dict.fromkeys(("I0","IB","IE","D"+chosen[1:],aux,chrono)))
    primary={"P1":[[new,"S3"]],"P2":[["IB","I0"],["IE","IB"]],
             "P3":[["I"+s,"D"+s] for s in ("0","B","E")],
             "P4":[[f"Ecal_match2_{r}",f"Ecal_{c}_{r}"] for r in (.1,.3) for c in ("diag","perm")]}
    selection=dict(best_new=new,overall=overall,best_MSE_input=chosen,representation=represented,
                   auxiliary=auxiliary,feedback=q,outer=outer,primary=primary)
    atomic_json(out/"selections.json",selection)
    bind=["PROTOCOL.md","hypotheses.json","experiment_plan.json","data_roles.json","reuse_manifest.json",
          "triplet_manifest.json","native_train.json","native_development.json","teacher_train.json","teacher_development.json",
          "E_calibration.json","IB_calibration.json","IE_calibration.json","temporal_basis.pt","temporal_basis.json",
          "time_maps.csv","history_interventions.csv","history_relationships.csv","oracle_modes.csv","matched_gradients.csv",
          "matched_gradient_leads.csv","selections.json","representation_selection.json","feedback_training_selection.json",
          "sequence_schedule.pt","shared_IB.pt","shared_IE.pt","outer_plan.json","period_plan.json",
          "validation_config.json","calibrated_state_archive.json"]
    # Reuse the actual pre-existing history plan without claiming fresh dates.
    atomic_json(out/"outer_plan.json",read_json(VALIDATION/"outer_split_manifest.json"))
    atomic_json(out/"period_plan.json",read_json(VALIDATION/"period_plan.json"))
    atomic_json(out/"validation_config.json",dict(main_policies=["selected","final"],outer_step=6000,
                                                period_step=6000,period_arms=sorted(models),
                                                sequence_models=q,run_modes=["ref","self"],
                                                bootstrap=dict(draws=10000,seed=42,unit="original trajectory"),
                                                feedback_interaction="(ref/self - ref/ref) - (self/self - self/ref)",
                                                scale=dict(main="frozen common training RMS",outer="outer active RMS"),
                                                frozen_controls=["Native","S3"],alpha=1.))
    hs={r["path"]:r["sha256"] for arm in models.values() for r in arm["training"]}
    bind.extend(sorted({s["path"] for r in models.values() for s in r.get("states",[])}))
    bind.extend(s["path"] for r in calibration_states.values() for s in r["retained"])
    assert len(active)==31 and len(outer)<=6
    assert all(sha(ROOT/path)==value for path,value in hs.items())
    atomic_json(out/"FROZEN.json",dict(prior_test_manifest_blob=old["prior_test_manifest_blob"],
                                     selection=selection,bindings={p:sha(out/p) for p in bind},
                                     checkpoints=hs,models=models,counts=dict(core=len(active),outer=len(outer)*3),
                                     omissions=["No independent confirmation trajectories in the signed inventory",
                                                "No optional optimization beyond the core arms"]))
    print("FROZEN",selection,flush=True)
