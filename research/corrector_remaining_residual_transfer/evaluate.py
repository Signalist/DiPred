"""Freeze-bound scoring and matched held-regime fits."""
import torch

from research.corrector.run import sha
from research.corrector_history_free_refinement.evaluate import RefinementScores, later_period
from research.corrector_incremental_factorial.core import load_head,prefix,stamp,step_row
from research.corrector_incremental_structure.fit import Fits,predict
from research.corrector_incremental_structure.teachers import bundle,full_teacher,main_bundle,outer_bundles
from research.corrector_incremental_structure.validation import confirmation_inventory,reuse_period_cache,target_invariance
from research.corrector_residual_feedback.data import guard
from research.corrector_residual_feedback.model import base,predict as teacher_predict
from research.corrector_simple_baseline.fit import table
from research.corrector_spacetime_validation.expansion import period_data
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_spacetime_validation.outer import subset
from research.corrector_window_study.data import ROOT,read_json,tensor_sha

from .core import archive_record,prior,verify_archive
from .interventions import examine
from .run import FACTORIAL,REFINEMENT,STRUCTURE


def comparison_lists(selection, plan):
    def pairs(split, present):
        clauses=[("S3_reference",n,"S3") for n in present if n not in ("S3","Native")]
        for loss in plan["terms"]:
            clauses += [("P1",f"ID_{loss}",f"IB_{loss}"),("P2",f"ID_{loss}",f"I0_{loss}")]
            for name in ("IB","IC","ID","IE","IF"):
                clauses.append(("same_loss",f"{name}_{loss}",f"I0_{loss}"))
            for a,b in (("IE","IB"),("IF","IB"),("IF","ID"),("IF","IE")):
                clauses.append(("combination",f"{a}_{loss}",f"{b}_{loss}"))
            if loss!="MSE":
                for name in plan["inputs"]:
                    clauses.append(("P3" if name=="ID" else "own_MSE",f"{name}_{loss}",f"{name}_MSE"))
        clauses.extend([("P4",selection[p],"S3") for p in ("final","selected")])
        for a in plan["focal"]:
            for loss in plan["early"]:
                clauses.extend([("early_fixed",f"{a}_{loss}_early",f"{a}_MSE"),
                                ("early_vs_late",f"{a}_{loss}_early",f"{a}_{loss}")])
        for loss in ("TM1","TM2"):
            fixed={a:f"{a}_{loss}_fixed" for a in plan["focal"]}
            if loss=="TM1":
                fixed["I0"]="I0_TM1"
            clauses.extend(("common_weight",fixed[a],fixed[b]) for a,b in (("ID","IB"),("ID","I0"),("IB","I0")))
        for name in selection["extensions"]:
            clauses.append(("mapping",name,"ID_MSE" if name.endswith("_MSE") else "ID_TM1"))
        return sorted({p for p in clauses if p[1]!=p[2] and all(n in present for n in p[1:])})
    return pairs


def policy_rows(record, selected=None):
    return dict(final=record["final"], selected=record["selected"] if selected is None else step_row(record,selected),
                step500=step_row(record,500),step1000=step_row(record,1000))


@torch.inference_mode()
def add_models(study, book, view, split, models, basis, rows, policies, audits):
    freeze_digest=sha(study.out/"FROZEN.json")
    for policy in policies:
        book.add_view(view,split,"Native",policy,torch.zeros_like(view.current),study.norm["std"])
        book.add_view(view,split,"S3",policy,view.current,study.norm["std"])
        for arm,record in models.items():
            if policy not in rows[arm]:
                continue
            step=rows[arm][policy]
            net=load_head(record,step,basis,view.current.device)
            output=predict(net,view,record["config"]["model"])
            target_invariance(view,lambda:None,dict(record,kind="structured"),net,output)
            audits.append(dict(split=split,arm=arm,policy=policy,step=step["step"],
                               checkpoint=step["checkpoint"],checkpoint_sha256=step["sha256"],
                               config_hash=record["config_hash"],freeze_sha256=freeze_digest,bitwise=True))
            book.add_view(view,split,arm,policy,output,study.norm["std"])


def choose_outer(region, arm, model):
    a,b=arm.split("_",1)
    possible=[]
    if b=="MSE" and a in ("I0","IB"):
        possible.append((STRUCTURE,f"outer_{region}_{a}"))
    elif a=="I0" and b in ("TM1","TM2","EN"):
        possible.append((REFINEMENT,f"outer_{region}_I0_{b}_rho03"))
    elif arm=="IB_TM2":
        possible.append((STRUCTURE,f"outer_{region}_IB_match2"))
    for folder, name in possible:
        path=folder/"structured_fits"/f"{name}.json"
        if path.is_file():
            r=archive_record(folder,name)
            if r["config"]["model"]==model:
                return r
    return None


def outer(study,train,freeze,audit,basis,book,plan,proof):
    old=read_json(FACTORIAL/"outer_lineage.json")
    lineage, all_records={},{}
    models=freeze["models"]
    names=[f"{arm}_{loss}" for arm in plan["focal"] for loss in plan["terms"]]
    extra=freeze["selection"]["outer_extra"]
    if extra:
        names.append(extra)
    for region in plan["outer"]:
        print("OUTER START",region,flush=True)
        active,held,schedule=outer_bundles(study,train,region)
        assert active.identity=={k:v for k,v in old["regimes"][region].items() if k!="candidates"}
        engine=Fits(study.out,study.cache,active,None,schedule,basis)
        records={}
        for name in names:
            model=models[name]["config"]["model"]
            family=name.split("_")[0]
            start=models[name]["config"]["first"]
            shared=prefix(records[f"{family}_MSE"],study.out,through=start) if start else None
            found=choose_outer(region,name,model)
            if found:
                candidate_shared=shared
                if name=="IB_TM2":
                    candidate_shared=STRUCTURE/f"shared_outer_{region}_IB.pt"
                verify_archive(found,engine,audit,model,candidate_shared)
                assert found["config"]["first"]==start
                records[name]=found
            else:
                tag=f"outer_{region}_{name}"
                records[name]=stamp(engine.fit(tag,model,shared=shared,shared_step=start or 1000),study.out)
        rows={n:policy_rows(item,models[n]["selected"]["step"]) for n,item in records.items()}
        add_models(study,book,held,f"holdout_{region}",records,basis,rows,("final","selected","step500","step1000"),proof)
        lineage[region]=dict(held.identity,candidates=records)
        all_records.update({f"outer_{region}_{name}":r for name,r in records.items()})
        atomic_json(study.out/"outer_lineage.json",dict(freeze_sha256=sha(study.out/"FROZEN.json"),regimes=lineage))
        print("OUTER COMPLETE",region,len(records),flush=True)
    return all_records


def fields_csv(out, view, fixed, freeze, basis):
    i,=[i for i,c in enumerate(view.data.cases) if (c["file"],c["start"])==(fixed["file"],fixed["start"])]
    lead=fixed["lead"]-1
    record=freeze["models"][freeze["selection"]["final"]]
    net=load_head(record,record["final"],basis,view.current.device)
    result=predict(net,view,record["config"]["model"])
    physical=dict(history_R=view.true[i,:,lead],history_P=view.historical[i,:,lead],
                  history_E=(view.true-view.historical.double())[i,:,lead],
                  current_E=(view.target-view.current.double())[i,:,lead],
                  increment=(result-view.current.double())[i,:,lead])
    rows=[dict(field=n,channel=c,y=a,x=b,value=v) for n,arr in physical.items()
          for c,component in zip(("u","v"),arr.tolist()) for a,line in enumerate(component) for b,v in enumerate(line)]
    table(out/"fixed_fields.csv",rows)


@torch.inference_mode()
def ordinary(study,freeze,basis,book,plan,proof):
    models=freeze["models"]
    rows={n:policy_rows(r) for n,r in models.items()}
    interventions,assignments=[],[]
    for flag,split in (("development","development"),("test","historical_test")):
        data,z,_=study.get(flag)
        view=main_bundle(study,data,z,flag)
        add_models(study,book,view,split,models,basis,rows,("final","selected","step500","step1000"),proof)
        effects,donor_rows=examine(view,models,plan,basis,split)
        interventions.extend(effects)
        assignments.extend(donor_rows)
        if flag=="development":
            fields_csv(study.out,view,plan["field"],freeze,basis)
            from .publication import publication_checks

            publication_checks(study,view,freeze,basis,plan)
        print("SCORED",split,flush=True)
    table(study.out/"history_interventions.csv",interventions)
    table(study.out/"history_donors.csv",assignments)
    reuse_period_cache(study)
    period,evidence=period_data(study,data)
    last=[i for i,c in enumerate(period.cases) if c["beyond_prior_window"]]
    later_plan=read_json(study.out/"later_plan.json")
    later=later_period(study,data,period,later_plan)
    table(study.out/"later_window_provenance.csv",[
        dict(c,history=tensor_sha(later.x[i]),native=tensor_sha(later.p[i]),future=tensor_sha(later.y[i]))
        for i,c in enumerate(later.cases)])
    anchor,_=full_teacher(study,data.p.device)
    scale=study.cfg["external_rms"]
    historical_tracks={c["file"] for c in data.cases}
    cohorts={"historical_test":dict(windows=len(data.cases),trajectories=len(historical_tracks))}
    for split,cohort,count in (("period",period,96),("period_new",subset(period,last),48),
                               ("period_extended",later,168)):
        tracks={c["file"] for c in cohort.cases}
        assert tracks==historical_tracks and len(cohort.cases)==count
        cohorts[split]=dict(windows=count,trajectories=len(tracks))
        z=base(cohort.x,cohort.p,study.norm)
        current=teacher_predict(anchor,z,scale)
        history=teacher_predict(anchor,base(cohort.previous_x,cohort.previous_p,study.norm),scale)
        view=bundle(cohort,z,current,history,study.scale,{"split":split})
        add_models(study,book,view,split,models,basis,rows,("final","selected"),proof)
        print("SCORED",split,len(cohort.cases),flush=True)
    independent=confirmation_inventory(study,read_json(study.out/"period_plan.json"))
    # The historical files contain no further complete 20+20 window after the last registered issue.
    assert all(max(p["starts"])+20+40>p["frames"] for p in later_plan)
    atomic_json(study.out/"period_provenance.json",dict(evidence,independent=independent,windows=[96,48,168],cohorts=cohorts,
                   later_hash=sha(study.out/"later_window_provenance.csv")))


def validate(study,figures):
    frozen=guard(study.out)
    assert frozen["prior_freeze"]==sha(REFINEMENT/"FROZEN.json")
    _,audit,basis=prior()
    plan=read_json(study.out/"experiment_plan.json")
    data,_,_=study.get("train")
    train_errors=(data.y.double()-data.p.double()).abs()[:,:,::2,::2,::2]
    thresholds=[float(torch.quantile(train_errors[:,i].contiguous().flatten(),.95)) for i in (0,1)]
    del train_errors
    book=RefinementScores(basis,thresholds)
    proof=[]
    outer_models=outer(study,data,frozen,audit,basis,book,plan,proof)
    del data
    ordinary(study,frozen,basis,book,plan,proof)
    controls={name:name.split("_")[0]+"_MSE" for name in frozen["models"]}
    evaluation=book.write(study.out,controls,comparison_lists(frozen["selection"],plan),shard_raw=True)
    from .release import complete

    complete(study.out,figures,frozen,outer_models,dict(evaluation,target_invariance=proof),plan)
