"""Freeze-bound scoring of trained, swapped, and explicitly missing histories."""
from pathlib import Path

import numpy as np
import torch

from research.corrector.run import sha
from research.corrector_history_free_refinement.evaluate import RefinementScores, later_period
from research.corrector_incremental_factorial.core import load_head, step_row
from research.corrector_incremental_structure.fit import Fits, predict
from research.corrector_incremental_structure.teachers import bundle, full_teacher, main_bundle
from research.corrector_incremental_structure.validation import confirmation_inventory
from research.corrector_residual_feedback.data import guard
from research.corrector_residual_feedback.model import base, predict as teacher_prediction
from research.corrector_simple_baseline.fit import table
from research.corrector_spacetime_validation.expansion import period_data
from research.corrector_spacetime_validation.fit import atomic_json
from research.corrector_spacetime_validation.outer import subset
from research.corrector_window_study.data import read_json, tensor_sha

from .core import prior
from .donors import assign
from .outer import fit_outer


def cases_for_scores(out, outer):
    manifest=read_json(out/"triplet_manifest.json")
    period=read_json(out/"period_plan.json")["chains"]
    later=read_json(out/"later_plan.json")
    def flatten(specs,key):
        return [dict(file=r["file"],re=r["re"],start=s) for r in specs for s in r[key]]
    return dict(development=manifest["development"]["cases"],historical_test=manifest["test"]["cases"],
                period=flatten(period,"starts"),period_new=flatten(period,"new_vs_prior_window_study"),
                period_extended=flatten(later,"starts"),
                **{f"holdout_{name}":view.data.cases for name,view in outer.items()})


def keys(cases):
    return [[r["file"],r["start"],r["re"]] for r in cases]


def freeze_all(out, main, outer, views, teachers, status, plan, gradients):
    assessments={}
    rule=read_json(out/"donor_plan.json")
    for label,cases in cases_for_scores(out,views).items():
        donor=assign(cases,np.arange(len(cases)),rule["assessment_seed"],rule["time_slack_frames"])
        assessments[label]=dict(cases=keys(cases),indices=donor.tolist(),tensor_sha256=tensor_sha(donor))
    atomic_json(out/"assessment_donors.json",assessments)
    atomic_json(out/"teacher_lineage.json",dict(
        main={s:read_json(out/f"teacher_{s}.json") for s in ("train","development")},
        outer={name:r["teacher"] for name,r in outer.items()},additional=status,
        backbone=main["backbone"],previous_freeze=main["previous_freeze"]))
    atomic_json(out/"outer_fits.json",outer)
    table(out/"gradient_diagnostics.csv",gradients)
    atomic_json(out/"additional_teachers.json",teachers)
    bound=("MAIN_FROZEN.json","assessment_donors.json","teacher_lineage.json","outer_fits.json",
           "additional_teachers.json","donor_outer.json","history_donors_outer.csv","gradient_diagnostics.csv",
           *(r["file"] for r in read_json(out/"donor_outer.json").values()))
    freeze=dict(main,bindings={**main["bindings"],**{n:sha(out/n) for n in bound}})
    freeze["additional"]=status
    freeze["new_runs"]=len(plan["new"])+len(teachers)+sum(
        r["directory"].endswith(out.name) for v in outer.values()
        for r in (*v["models"].values(),*v["training_only"].values()))
    assert freeze["new_runs"]<=plan["budget"]
    path=out/"FROZEN.json"
    if path.exists():
        assert guard(out)==freeze
    else:
        atomic_json(path,freeze)
    print("FROZEN",sha(path),"new runs",freeze["new_runs"],flush=True)
    return freeze


def policies(record, main_record, cohort):
    if cohort in ("development","historical_test") or cohort.startswith("holdout_"):
        yield "step500",step_row(record,500)
        yield "step1000",step_row(record,1000)
    yield "final",record["final"]
    yield "selected",step_row(record,main_record["selected"]["step"])


class Evaluation:
    def __init__(self,study,freeze,basis,book,plan):
        self.study,self.freeze,self.basis,self.book,self.plan=study,freeze,basis,book,plan
        self.assessments=read_json(study.out/"assessment_donors.json")
        self.checkpoints=[]
        self.gain=[]
        self.strata=[]
        self.sensitivity=[]
        self.assigned=[]
        self.thresholds=read_json(study.out/"observable_thresholds.json")
        self.example=None

    @torch.inference_mode()
    def score(self, split, view, records):
        from .metrics import conditional_rows, observations, sensitivity_rows

        registered=self.assessments[split]
        cases=view.data.cases
        assert keys(cases)==registered["cases"]
        donor=torch.tensor(registered["indices"],device=view.current.device)
        assert tensor_sha(donor)==registered["tensor_sha256"]
        can_move=donor!=torch.arange(len(cases),device=donor.device)
        for i,j in enumerate(registered["indices"]):
            a,b=cases[i],cases[j]
            assert a["re"]==b["re"] and (i==j or a["file"]!=b["file"])
            self.assigned.append(dict(scope=split,recipient=i,donor=j,count=1,re=a["re"],
                recipient_file=a["file"],donor_file=b["file"],recipient_start=a["start"],donor_start=b["start"],moved=i!=j))
        std=self.study.norm["std"]
        normalization=getattr(view,"identity",{}).get("normalization")
        if normalization:
            std=view.current.new_tensor(normalization["std"]).view(1,2,1,1,1)
        controls={}
        scopes={p for p,_ in policies(next(iter(records.values())),next(iter(self.freeze["models"].values())),split)}
        for policy in sorted(scopes):
            self.book.add_view(view,split,"Native",policy,torch.zeros_like(view.current),std)
            self.book.add_view(view,split,"S3",policy,view.current,std)
        controls["S3"]=view.current
        for arm,record in records.items():
            for policy,step in policies(record,self.freeze["models"][arm],split):
                config=record["config"]["model"]
                net=load_head(record,step,self.basis,view.current.device)
                actual=predict(net,view,config)
                old,truth=view.target,view.data.y
                try:
                    view.data.y=truth.roll(1,0)+.03125
                    view.target=old.roll(1,0)+.03125
                    assert torch.equal(actual,predict(net,view,config))
                finally:
                    view.target,view.data.y=old,truth
                self.book.add_view(view,split,arm,policy,actual,std)
                self.checkpoints.append(dict(split=split,arm=arm,policy=policy,step=step["step"],
                    checkpoint=step["checkpoint"],sha256=step["sha256"],future_target_invariant=True))
                if config.get("paired_history"):
                    without=predict(net,view,config,history_available=False)
                    clean=type(view)(**{k:v for k,v in vars(view).items() if k not in ("true","historical")})
                    assert torch.equal(without,predict(net,clean,config,history_available=False))
                    self.book.add_view(view,split,arm+"__missing",policy,without,std)
                    self.gain.extend(conditional_rows(split,arm,policy,cases,view.target,actual,without,self.basis))
                    if policy=="final":
                        controls[arm+"__missing"]=without
                base_arm=arm.split("_")[0]
                if base_arm in ("IB","ID") and arm.endswith(("_MSE","_EN")) and "_pair_" not in arm:
                    swapped=predict(net,view,config,history_ix=donor)
                    self.book.add_view(view,split,arm+"__wrong",policy,swapped,std)
                    if policy=="final":
                        self.sensitivity.extend(sensitivity_rows(split,arm,cases,view.target,actual,swapped,can_move))
                if policy=="final":
                    controls[arm]=actual
        self.strata.extend(observations(split,view,self.thresholds,controls))
        if split=="development":
            self.example=(view,controls)
        print("SCORED",split,len(cases),flush=True)


@torch.inference_mode()
def ordinary(study, evaluator, freeze):
    for role, split in (("development","development"),("test","historical_test")):
        data,features,_=study.get(role)
        view=main_bundle(study,data,features,role)
        evaluator.score(split,view,freeze["models"])
    period,evidence=period_data(study,data)
    later=later_period(study,data,period,read_json(study.out/"later_plan.json"))
    last=[i for i,c in enumerate(period.cases) if c["beyond_prior_window"]]
    teacher,_=full_teacher(study,data.p.device)
    for split, cohort in (("period",period),("period_new",subset(period,last)),("period_extended",later)):
        z=base(cohort.x,cohort.p,study.norm)
        history=base(cohort.previous_x,cohort.previous_p,study.norm)
        scale=study.cfg["external_rms"]
        view=bundle(cohort,z,teacher_prediction(teacher,z,scale),teacher_prediction(teacher,history,scale),
                    study.scale,{"split":split})
        evaluator.score(split,view,freeze["models"])
    independent=confirmation_inventory(study,read_json(study.out/"period_plan.json"))
    atomic_json(study.out/"period_provenance.json",dict(evidence,independent=independent,counts=[96,48,168]))
    return independent


def validate(study, figures):
    from .diagnostics import gradient_ratios
    from .metrics import comparisons, own_controls
    from .release import release

    out=study.out
    main=read_json(out/"MAIN_FROZEN.json")
    assert all(sha(out/name)==value for name,value in main["bindings"].items())
    plan=read_json(out/"experiment_plan.json")
    old,audit,basis=prior()
    assert main["previous_freeze"]==read_json(out/"data_roles.json")["previous_freeze_sha256"]
    data,features,_=study.get("train")
    train=main_bundle(study,data,features,"train")
    dev_engine=Fits(out,study.cache,train,None,study.schedule,basis)
    slots=torch.load(out/"donor_batches_main.pt",map_location="cpu",weights_only=True)
    gradients=gradient_ratios(dev_engine,{n:main["models"][n] for n in plan["new"]},slots)
    del train,dev_engine,features
    thresholds=(data.y.double()-data.p.double()).abs()[:,:,::2,::2,::2]
    book=RefinementScores(basis,[float(torch.quantile(thresholds[:,c].flatten(),.95)) for c in (0,1)])
    del thresholds
    outer, views, teachers, status, outer_gradients=fit_outer(study,data,main,plan,audit,basis)
    del data
    freeze=freeze_all(out,main,outer,views,teachers,status,plan,gradients+outer_gradients)
    evaluator=Evaluation(study,freeze,basis,book,plan)
    for label,view in views.items():
        evaluator.score(f"holdout_{label}",view,outer[label]["models"])
    independent=ordinary(study,evaluator,freeze)
    scores=book.write(out,own_controls(freeze["models"]),comparisons(freeze),shard_raw=True)
    release(evaluator,figures,outer,teachers,scores,independent)
