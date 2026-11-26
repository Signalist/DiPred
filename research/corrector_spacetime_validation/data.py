"""Read only previously certified fields before the new validation freeze."""
from collections import defaultdict

from research.corrector.run import dump,sha
from research.corrector_residual_feedback.data import PRIOR as S3_ROOT,ROOT,guard
from research.corrector_residual_feedback.run import references
from research.corrector_spacetime_optimization.data import Study as PreviousStudy,OUT as PREVIOUS
from research.corrector_window_study.data import RAW,read_json,tensor_sha

OUT = ROOT/"results/corrector_spacetime_validation"
CACHE = ROOT/".cache/corrector_spacetime_validation"
FIGURES = ROOT/".User/corrector_spacetime_validation"
START_REV = "22f4e7c47fe925fbb53092cfda947ba43cb743ad"
HISTORICAL = ("B_MSE","E_MSE","B_ACC_raw_1","E_MATCH_raw_1","E_ext_match2","B_MAG_rho_0.1",
              "E_JOINT_tuned","E_JOINT_top2")


def fold_assignments(cases):
    regimes = defaultdict(set)
    for c in cases:
        regimes[c["re"]].add(c["file"])
    folds = [set() for _ in range(4)]
    for key in sorted(regimes):
        by_size = sorted(range(4),key=lambda i:(len(folds[i]),i))
        for i,file in enumerate(sorted(regimes[key])):
            folds[by_size[i%4]].add(file)
    assert all(folds) and sum(map(len,folds))==len(set().union(*folds))
    return folds


def outer_manifest(cases,configured):
    from types import SimpleNamespace

    from research.corrector_residual_optimization.fit import shared_plan

    by_re = defaultdict(set)
    for case in cases:
        by_re[case["re"]].add(case["file"])
    known = set().union(*by_re.values())
    assert min(by_re)==configured["lower"] and len(known)==46
    splits = {}
    for name,re in configured.items():
        held = by_re[re]
        active = known-held
        held_index = [i for i,c in enumerate(cases) if c["file"] in held]
        active_index = [i for i,c in enumerate(cases) if c["file"] in active]
        training = [cases[i] for i in active_index]
        rows = []
        for i,excluded in enumerate((*fold_assignments(training),set())):
            files = sorted(active-excluded)
            cohort = [c for c in training if c["file"] in files]
            rows.append(dict(fold=i,training_files=files,prediction_files=sorted(excluded or held),
                             sampling_sha256=tensor_sha(shared_plan(SimpleNamespace(cases=cohort)))))
        splits[name] = dict(re=re,held=held_index,active=active_index,held_files=sorted(held),active_files=sorted(active),
                            reference_plan=rows,
                            sampling_sha256=tensor_sha(shared_plan(SimpleNamespace(cases=training))))
    return splits


class Study(PreviousStudy):
    def __init__(self,out=OUT,cache=CACHE,raw=RAW):
        super().__init__(out,cache,raw)
        self.registry = guard(PREVIOUS)
        assert self.registry["prior_test_manifest_blob"]==self.prior["prior_test_manifest_blob"]
        self.plan = read_json(out/"experiment_plan.json")
        self.reference,self.s3_init,self.initial = references()
        self.splits = outer_manifest(self.triplets.plan["train"]["cases"],self.plan["verification"]["splits"])
        self.s3_scale = self.cfg["external_rms"]

    def prepare(self):
        train,base,z = self.get("train")
        dev,dev_base,dz = self.get("development")
        z["S3"],dz["S3"] = base,dev_base
        assert len(train.cases)==822 and len(dev.cases)==24
        assert len({c["file"] for c in train.cases})==46 and len({c["file"] for c in dev.cases})==12
        assert self.s3_scale==self.scale
        dump(self.out/"outer_split_manifest.json",self.splits)
        dump(self.out/"sources.json",dict(starting_commit=START_REV,backbone=self.triplets.source.identity,
                                          previous_freeze=sha(PREVIOUS/"FROZEN.json"),
                                          s3_freeze=sha(S3_ROOT/"FROZEN.json"),
                                          main_data=self.guards,shared_schedule=tensor_sha(self.schedule),
                                          s3_initial=sha(S3_ROOT/self.s3_init),
                                          historical={n:self.registry["selections"]["models"][n] for n in HISTORICAL},
                                          historical_s3=self.reference))
        return train,dev,z,dz
