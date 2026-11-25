"""Load and verify the previously issued P and causal historical OOF feedback."""
import torch

from research.corrector.run import sha
from research.corrector_residual_feedback.data import CACHE as HISTORY, OUT as LAST, ROOT, Triplets, guard
from research.corrector_residual_feedback.model import ARMS,base,encode
from research.corrector_window_study.data import RAW,read_json,tensor_sha

OUT = ROOT/"results/corrector_spacetime_optimization"
CACHE = ROOT/".cache/corrector_spacetime_optimization"
FIGURES = ROOT/".User/corrector_spacetime_optimization"
BRANCHES = ("B","E")


class Study:
    def __init__(self,out,cache,raw):
        self.out,self.cache = out,cache
        cache.mkdir(parents=True,exist_ok=True)
        self.prior = guard(LAST)
        self.cfg = self.prior["config"]
        self.scale = self.cfg["rms"]
        self.triplets = Triplets(out,HISTORY,raw)
        assert sha(out/"triplet_manifest.json")==self.prior["bindings"]["triplet_manifest.json"]
        self.norm = self.triplets.source.norm
        self.guards = {}
        self.old_runs = {name:read_json(LAST/f"fit_{name}.json") for name in BRANCHES}
        path = LAST/"schedule_main.pt"
        assert sha(path)==self.prior["bindings"]["schedule_main.pt"]
        self.schedule = torch.load(path,map_location="cpu",weights_only=True)
        assert self.schedule.shape==(6000,8)
        assert all(r["config"]["schedule_sha256"]==tensor_sha(self.schedule) for r in self.old_runs.values())
        self.crossfit = read_json(LAST/"crossfit_manifest.json")
        self.feedback_meta = read_json(LAST/"feedback_cache_manifest.json")
        self.verify_crossfit()

    def verify_crossfit(self):
        cases = self.triplets.plan["train"]["cases"]
        used = []
        for f in self.crossfit["folds"]:
            assert not set(f["holdout_files"])&set(f["training_files"])
            ids = f["holdout_indices"]
            assert all(cases[i]["file"] in f["holdout_files"] for i in ids)
            assert all(self.feedback_meta["train"]["fold"][i]==f["fold"] for i in ids)
            assert sha(LAST/f["reference"]["checkpoint"])==f["reference"]["sha256"]
            assert self.feedback_meta["train"]["sources"]["reference_sha256"][f["fold"]]==f["reference"]["sha256"]
            used.extend(ids)
        assert sorted(used)==list(range(len(cases)))

    def inputs(self,data):
        current = base(data.x,data.p,self.norm)
        true = data.x.double()-data.previous_p.double()
        predicted = data.historical_prediction
        remaining = true-predicted.double()
        design = {a:encode(current,(true,predicted,remaining),ARMS[a],self.scale) for a in BRANCHES}
        return current,design

    def get(self,name):
        if name=="test":
            guard(self.out)
            assert read_json(self.out/"FROZEN.json")["prior_test_manifest_blob"]==self.prior["prior_test_manifest_blob"]
            meta = read_json(LAST/"feedback_test_cache_manifest.json")
        else:
            meta = self.feedback_meta[name]
        prior_native = read_json(LAST/f"native_{name}.json")
        if prior_native["new_previous"]:
            cached_previous = HISTORY/f"previous_native_{name}.pt"
            assert sha(cached_previous)==prior_native["generated_cache_sha256"]
            assert read_json(cached_previous.with_suffix(".json"))["sha256"]==prior_native["generated_cache_sha256"]
        data = self.triplets.get(name)
        own = read_json(self.out/f"native_{name}.json")
        assert {k:v for k,v in own.items() if k!="test_freeze_sha256"}=={
            k:v for k,v in prior_native.items() if k!="test_freeze_sha256"}
        path = HISTORY/f"feedback_{name}.pt"
        assert sha(path)==meta["sha256"]
        predicted = torch.load(path,map_location=data.p.device,weights_only=True)
        true = data.x.double()-data.previous_p.double()
        remaining = true-predicted.double()
        for k,value in zip(("true","predicted","remaining"),(true,predicted,remaining)):
            assert tensor_sha(value)==meta["fields"][k]["sha256"]
        assert meta["cases"]==[[c["file"],c["start"]-20] for c in data.cases]
        if name!="train":
            assert meta["sources"]["reference_sha256"]==[self.prior["reference"]["selected"]["sha256"]]
        self.guards[name] = dict(prior_native=sha(LAST/f"native_{name}.json"),feedback_cache=sha(path),
                                 current_native=own["matched_P_sha256"],previous_native=own["previous_P_sha256"])
        data.historical_prediction = predicted
        current,design = self.inputs(data)
        if name!="test":
            for a,z in design.items():
                key = "train_input" if name=="train" else "dev_input"
                assert tensor_sha(z)==self.old_runs[a]["config"][key]
        return data,current,design
