"""Matched observed triplets and audited four-call native release caches."""
from collections import defaultdict
from hashlib import sha1
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from research.corrector.run import dump, sha
from research.corrector_simple_baseline.data import Data, CACHE as NATIVE_CACHE, OUT as SIMPLE, RAW, ROOT
from research.corrector_window_study.data import read_json, tensor_sha

OUT = ROOT/"results/corrector_residual_feedback"
CACHE = ROOT/".cache/corrector_residual_feedback"
PRIOR = ROOT/"results/corrector_residual_optimization"
FIGURES = ROOT/".User/corrector_residual_feedback"
LABELS = [99,66,33,0]


def keys(cases):
    return [(c["file"],c["start"]) for c in cases]


def write_tensor(path,value):
    temp = path.with_suffix(".tmp")
    torch.save(value,temp)
    temp.replace(path)


def guard(out):
    f = read_json(out/"FROZEN.json")
    for k,v in f["bindings"].items():
        assert sha(out/k)==v,k
    return f


class Triplets:
    def __init__(self,out,cache,raw=RAW):
        self.out,self.cache = out,cache
        cache.mkdir(parents=True,exist_ok=True)
        self.source = Data(raw,NATIVE_CACHE,out)
        self.plan = {}
        for split,cases in self.source.cases.items():
            frames = {r["file"]:r["frames"] for r in self.source.records[split]}
            kept,rejected = [],[]
            for i,c in enumerate(cases):
                start = c["start"]
                if start<20 or start+40>frames[c["file"]]:
                    rejected.append(dict(c,reason="incomplete consecutive real windows"))
                else:
                    kept.append(dict(c,index=i,earlier=[start-20,start],issue_frame=start+20,score_frame=start+40))
            self.plan[split] = dict(cases=kept,excluded=rejected)
        dump(out/"triplet_manifest.json",self.plan)

    @torch.inference_mode()
    def get(self,name):
        if name=="test":
            f = guard(self.out)
            payload = (SIMPLE/"native_test.json").read_bytes()
            assert sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest()==f["prior_test_manifest_blob"]
        ds = self.source
        record = read_json(SIMPLE/f"native_{name}.json")
        meta = read_json(NATIVE_CACHE/f"{name}.json")
        assert {k:v for k,v in record.items() if k!="test_freeze_sha256"}==meta
        identity = meta["identity"]
        assert identity["protocol"]=="single zero-initial-noise path, batch 1"
        assert identity["denoising_labels"]==LABELS and identity["backbone"]==ds.identity
        assert identity["cases"]==ds.cases[name]
        x,y = ds.fields(name)
        assert tensor_sha(x)==identity["x_sha256"] and tensor_sha(y)==identity["y_sha256"]
        cache = NATIVE_CACHE/f"{name}.pt"
        assert sha(cache)==meta["file_sha256"]
        p = torch.load(cache,map_location=x.device,weights_only=True)
        assert p.shape==x.shape==y.shape and tensor_sha(p)==meta["P_sha256"]

        cases = self.plan[name]["cases"]
        indices = [c["index"] for c in cases]
        prior_x = torch.empty_like(x[indices])
        prior_p = torch.empty_like(prior_x)
        by_file = defaultdict(list)
        lookup = {k:i for i,k in enumerate(keys(ds.cases[name]))}
        missing = []
        cached = 0
        for i,c in enumerate(cases):
            start = c["start"]
            old = lookup.get((c["file"],start-20))
            if old is not None:
                assert torch.equal(y[old],x[c["index"]])
                prior_x[i],prior_p[i] = x[old],p[old]
                cached += 1
            else:
                by_file[c["file"]].append((i,start))
                missing.append([c["file"],start-20])
        for file,items in by_file.items():
            with h5py.File(ds.raw/file,"r") as handle:
                for i,start in items:
                    fields = np.stack([handle[c][start-20:start:1,::2,::2] for c in ("u","v")]).astype("float32")
                    assert fields.shape==(2,20,32,64) and np.isfinite(fields).all()
                    prior_x[i] = torch.from_numpy(fields).to(x.device)
        extra,manifest = self.cache/f"previous_native_{name}.pt",self.cache/f"previous_native_{name}.json"
        order = [i for items in by_file.values() for i,_ in items]
        recipe = dict(keys=[[cases[i]["file"],cases[i]["start"]-20] for i in order],history_sha256=tensor_sha(prior_x[order]),
                      backbone=ds.identity,denoising_labels=LABELS,zero_initial_noise=True)
        assert set(map(tuple,recipe["keys"]))==set(map(tuple,missing)) and len(missing)==len(order)
        if order:
            if extra.exists() and manifest.exists():
                old = read_json(manifest)
                assert old["recipe"]==recipe and sha(extra)==old["sha256"]
                added = torch.load(extra,map_location=x.device,weights_only=True)
            else:
                calls = []
                def observe(_,args):
                    if not calls:
                        assert not torch.count_nonzero(args[0])
                    assert len(args[0])==1
                    calls.append(int(args[1].item()))
                hook = ds.backbone.net.register_forward_pre_hook(observe)
                try:
                    added = []
                    for i in order:
                        calls.clear()
                        history = (prior_x[i:i+1]-ds.norm["mean"])/ds.norm["std"]
                        normalized = ds.backbone.rollout(history,seed=500000,sign=0)
                        assert calls==LABELS
                        added.append((normalized*ds.norm["std"]+ds.norm["mean"])[0])
                    added = torch.stack(added)
                finally:
                    hook.remove()
                write_tensor(extra,added.cpu())
                dump(manifest,dict(recipe=recipe,sha256=sha(extra)))
            prior_p[order] = added
        assert torch.isfinite(prior_p).all() and torch.isfinite(prior_x).all()
        proof = dict(current_cache_sha256=meta["file_sha256"],matched_P_sha256=tensor_sha(p[indices]),
                     previous_P_sha256=tensor_sha(prior_p),previous_X_sha256=tensor_sha(prior_x),
                     reused_previous=cached,new_previous=len(order),added_recipe=recipe,
                     generated_cache_sha256=sha(extra) if order else None,
                     test_freeze_sha256=sha(self.out/"FROZEN.json") if name=="test" else None)
        dump(self.out/f"native_{name}.json",proof)
        print("TRIPLETS",name,len(cases),"previous reuse/new",cached,len(order),flush=True)
        return SimpleNamespace(cases=cases,x=x[indices],p=p[indices],y=y[indices],previous_x=prior_x,
                               previous_p=prior_p)
