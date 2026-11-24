"""Pinned RealPDE windows and one physical, zero-noise forecast per window."""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import h5py
import numpy as np
import torch

from research.corrector.run import dump, sha
from research.corrector_window_study.data import RAW, ROOT, WORKSPACE, read_json, save_tensor, tensor_sha
from research.residual_dynamics.run import load_backbone

OUT = ROOT / "results/corrector_simple_baseline"
CACHE = ROOT / ".cache/corrector_simple_baseline"
BASE = ROOT / "results/corrector_baseline_recovery"
WINDOW = ROOT / "results/corrector_window_study/cache_manifest.json"
DYNAMICS = ROOT / "results/residual_dynamics/protocol.json"
LABELS = (99,66,33,0)
SPLITS = ("train","development","test")


def keys(cases):
    return [(c["file"],c["start"]) for c in cases]


class Data:
    def __init__(self, raw, cache, out):
        self.raw, self.cache, self.out = raw, cache, out
        cache.mkdir(parents=True,exist_ok=True)
        self.manifest = read_json(WINDOW)
        self.previous = read_json(BASE/"legacy_replay.json")
        protocol = read_json(DYNAMICS)
        source = read_json(BASE/"sources.json")
        assert source["expansion_sha256"]==sha(BASE/"expansion.json")
        assert sha(DYNAMICS)==self.manifest["protocol_sha256"]
        assert source["window_cache_manifest_sha256"]==sha(WINDOW)
        self.expansion = read_json(BASE/"expansion.json")
        assert self.expansion["legacy_manifest_sha256"]==sha(BASE/"legacy_windows_819.json")
        self.records = protocol["records"]
        self.cases = {"train":self.expansion["cases"]}
        for name in SPLITS[1:]:
            self.cases[name] = [
                {k:c[k] for k in ("file","start","re","history","target")}
                for c in self.manifest["splits"][name]["cases"] if c["start"] in (400,420)]
        files = []
        for name,n,nt in zip(SPLITS,(822,24,24),(46,12,12)):
            identity = set(self.manifest["splits"][name]["files"])
            own = {r["file"]:r for r in self.records[name]}
            assert len(identity)==nt and identity==set(own)
            assert len(set(keys(self.cases[name])))==len(self.cases[name])==n
            assert {r["file"] for r in self.cases[name]}==identity
            files.append(identity)
            for case in self.cases[name]:
                f,s = case["file"],case["start"]
                assert case["history"]==[s,s+20] and case["target"]==[s+20,s+40]
                assert own[f]["frames"]>=s+40 and own[f]["re"]==case["re"]
        assert sum(map(len,files))==len(set.union(*files))
        self.backbone,self.norm,self.identity = load_backbone(SimpleNamespace(workspace=WORKSPACE))
        assert self.identity==self.manifest["native_checkpoint"]==self.previous["backbone"]
        assert not any(v.requires_grad for v in self.backbone.parameters())
        assert {k:v.flatten().tolist() for k,v in self.norm.items()}==self.previous["normalization"]

    def fields(self,name):
        grouped = defaultdict(list)
        for i,case in enumerate(self.cases[name]):
            grouped[case["file"]].append((i,case["start"]))
        records = {r["file"]:r for r in self.records[name]}
        x = torch.empty((len(self.cases[name]),2,20,32,64))
        y = torch.empty_like(x)
        for file,windows in grouped.items():
            start = min(s for _,s in windows)
            stop = max(s for _,s in windows)+40
            with h5py.File(self.raw/file,"r") as handle:
                assert all(handle[c].shape[0]==records[file]["frames"] for c in ("u","v"))
                fields = np.stack([handle[c][start:stop,::2,::2] for c in ("u","v")]).astype("float32")
            fields = torch.from_numpy(fields)
            for i,s in windows:
                offset = s-start
                x[i] = fields[:,offset:offset+20]
                y[i] = fields[:,offset+20:offset+40]
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        by_start = defaultdict(list)
        for i,r in enumerate(self.cases[name]):
            by_start[r["start"]].append(i)
        if name=="train":
            for row in self.expansion["batches"]:
                ix = by_start[row["start"]]
                assert [self.cases[name][i]["file"] for i in ix]==row["files"]
                assert tensor_sha(y[ix])==row["targets_sha256"]
                if row["files"]==row["history_order"]:
                    assert tensor_sha(x[ix])==row["history_sha256"]
        else:
            for start,ix in by_start.items():
                reference = self.manifest["splits"][name]["native"][str(start)]
                assert [self.cases[name][i]["file"] for i in ix]==reference["identity"]["files"]
                for k,v in (("history",x),("truth",y)):
                    assert tensor_sha(v[ix])==reference[k+"_sha256"]
        return x.cuda(),y.cuda()

    def old_cache(self,name):
        old = {"train":"train","development":"late"}.get(name)
        if not old:
            return None,None,{}
        paths = [Path(f"runs/{exp}_20261117/{old}.pt") for exp in ("error_structure","active_observation")]
        fingerprints = {str(p):self.previous["source_sha256"][str(p)] for p in paths}
        archives = []
        for relative in paths:
            path = WORKSPACE/relative
            if not path.exists():
                return None,None,{}
            assert sha(path)==fingerprints[str(relative)], path
            # These historical containers include the Python window metadata.
            archives.append(torch.load(path,map_location="cpu",weights_only=False))
        xy, forecast = archives
        assert xy["meta"]==forecast["meta"] and len(keys(xy["meta"]))==len(set(keys(xy["meta"])))
        return xy,forecast,fingerprints

    @torch.inference_mode()
    def split(self,name):
        frozen = None
        if name=="test":
            path = self.out/"FROZEN.json"
            assert path.is_file(), "Test access requires frozen development selection"
            frozen = sha(path)
        x,y = self.fields(name)
        mean,std = self.norm["mean"],self.norm["std"]
        history = (x-mean)/std
        identity = dict(protocol="single zero-initial-noise path, batch 1",denoising_labels=list(LABELS),
                        raw=str(self.raw),backbone=self.identity,normalization={k:tensor_sha(v) for k,v in self.norm.items()},
                        cases=self.cases[name],x_sha256=tensor_sha(x),y_sha256=tensor_sha(y))
        meta_path, cache_path = self.cache/f"{name}.json",self.cache/f"{name}.pt"
        if meta_path.is_file() and cache_path.is_file():
            meta = read_json(meta_path)
            assert meta["identity"]==identity and sha(cache_path)==meta["file_sha256"]
            p = torch.load(cache_path,map_location=x.device,weights_only=True)
        else:
            started = perf_counter()
            old,legacy,hashes = self.old_cache(name)
            lookup = {key:i for i,key in enumerate(keys(old["meta"]))} if old is not None else {}
            p = torch.empty_like(x)
            reviewed = set()
            calls, verified, new, reused = [], [], [], []
            def record(_,args):
                assert len(args[0])==1 and len(args[1])==1
                if not calls:
                    assert not torch.count_nonzero(args[0])
                calls.append(int(args[1].item()))
            hook = self.backbone.net.register_forward_pre_hook(record)
            def forecast(h):
                calls.clear()
                result = self.backbone.rollout(h,seed=500000,sign=0)
                assert tuple(calls)==LABELS
                return result[0]
            try:
                for i,key in enumerate(keys(self.cases[name])):
                    j = lookup.get(key)
                    normalized = history[i:i+1]
                    if j is None:
                        point = forecast(normalized)
                        new.append(list(key))
                    else:
                        assert torch.equal(normalized[0].cpu(),old["x"][j])
                        assert torch.equal(((y[i:i+1]-mean)/std)[0].cpu(),old["y"][j])
                        point = legacy["p"][j].cuda()
                        reused.append(list(key))
                        if key[1] not in reviewed:
                            assert torch.equal(point,forecast(normalized)), key
                            reviewed.add(key[1])
                            verified.append(list(key))
                    p[i] = point*std[0]+mean[0]
            finally:
                hook.remove()
            save_tensor(cache_path,p)
            meta = dict(identity=identity,legacy_source_hashes=hashes,reused_windows=reused,
                        replay_verified=verified,generated_windows=new,passes_per_generated_window=4,
                        file_sha256=sha(cache_path),P_sha256=tensor_sha(p),cache_bytes=cache_path.stat().st_size,
                        first_preparation_seconds=perf_counter()-started)
            dump(meta_path,meta)
        assert p.shape==x.shape and torch.isfinite(p).all() and tensor_sha(p)==meta["P_sha256"]
        dump(self.out/f"native_{name}.json",dict(meta,test_freeze_sha256=frozen))
        print("NATIVE",name,len(p),"reused",len(meta["reused_windows"]),
              "new",len(meta["generated_windows"]),"replayed",len(meta["replay_verified"]),flush=True)
        return SimpleNamespace(x=x,p=p,y=y,cases=self.cases[name])
