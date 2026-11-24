"""Read the already-issued single-path cache without generating a new path."""
from hashlib import sha1
from types import SimpleNamespace

import torch

from research.corrector.run import dump, sha
from research.corrector_simple_baseline.data import CACHE, OUT as SIMPLE, RAW, ROOT, Data
from research.corrector_window_study.data import read_json, tensor_sha

OUT = ROOT/"results/corrector_residual_optimization"


def split(data,name):
    if name=="test":
        frozen = read_json(data.out/"FROZEN.json")
        for file,digest in frozen["bindings"].items():
            assert sha(data.out/file)==digest
        auth = sha(data.out/"FROZEN.json")
    else:
        auth = None
    path = SIMPLE/f"native_{name}.json"
    if name=="test":
        manifest_bytes = path.read_bytes()
        blob = sha1(f"blob {len(manifest_bytes)}\0".encode()+manifest_bytes).hexdigest()
        assert blob==frozen["previous_test_manifest_blob"]
    published = read_json(path)
    metadata = read_json(data.cache/f"{name}.json")
    assert {k:v for k,v in published.items() if k!="test_freeze_sha256"}==metadata
    identity = metadata["identity"]
    assert identity["protocol"]=="single zero-initial-noise path, batch 1"
    assert identity["denoising_labels"]==[99,66,33,0] and identity["backbone"]==data.identity
    assert identity["cases"]==data.cases[name]
    assert identity["normalization"]=={k:tensor_sha(v) for k,v in data.norm.items()}
    assert set(map(tuple,metadata["reused_windows"])).isdisjoint(map(tuple,metadata["generated_windows"]))
    assert len(metadata["reused_windows"])+len(metadata["generated_windows"])==len(data.cases[name])
    x,y = data.fields(name)
    assert tensor_sha(x)==identity["x_sha256"] and tensor_sha(y)==identity["y_sha256"]
    path = data.cache/f"{name}.pt"
    assert sha(path)==metadata["file_sha256"]
    p = torch.load(path,map_location=x.device,weights_only=True)
    assert p.shape==x.shape==y.shape and tensor_sha(p)==metadata["P_sha256"]
    dump(data.out/f"native_{name}.json",dict(published,test_freeze_sha256=auth))
    print("VERIFIED SINGLE PATH",name,len(x),metadata["P_sha256"],flush=True)
    return SimpleNamespace(x=x,p=p,y=y,cases=data.cases[name])


def subset(data,indices):
    return SimpleNamespace(x=data.x[indices],p=data.p[indices],y=data.y[indices],
                           cases=[data.cases[i] for i in indices])
