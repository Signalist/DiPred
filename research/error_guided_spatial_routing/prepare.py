"""Build the fixed causal windows and native path cache for this experiment."""
import hashlib, json, os
from pathlib import Path
import sys
import h5py
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WORK = ROOT.parent
DATA = ROOT.parents[1] / "OLD/REALPDE/realpdebench-competition/data/train_real"
SPEC = ROOT / "results/native_restart_consistency/config.json"
DEFAULT_CACHE = ROOT / ".cache/error_guided_spatial_routing"
STARTS = (400, 420, 440, 460, 480)
MEMBERS = tuple(range(500000, 500032))


def sha(x):
    h = hashlib.sha256()
    with open(x, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def raw(files, start):
    out=[]
    for name in files:
        with h5py.File(DATA/name, "r") as f:
            a=np.stack([f[k][start:start+20,::2,::2] for k in ("u","v")]).astype("float32")
        if a.shape!=(2,20,32,64) or not np.isfinite(a).all(): raise ValueError(name)
        out.append(torch.from_numpy(a))
    return torch.stack(out)


def records():
    cfg=json.loads(SPEC.read_text())
    return {k:v for k,v in cfg["records"].items()}


def cache_path(cache, split, start): return cache/f"{split}_{start}_native.pt"


@torch.no_grad()
def ensure_native(cache, split, start, model, norm, recs):
    dest=cache_path(cache, split, start)
    files=[r["file"] for r in recs]
    if dest.exists():
        value=torch.load(dest,map_location="cpu",weights_only=False)
        if value["identity"]=={"split":split,"start":start,"files":files,"members":list(MEMBERS)}:
            return value
    history=raw(files,start).cuda()
    h=(history-norm["mean"])/norm["std"]
    paths=[]
    for member in MEMBERS:
        p=model.rollout(h,seed=member,sign=1)*norm["std"]+norm["mean"]
        paths.append(p.cpu())
    paths=torch.stack(paths,1).contiguous()
    value={"identity":{"split":split,"start":start,"files":files,"members":list(MEMBERS)},
           "paths":paths,"history_sha256":hashlib.sha256(history.cpu().numpy().tobytes()).hexdigest()}
    dest.parent.mkdir(parents=True,exist_ok=True);torch.save(value,dest)
    return value


def build(cache=None):
    from reliability.networks import DynamicsDiffusion20
    cache=Path(cache or os.environ.get("DIPRED_TRUST_CACHE",DEFAULT_CACHE));cache.mkdir(parents=True,exist_ok=True)
    ck=torch.load(ROOT/"reliability/checkpoints/current.pt",map_location="cuda",weights_only=False)
    model=DynamicsDiffusion20(ck["dynamics_scale"]).cuda().eval();model.load_state_dict(ck["predictor"])
    norm={k:v.cuda() for k,v in ck["normalization"].items()}
    rs=records()
    needed={"train":(420,440,460),"development":(420,440,460),"test":(460,480)}
    # Previous starts are included so each risk map is reproducible from an
    # already-issued path and the revealed truth at the next current start.
    for split, starts in needed.items():
        recs=rs[split]
        for start in sorted(set(starts)|{s-20 for s in starts}):
            ensure_native(cache,split,start,model,norm,recs)
    stats=[]
    for split, starts in (("train",(420,440,460)),("development",(420,440,460)),("test",(480,))):
        recs=rs[split];files=[r["file"] for r in recs]
        items=[]
        for start in starts:
            valid=[(i,r) for i,r in enumerate(recs) if int(r.get("frames",10**9))>=start+40]
            valid_files=[r["file"] for _,r in valid]
            current=raw(valid_files,start); truth=raw(valid_files,start+20)
            prev=torch.load(cache_path(cache,split,start-20),map_location="cpu",weights_only=False)["paths"]
            # [trajectory, member, channel, lead, y, x] -> [trajectory,y,x]
            valid_indices=[i for i,r in enumerate(recs) if int(r.get("frames",10**9))>=start+40]
            err=(prev[valid_indices].mean(1)-truth)/norm["std"].cpu()
            R=err.square().mean((1,2))
            items.append((start,current,truth,R))
            if split=="train": stats.append(R)
        if split=="train":
            allr=torch.cat(stats); q=float(torch.quantile(torch.log1p(allr),.99).clamp_min(1e-6))
            (cache/"risk_stats.json").write_text(json.dumps({"log1p_p99":q,"clip":[0.0,1.0]},indent=2)+"\n")
        else: q=json.loads((cache/"risk_stats.json").read_text())["log1p_p99"]
        rows=[]
        for start,current,truth,R in items:
            risk=(torch.log1p(R)/q).clamp(0,1)[:,None]
            for local,(i,r) in enumerate(valid):
                rows.append({"split":split,"trajectory_index":i,"file":r["file"],"Re":r.get("re",r.get("Re",0)),"start":start,
                             "history":current[local],"truth":truth[local],"risk":risk[local]})
        torch.save({"rows":rows,"normalization":{k:v.cpu() for k,v in norm.items()}},cache/f"{split}_windows.pt")
    (cache/"prepare_manifest.json").write_text(json.dumps({"starts":STARTS,"members":MEMBERS,"cache":str(cache),"records":rs},indent=2)+"\n")
    return cache


if __name__=="__main__": build()
