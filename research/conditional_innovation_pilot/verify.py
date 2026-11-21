"""Read-only evidence audit. Does not fit models or regenerate predictions."""
from collections import defaultdict
from core import *


def main():
    frozen=json.loads((OUT/"FROZEN.json").read_text())
    gate=json.loads((OUT/"selection_A.json").read_text())
    assert frozen["stage_A"]==gate and frozen["config_sha256"]==CONFIG_HASH
    for name,sha in frozen["weights"].items():
        assert digest(OUT/"weights"/name)==sha
        ck=torch.load(OUT/"weights"/name,map_location="cpu",weights_only=False)
        if name.startswith(("R1_","D1_")):
            assert ck["coefficient"].shape==(88,40)
            assert torch.isfinite(ck["coefficient"]).all() and (ck["xstd"]>0).all()
    for name,sha in CFG["dependencies"].items():assert digest(ROOT/name)==sha
    for name,sha in CFG["old_heads"].items():
        assert digest(ROOT/"results/residual_dynamics_phase3/weights"/name)==sha
    for line in (PRIOR/"SHA256SUMS").read_text().splitlines():
        sha,name=line.split("  ",1)
        assert digest(ROOT/name)==sha,name
    for left,right in [("train","development"),("train","test"),("development","test")]:
        for key in ("file","group"):
            assert {r[key] for r in RECORDS[left]}.isdisjoint({r[key] for r in RECORDS[right]})
    base=torch.load(OUT/"weights/baseline.pt",map_location="cpu",weights_only=False)
    original=torch.load(PRIOR/"training_innovation_mean.pt",map_location="cpu",weights_only=False)
    mean_error=float((base["source_mean"][0,1:]-original["mean"]).abs().max())
    assert mean_error<1e-11
    for reg in CFG["ridge"]:
        source_ck=torch.load(OUT/"weights"/f"R1_{reg}.pt",map_location="cpu",weights_only=False)
        direct_ck=torch.load(OUT/"weights"/f"D1_{reg}.pt",map_location="cpu",weights_only=False)
        assert torch.allclose(source_ck["coefficient"][:,:2],direct_ck["coefficient"][:,:2],rtol=0,atol=1e-12)
        assert torch.equal(source_ck["mean"][:,:1],direct_ck["mean"][:,:1])
    for filename,sha in json.loads((PRIOR/"data_hashes.json").read_text()).items():
        assert digest(DATA/filename)==sha
    scores=read_csv(OUT/"scores.csv")
    rows=read_csv(OUT/"development_scores_A.csv")
    ref=np.mean([float(r["r_smse"]) for r in rows if r["method"]=="R0"])
    value=np.mean([float(r["r_smse"]) for r in rows if r["method"]=="R1_ridge"])
    assert abs(gate["r_gain"]-(1-value/ref))<1e-12
    measured_gate=(gate["r_gain"]>=.01 and min(gate["window_gains"].values())>0 and
                   min(gate["channel_gains"].values())>=0 and sum(v>0 for v in gate["Re_gains"].values())>=2)
    assert measured_gate==gate["gate"]==frozen["stage_B"]["executed"]
    methods=[("native",0),("zero",0),("R0",0),("R1_ridge",0),("D1_ridge",0),("R1",0),("D1",0)]
    methods += [("phase3",seed) for seed in CFG["seeds"]]
    if gate["gate"]:methods += [(kind,seed) for kind in ("D","T_local","T_diag") for seed in CFG["seeds"]]
    expected={(split,r["file"],str(start),method,str(seed))
              for split,starts in [("test",CFG["starts"]),("extension",CFG["extension"])]
              for start in starts for r in RECORDS["test"] for method,seed in methods}
    fields=["split","trajectory","start","method","seed"]
    assert len(scores)==len(expected)
    assert {tuple(r[k] for k in fields) for r in scores}==expected
    metrics=["crps","energy","mse","smse","r_mse","r_smse","brier"]
    groups=defaultdict(list)
    for r in scores:groups[r["split"],r["trajectory"],r["method"]].append(r)
    per_trajectory={key:{m:np.mean([float(r[m]) for r in rr]) for m in metrics} for key,rr in groups.items()}
    for r in read_csv(OUT/"paired.csv"):
        left,right=r["contrast"].split(" minus ")
        metric=r["metric"];split=r["split"]
        delta=[]
        for trajectory in sorted({t for s,t,m in per_trajectory if s==split and m==left}):
            def get(method):
                if method=="zero" or method in ("R1","D1") and gate["selected"][method]["deployment"]=="native":
                    method="native"
                return per_trajectory[split,trajectory,method][metric]
            delta.append(get(left)-get(right))
        result=bootstrap(delta)
        assert abs(result["mean_delta"]-float(r["mean_delta"]))<1e-12
        assert abs(result["ci95"][0]-float(r["ci95_low"]))<1e-12
        assert abs(result["ci95"][1]-float(r["ci95_high"]))<1e-12
        assert result["improved"]==int(r["improved"])
    for name,sha in json.loads((OUT/"cache_hashes.json").read_text()).items():assert digest(Path(name))==sha
    print("PASS weights/config/cache hashes and all previous followup artifact hashes")
    print("PASS original trajectory/Re isolation, source gate, exact method/window coverage, paired bootstrap")
    print("training fixed innovation mean versus previous cache max error",mean_error)
    print("selected",gate["selected"],"neural training count",frozen["stage_B"]["neural_runs"])


if __name__=="__main__":
    main()
