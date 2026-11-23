"""Train-only calibration, development mechanisms, then one frozen evaluation."""
import argparse
import shlex
import sys
import time
from pathlib import Path

import torch

from research.corrector.run import dump, sha, table
from research.corrector_window_study.data import LOCAL, RESULTS, ROOT, StudyData, read_json
from research.corrector_window_study.run import batches, load_checkpoint, physical_forecast
from research.corrector_window_study.verify import verify as window_verify

from .audit import confirmation, sources
from .calibration import STATIC, TrainingCRPS, apply_scale, scales, static_basis, synthetic_check
from .conditional import METHODS, basis as conditional_basis, features, knots, release_features
from .mechanism import centers, compare_marginals, evaluate, ordered, rank_interventions, tie_rates
from .summary import aggregate, comparisons, reproduce


OUT = ROOT / "results/corrector_distribution_study"
PRIMARY = {"S*-native":"crps", "A-S*":"crps", "C-S*":"crps", "C-C_native_rank":"energy"}


def setup():
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_bf16_supported()
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    print("COMMAND", shlex.join([sys.executable, *sys.argv]), flush=True)


def load_inputs():
    cfg = read_json(RESULTS / "config.json")
    selected = [r for r in read_json(RESULTS / "selected.json") if r["arm"] in {"A","C"}]
    assert {(r["arm"],r["seed"]) for r in selected} == {(a,s) for a in "AC" for s in (42,43)}
    manifest = read_json(RESULTS / "cache_manifest.json")
    assert cfg["cache_manifest_sha256"] == sha(RESULTS / "cache_manifest.json")
    normalization = tuple(torch.tensor(cfg["normalization"][k],device="cuda").view(1,2,1,1,1)
                          for k in ("mean","scale"))
    return manifest, selected, normalization


def append(rows, method, seed, data, idx, values):
    for position, i in enumerate(idx):
        case = data.cases[i]
        rows.append(dict(method=method,seed=seed,trajectory=case["file"],re=case["re"],start=case["start"],
                         **{k: v[position] for k,v in values.items()}))


@torch.inference_mode()
def core(data, selected, normalization, fitted, conditional=None):
    rows,ties = [],[]
    maximal_marginal_error = 0.
    scale = normalization[1]
    models = [(r["arm"],r["seed"],load_checkpoint(data,LOCAL,r)) for r in selected]
    basis = {name:static_basis(name, len(data.cases), scale.device) for name in fitted}
    for idx in batches(data):
        release = data.release(idx)
        p = release[2]
        observed = data.target(idx)
        native_scores = evaluate(p,observed,scale)
        append(rows,"native","",data,idx,native_scores)
        p_sort = ordered(p)
        for name, cfg in fitted.items():
            alpha = scales(basis[name][idx],cfg["coefficients"])
            assert (alpha >= .25-1e-6).all() and (alpha <= 4+1e-6).all()
            adjusted = apply_scale(p,alpha)
            max_mean_error = float((adjusted.mean(1)-p.mean(1)).abs().max())
            assert max_mean_error < 3e-7
            assert (adjusted.gather(1,p_sort.indices).diff(dim=1) >= -1e-7).all()
            append(rows,name,"",data,idx,evaluate(adjusted,observed,scale))
        if conditional is not None:
            scale_conditions(rows,data,idx,release,observed,scale,conditional)
        for arm,seed, model in models:
            q = physical_forecast(model,release,arm,normalization)
            ans = evaluate(q,observed,scale)
            append(rows,arm,seed,data,idx,ans)
            mu, center = centers(p,q)
            append(rows,arm+"_mean",seed,data,idx,evaluate(mu,observed,scale))
            append(rows,arm+"_center",seed,data,idx,evaluate(center,observed,scale))
            q_sort = ordered(q)
            qn,pq = rank_interventions(p_sort,q_sort)
            permuted = evaluate(qn,observed,scale)
            ref = evaluate(pq,observed,scale)
            maximal_marginal_error = max(maximal_marginal_error,compare_marginals(ans,permuted),
                                        compare_marginals(native_scores,ref))
            append(rows,arm+"_native_rank",seed,data,idx,permuted)
            append(rows,"P_"+arm+"_rank",seed,data,idx,ref)
            native_ties, q_ties = tie_rates(p_sort.values),tie_rates(q_sort.values)
            for j,i in enumerate(idx):
                case = data.cases[i]
                ties.append(dict(arm=arm,seed=seed,trajectory=case["file"],start=case["start"],
                                 native_adjacent_ties=native_ties[j],corrector_adjacent_ties=q_ties[j]))
    return rows,ties,maximal_marginal_error


def scale_conditions(rows,data,idx,release,target,scale,config):
    native = release[2]
    state = release_features(release,scale)
    for method, fitted in config["models"].items():
        alpha = scales(conditional_basis(method,state,config["feature_limits"]),fitted["coefficients"]).float()
        adjusted = apply_scale(native,alpha)
        assert (alpha >= .25-1e-6).all() and (alpha <= 4+1e-6).all()
        assert (adjusted.mean(1)-native.mean(1)).abs().max() < 3e-7
        append(rows,method,"",data,idx,evaluate(adjusted,target,scale))


def exploration(out,manifest,norm):
    assert (out/"DEVELOPMENT_SELECTION.json").exists() and not (out/"FROZEN.json").exists()
    assert not (out/"EXPLORATION_PARAMETERS.json").exists()
    assert (out/"EXPLORATION_LOG.md").exists()
    train = StudyData(LOCAL,manifest,"train","cuda")
    dev = StudyData(LOCAL,manifest,"development","cuda")
    scale = norm[1]
    observed_features = features(train,scale)
    limits = knots(observed_features)
    case = train.cases[0]
    physical_target = train.labels[0].clone()
    position = case["start"] - 360
    before = release_features(train.release([0]),scale)
    raw_target = train.observed[case["index"],:,position:position+20].clone()
    train.labels[0] = physical_target + .25
    train.observed[case["index"],:,position:position+20] = raw_target + .25
    assert torch.equal(before,release_features(train.release([0]),scale))
    train.labels[0] = physical_target
    train.observed[case["index"],:,position:position+20] = raw_target
    objective = TrainingCRPS(train,scale)
    fitted = {}
    for method in METHODS:
        fitted[method] = objective.fit(conditional_basis(method,observed_features,limits))
        print("FIT",method,fitted[method],flush=True)
    config = dict(feature_limits=limits,models=fitted,target_mutation_bitwise_equal=True)
    dump(out/"EXPLORATION_PARAMETERS.json",config)
    del objective,train
    new_rows = []
    for idx in batches(dev):
        scale_conditions(new_rows,dev,idx,dev.release(idx),dev.target(idx),scale,config)
    table(out/"development_exploration_cases.csv",new_rows)
    with (out/"development_cases.csv").open() as stream:
        import csv
        old_rows = list(csv.DictReader(stream))
    for row in old_rows:
        row["start"] = int(row["start"])
        if row["seed"]:
            row["seed"] = int(row["seed"])
    simple = read_json(out/"DEVELOPMENT_SELECTION.json")["simple"]
    _,traj,scores = aggregate(old_rows + new_rows)
    table(out/"development_exploration_scores.csv",scores)
    table(out/"development_exploration_contrasts.csv",comparisons(traj,simple,PRIMARY))


def write_split(out, name, rows, ties, simple):
    table(out/f"{name}_cases.csv", rows)
    table(out/f"{name}_ties.csv", ties)
    seeds, traj, scores = aggregate(rows)
    table(out/f"{name}_seed_trajectory.csv", seeds)
    table(out/f"{name}_trajectory.csv", traj)
    table(out/f"{name}_scores.csv", scores)
    table(out/f"{name}_contrasts.csv", comparisons(traj, simple, PRIMARY))
    return scores


def development(out, manifest, chosen, norm):
    assert not (out/"DEVELOPMENT_SELECTION.json").exists(), "development already completed"
    independent = confirmation(manifest)
    dump(out/"CONFIRMATION_INVENTORY.json",independent)
    train = StudyData(LOCAL,manifest,"train","cuda")
    dev = StudyData(LOCAL,manifest,"development","cuda")
    checked = window_verify(train,manifest,norm)
    checked.update(synthetic_check())
    calibration = out/"CALIBRATION.json"
    if calibration.exists():
        fitted = read_json(calibration)
        assert list(fitted) == list(STATIC[1:]) and all(p["converged"] for p in fitted.values())
        print("RESUME completed static fits without refitting",flush=True)
    else:
        fitted = {}
        timer = time.monotonic()
        objective = TrainingCRPS(train,norm[1])
        for family in STATIC[1:]:
            fitted[family] = objective.fit(static_basis(family,len(train.cases),"cuda"))
            print("FIT",family,fitted[family],flush=True)
        dump(calibration,fitted)
        del objective
        print("FITTING_SECONDS",time.monotonic()-timer,flush=True)
    del train
    rows,ties,error = core(dev,chosen,norm,fitted)
    checked["development_reproduction_maxabs"] = reproduce(rows, chosen, RESULTS, "development")
    _,_, scores = aggregate(rows)
    absolute = {r["method"]:r["crps"] for r in scores if r["scope"]=="all"}
    simple = min(("native",*STATIC[1:]),key=absolute.__getitem__)
    write_split(out,"development",rows,ties,simple)
    dump(out/"DEVELOPMENT_SELECTION.json",dict(simple=simple,development_crps=absolute))
    dump(out/"PREFLIGHT.json",dict(**checked,development_marginal_score_maxabs=error))
    print("DEVELOPMENT",simple,absolute,flush=True)


def final(out,manifest,chosen,norm):
    assert not (out/"FROZEN.json").exists() and not (out/"test_cases.csv").exists()
    assert (out/"PREFLIGHT.json").exists()
    simple = read_json(out/"DEVELOPMENT_SELECTION.json")["simple"]
    parameters = read_json(out/"CALIBRATION.json")
    inventory = read_json(out/"CONFIRMATION_INVENTORY.json")
    assert inventory["confirmation"] == []
    fixed = {p.name:sha(p) for p in (out/"CALIBRATION.json",out/"DEVELOPMENT_SELECTION.json",
                                     out/"EXPLORATION_LOG.md",out/"EXPLORATION_PARAMETERS.json",
                                     out/"CONFIRMATION_INVENTORY.json",out/"SOURCE_HASHES.json")}
    code = {p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.suffix in (".py",".md")}
    conditional = read_json(out/"EXPLORATION_PARAMETERS.json")
    dump(out/"FROZEN.json",dict(static=parameters, conditional=conditional, selected_simple=simple, checkpoints=chosen,
        primary=PRIMARY, new_confirmation=inventory["confirmation"], artifacts=fixed, code=code,
        windows=[400,420,440,460], members=manifest["seeds"],
        test_status="existing trajectories; exploratory",
        runtime=dict(torch=torch.__version__,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                     matmul_tf32=torch.backends.cuda.matmul.allow_tf32)))
    test = StudyData(LOCAL,manifest,"test","cuda")
    rows,ties,error = core(test,chosen,norm,parameters,conditional)
    write_split(out,"test",rows,ties,simple)
    replay_error = reproduce(rows,chosen,RESULTS,"test")
    dump(out/"TEST_VERIFICATION.json",dict(marginal_score_maxabs=error,reproduction_maxabs=replay_error))
    print("EXPLORATORY TEST COMPLETE",len(rows),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("development","exploration","final"),required=True)
    parser.add_argument("--out",type=Path,default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    setup()
    torch.cuda.reset_peak_memory_stats()
    manifest, selected, norm = load_inputs()
    before = sources(RESULTS,LOCAL,manifest,selected)
    try:
        if args.phase == "development":
            hashes = args.out/"SOURCE_HASHES.json"
            if hashes.exists():
                assert read_json(hashes) == before
            else:
                dump(hashes,before)
            development(args.out,manifest,selected,norm)
        else:
            assert before == read_json(args.out/"SOURCE_HASHES.json")
            if args.phase == "exploration":
                exploration(args.out,manifest,norm)
            else:
                final(args.out,manifest,selected,norm)
        assert before == sources(RESULTS,LOCAL,manifest,selected)
        dump(args.out/f"{args.phase.upper()}_RUN.json",dict(passed=True,torch=torch.__version__,
            arguments=sys.argv,wall_seconds=time.monotonic()-started,new_neural_training_runs=0,
            reexecuted_native_paths=0,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            workspace_cache_bytes=sum(p.stat().st_size for p in (ROOT/".cache/corrector_distribution_study").rglob("*") if p.is_file())))
    except Exception as error:
        dump(args.out/f"{args.phase.upper()}_STOP.json",dict(error=repr(error),wall_seconds=time.monotonic()-started))
        raise


if __name__ == "__main__":
    main()
