"""Reproduce B/E, perform staged search, freeze, and score."""
import argparse

from reliability.backbone import configure
from research.corrector.run import dump,sha
from research.corrector_residual_feedback.data import OUT as LAST
from research.corrector_window_study.data import tensor_sha

from .data import Study,OUT,CACHE,FIGURES,RAW,BRANCHES
from .fit import Runner,STEPS,START


def study_setup(out,cache,raw):
    study = Study(out,cache,raw)
    train,_,z_train = study.get("train")
    dev,_,z_dev = study.get("development")
    source = dict(previous_commit="045045dcfd8a4bb89a66067324ed6a4c4f349b44",
                  prior_freeze_sha256=sha(LAST/"FROZEN.json"),protocol_sha256=sha(out/"PROTOCOL.md"),
                  input_identity=study.guards,prior_test_manifest_blob=study.prior["prior_test_manifest_blob"],
                  historical_references=study.feedback_meta,backbone=study.triplets.source.identity,
                  schedule_sha256=tensor_sha(study.schedule))
    config = dict(seed=42,scale=study.scale,updates=6000,shared_step=START,lr=.001,weight_decay=1e-5,
                  lr_end=1e-6,batch=8,clip=1,layout="FP32 contiguous NCDHW",steps=STEPS,
                  scale_inputs=study.cfg["normalization"],case=study.cfg["case"],
                  core_budget=60,extension_budget=20,epsilon=1e-12,lambda_max=100.,
                  window_counts=dict(train=len(train.cases),development=len(dev.cases),
                                     test=len(study.triplets.plan["test"]["cases"])))
    dump(out/"sources.json",source)
    dump(out/"config.json",config)
    return Runner(study,train,dev,z_train,z_dev)


def train(out,cache,raw,baseline_only=False):
    runner = study_setup(out,cache,raw)
    replay = {}
    for branch in BRANCHES:
        runner.train_arm(branch,"MSE",{},"replay")
        replay[branch] = runner.verify_mse(branch)
    dump(out/"baseline_replay.json",replay)
    if baseline_only:
        return
    from .search import execute
    execute(runner)


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",default="all",choices=("all","baseline","train","score","audit","report"))
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--cache",type=Path,default=CACHE)
    parser.add_argument("--raw",type=Path,default=RAW)
    parser.add_argument("--figures",type=Path,default=FIGURES)
    args = parser.parse_args()
    configure()
    args.out.mkdir(parents=True,exist_ok=True)
    if not (args.out/"PROTOCOL.md").exists():
        (args.out/"PROTOCOL.md").write_text((OUT/"PROTOCOL.md").read_text())
    if args.action in ("all","baseline","train") and not (args.out/"FROZEN.json").exists():
        train(args.out,args.cache,args.raw,baseline_only=args.action=="baseline")
    if args.action in ("all","score"):
        from .evaluate import score
        from .plot import plot
        score(Study(args.out,args.cache,args.raw))
        plot(args.out,args.figures)
    if args.action in ("all","audit"):
        from .audit import audit
        audit(Study(args.out,args.cache,args.raw),args.figures)
    if args.action in ("all","report"):
        from .report import write_report
        write_report(args.out,args.figures)


if __name__=="__main__":
    main()
