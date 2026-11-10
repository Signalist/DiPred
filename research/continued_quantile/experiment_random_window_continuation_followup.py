"""Complete follow-up after the useful matched quantile control improves.

Proper-density gates remain failed. The quantile control improves all eight
train-pilot macro metrics versus its frozen source under one fixed controller,
motivating this separate exploratory evaluation with all density controls kept.
"""

import json
import time

import torch

from clean_reliability_common import ROOT,configure,dump,digest
from experiment_distribution_error_risk import OUT as TRAIN,NAMES,OBJECTIVES,Provider,load
from experiment_reliability_transport_e12 import normalization,PREDICTOR
from experiment_continue_rich_uncertainty import MODES,controller
from replay_conditional_risk import replay


OUT = ROOT/"runs/random_window_continuation_followup_20261110"
METRICS = {"standardized_interval_score":-1,"standardized_width":-1,"relative_aurc":-1,
           "top10_energy":1,"per_frame_top10_energy":1,"worst_frame_coverage":1,
           "worst_patch_coverage":1,"coverage_error":-1}


def main():
    configure()
    assert json.loads((TRAIN/"independent_training_pilot_audit.json").read_text())["passed"]
    gate = json.loads((TRAIN/"gate.json").read_text())
    assert gate["proceed_full_evaluation"] is False
    evidence = {}
    for mode,result in gate["results"].items():
        new,old = result["values"]["density_quantile"],result["values"]["density_reference"]
        gains = {metric:direction*(new[metric]-old[metric]) for metric,direction in METRICS.items()}
        evidence[mode] = dict(gains=gains,all_eight_strictly_improved=all(value>1e-6 for value in gains.values()))
    assert any(row["all_eight_strictly_improved"] for row in evidence.values())
    OUT.mkdir(exist_ok=False)
    paths = [PREDICTOR/"augmented_diffusion/model.pt",TRAIN/"calibration.pt"]
    paths += [TRAIN/(name+".pt") for name in OBJECTIVES]
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in paths}
    dump(OUT/"protocol.json",dict(time=time.time(),source_sha256=digest(ROOT/"experiment_random_window_continuation_followup.py"),
        predictor_sha256=digest(paths[0]),artifacts=hashes,names=NAMES,modes=MODES,
        original_density_gate_passed=False,original_failed_gate_sha256=digest(TRAIN/"gate.json"),
        original_train_only_evidence=evidence,
        reason="GaussianandStudent-tfulldensityobjectivesfailversusmatchedquantile. Quantilecontrolitself improvesall8pilotmacrocriteria versusfrozencurrent under memory_dynamic. Separate exploratoryfollowup retainsall4heads/2controllers; originaldensityfailureisnotrewritten.",
        fitting="Onlyoriginal819trainpairs200:600,seed42,2000additionalriskheadupdates,48340parameters. Samefixednativefourcallpointforecaster,statsandgeometry. No development/testfitting or calibration-labelNNtraining.",
        scope="Alloriginal8metrics,96development/testRe-channelcells,1e-6tolerance,all47previousreferencepairs,allmethods/failures retained. Same reused evalidentities; pilot improvement doesnotestablishfull-line success or independentconfirmation.",
        invariant="Original99,66,33,0,joint20to20,actualsavedpreviousPerror,truthonlyaftercomplete20issue. No extrasampling,pointcorrection,folds,distillation or newtrainingseeds."))
    initial = load(TRAIN/"calibration.pt")
    torch.save({name:{mode:{key:value.cpu() for key,value in state.items()} for mode,state in modes.items()}
                for name,modes in initial.items()},OUT/"calibration.pt")
    replay(OUT,Provider(normalization()),NAMES,MODES,controller,initial)
    assert all(digest(ROOT/name)==sha for name,sha in hashes.items())
    dump(OUT/"complete.json",dict(time=time.time(),all_neural_artifacts_unchanged=True))
    print("RANDOM_WINDOW_CONTINUATION_FOLLOWUP_COMPLETE",flush=True)


if __name__=="__main__":
    main()
