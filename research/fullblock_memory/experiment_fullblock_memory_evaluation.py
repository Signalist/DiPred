"""Frozen full-block memory heads on the original complete96-cell evaluation."""

import json
import time

import torch

from clean_reliability_common import ROOT, configure, digest, dump
from experiment_fullblock_memory_risk import OUT as TRAIN, PREDICTOR, load
from temporal_mean_memory_control import OUT as MEAN, Provider
from pilot_fullblock_memory_risk import OUT as PILOT
from experiment_reliability_transport_e12 import normalization
from experiment_continue_rich_uncertainty import MODES, controller
from replay_conditional_risk import replay


OUT = ROOT / "runs/fullblock_memory_evaluation_20261109"
NAMES = ("frozen_rich", "none", "last", "mean", "full")


def main():
    configure()
    assert (TRAIN / "complete.json").exists() and (MEAN / "complete.json").exists()
    assert json.loads((PILOT / "gate.json").read_text())["proceed_full_evaluation"]
    assert json.loads((TRAIN / "training_independent_audit.json").read_text())["passed"]
    OUT.mkdir(exist_ok=False)
    paths = [PREDICTOR / "augmented_diffusion/model.pt", TRAIN / "calibration.pt", MEAN / "calibration.pt", MEAN / "mean.pt"]
    paths += [TRAIN / (mode + ".pt") for mode in ("none", "last", "full")]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    dump(OUT / "protocol.json", dict(time=time.time(), source_sha256=digest(ROOT / "experiment_fullblock_memory_evaluation.py"),
        predictor_sha256=digest(paths[0]), artifacts=hashes, heads=NAMES, modes=MODES,
        source_training_protocol_sha256=digest(TRAIN / "protocol.json"),
        temporal_mean_control_protocol_sha256=digest(MEAN / "protocol.json"),
        information="Frozen same48340parameter controls:none,lastsnapshot,encodedtemporalmean,full20residualframes. Warm core,819pairs,2000updates,seed42,initialization/minibatches/availabilitymasks identical. Reference45060core not a matched memory control.",
        gate="Originaltrain-ID pilot last/full passed jointScore/top10/coverage versusnone before development/test field evaluation. Temporalmean added as stronger averaging control before evaluation, not used to select a test-regime policy.",
        invariant="Original4calls99,66,33,0,joint20to20,commonpointfrozen. Memory=currentknownhistory minus actual savedpreviouspoint,never currentfuturelabel or regeneratedstochasticsample. Each whole20target follows allforecastissues.",
        comparisons="Keep all8metrics/96cells,tolerance1e-6,allstrongreferences and failures. Rawrisk rankings retained alongside actual issuedhalfwidth. No extra amplitude forecaster/adapters trained for these heads.",
        limits="Same historically reused evaluation trajectories, not independentconfirmation. Identical allocatedparameters, intentionally different input temporalrank; no temporal transport or physicaluncertainty claim from weights alone."))
    initial = load(TRAIN / "calibration.pt")
    initial["mean"] = load(MEAN / "calibration.pt")["mean"]
    torch.save({name: {mode: {key: value.cpu() for key,value in state.items()} for mode,state in modes.items()}
                for name,modes in initial.items()}, OUT / "calibration.pt")
    replay(OUT, Provider(normalization()), NAMES, MODES, controller, initial)
    assert all(digest(ROOT / path) == sha for path,sha in hashes.items())
    dump(OUT / "complete.json", dict(time=time.time(), all_neural_artifacts_unchanged=True))
    print("FULLBLOCK_MEMORY_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
