"""Complete exploratory follow-up of an eight-metric train-pilot improvement.

The transport-versus-modulation mechanism gate stays FAILED. This separate
question asks whether the frozen warp candidate improves the current base
under the complete original evaluation, retaining modulation as a strong
control and never claiming a unique transport advantage from the pilot.
"""

import json
import time

import torch

from clean_reliability_common import ROOT, configure, dump, digest
from experiment_dynamic_risk_memory import OUT as TRAIN, NAMES, ARMS, Provider, load, METRICS
from experiment_reliability_transport_e12 import PREDICTOR, normalization
from experiment_continue_rich_uncertainty import MODES, controller
from replay_conditional_risk import replay


OUT = ROOT / "runs/dynamic_risk_memory_followup_20261110"


def main():
    configure()
    assert json.loads((TRAIN / "independent_training_pilot_audit.json").read_text())["passed"]
    gate = json.loads((TRAIN / "gate.json").read_text())
    assert not gate["proceed_full_evaluation"]
    values = json.loads((TRAIN / "pilot_macro.json").read_text())
    evidence = {}
    for mode, methods in values.items():
        gains = {metric: direction * (methods["warp"][metric] - methods["frozen_best"][metric])
                 for metric, direction in METRICS.items()}
        evidence[mode] = dict(gains=gains, all_eight_strictly_improved=all(value > 1e-6 for value in gains.values()))
    assert any(row["all_eight_strictly_improved"] for row in evidence.values())
    OUT.mkdir(exist_ok=False)
    paths = [TRAIN / (name + ".pt") for name in ARMS]
    paths += [TRAIN / "calibration.pt", PREDICTOR / "augmented_diffusion/model.pt"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    dump(OUT / "protocol.json", dict(
        time=time.time(), source_sha256=digest(ROOT / "experiment_dynamic_risk_memory_followup.py"),
        predictor_sha256=digest(paths[-1]), artifacts=hashes, names=NAMES, modes=MODES,
        original_mechanism_gate_passed=False, original_failed_gate_sha256=digest(TRAIN / "gate.json"),
        original_train_only_evidence=evidence,
        reason="Transportdoesnotdominateequalcapacitymodulation,so originalmechanismgate remainsfalse. WarpneverthelessimprovesALL8train-IDpilotmacrocriteria versus currentfrozenbest under BOTHfixedcontrollers. Separate prospectivecompletehistorical evaluation keepsall3heads/2controllers andallstrongreferences. No claimofunique transportbenefit, no test-driven refit or policychoice.",
        fitting="No new fitting. Existing48340riskweights andfour-callpointfrozen; eachcandidate adds10430trainablemoduleparams fitonly819originaltrainpairs/2000updates/seed42. Independent10IDcalibrationlockedbeforepilot/fullreplay.",
        scope="Alloriginal8metrics/96development-testRe-channelcells/tolerance1e-6/all55previousreferencepairs, plusall6new controls. Keepall failures andcold/warmwindows; no macro-only achievement. Reuseddata,not independentconfirmation.",
        invariant="Native99,66,33,0, full20to20,actualsavedpreviousPerror,truthonlyafterwhole20issue. No pointcorrector,folds,distillation,seedsweep,extraDiffusioncalls or32sampleinference.",
    ))
    initial = load(TRAIN / "calibration.pt")
    torch.save({name: {mode: {key: value.cpu() for key, value in state.items()}
                              for mode, state in modes.items()}
                for name, modes in initial.items()}, OUT / "calibration.pt")
    replay(OUT, Provider(normalization()), NAMES, MODES, controller, initial)
    assert all(digest(ROOT / path) == sha for path, sha in hashes.items())
    dump(OUT / "complete.json", dict(time=time.time(), all_neural_artifacts_unchanged=True,
                                     original_mechanism_gate_not_rewritten=True))
    print("DYNAMIC_RISK_MEMORY_FOLLOWUP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
