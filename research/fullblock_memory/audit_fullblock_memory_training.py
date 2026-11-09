"""Check previous-forecast lineage, train-only memory RMS and dense projection."""

import hashlib
import json

import numpy as np
import torch

from clean_reliability_common import ROOT, configure, dump, digest, manifest
from experiment_fullblock_memory_risk import OUT, SOURCE, HEAD_SOURCE, Provider, load
from experiment_reliability_transport_e12 import FIT_RECORDS, FIT_STARTS, normalization, history, read_fit, geometry
from experiment_extended_train_error_adapters import LATE_RECORDS, LATE_STARTS
from experiment_direct_error_uncertainty import cached
from infer_structured_fourstep_uncertainty import StructuredFourStepPredictor
from direct_error_uncertainty import forecast_features
from experiment_temporal_error_uncertainty import pack, normalized


@torch.no_grad()
def main():
    configure()
    assert (OUT / "complete.json").exists()
    protocol = json.loads((OUT / "protocol.json").read_text())
    frozen = json.loads((OUT / "heads_frozen.json").read_text())
    assert all(digest(ROOT / path) == sha for path, sha in protocol["frozen_artifacts"].items())
    assert all(digest(OUT / path) == sha for path, sha in frozen["artifacts"].items())
    held = {r["file"] for split, rows in manifest()["files"].items() if split != "train" for r in rows}
    assert not set(protocol["fit_files"]) & held
    assert len(set(frozen["minibatches_and_availability_masks_sha256"].values())) == 1
    for mode in ("none", "last", "full"):
        checkpoint = load(OUT / (mode + ".pt"))
        assert checkpoint["seed"] == 42 and checkpoint["additional_steps"] == 2000 and checkpoint["examples"] == 819
    norm = normalization()
    forecaster = StructuredFourStepPredictor(geometry(FIT_RECORDS))
    prelude = {}
    for start in (180, 380):
        observed = read_fit(start)
        saved = load(OUT / ("previous_fit" + str(start) + ".pt"))
        assert saved["files"] == [r["file"] for r in FIT_RECORDS]
        assert saved["history_sha256"] == hashlib.sha256(observed.cpu().numpy().tobytes()).hexdigest()
        labels = []
        hook = forecaster.model.net.register_forward_pre_hook(lambda module, args: labels.append(int(args[1][0])))
        try:
            point, _ = forecast_features(forecaster.model, (observed - norm["mean"]) / norm["std"])
        finally:
            hook.remove()
        assert labels == [99, 66, 33, 0] and torch.equal(point, saved["prediction"])
        prelude[start] = saved["prediction"]
    total_square, count = np.zeros(2), 0
    index = {r["file"]: i for i, r in enumerate(FIT_RECORDS)}
    for later, starts in ((False, FIT_STARTS), (True, LATE_STARTS)):
        previous = prelude[380][[index[r["file"]] for r in LATE_RECORDS]] if later else prelude[180]
        for start in starts:
            if later:
                saved = load(SOURCE / "later_data" / (str(start) + ".pt"))
                observed, point = saved["observed"], saved["prediction"]
            else:
                observed, point = history("fit", start), cached("fit", start)["prediction"]
            h = (observed - norm["mean"]) / norm["std"]
            memory = h - previous
            array = memory.cpu().numpy().astype(np.float64)
            total_square += (array ** 2).sum((0,2,3,4))
            count += len(memory) * 20 * 32 * 64
            previous = point
    expected_scale = np.maximum(np.sqrt(total_square / count), .01)
    scale = load(OUT / "statistics.pt")["memory_scale"]
    error = float(abs(expected_scale - scale.flatten().cpu().numpy()).max())
    assert error < 1e-6
    events = [json.loads(line) for line in (OUT / "training_events.jsonl").read_text().splitlines()]
    assert len(events) == 36
    for a,b in zip(events[::2],events[1::2]):
        assert a["event"] == "features_ready" and b["event"] == "supervision_selected"
        assert a["start"] == b["start"] and a["later"] == b["later"] and a["time"] <= b["time"]
        assert a["previous_query"] == a["start"] - 20
    provider = Provider(norm)
    point, base, last = pack("fit", 200, norm, geometry(FIT_RECORDS))
    base, last = normalized(base, last, provider.stats)
    h = (history("fit", 200) - norm["mean"]) / norm["std"]
    memory = h - prelude[180]
    projection_checks = []
    for mode, head in provider.heads.items():
        assert sum(p.numel() for p in head.parameters()) == 48340
        raw = memory[:2].cpu().numpy().astype(np.float64)
        if mode == "none":
            raw = raw * 0
        elif mode == "last":
            raw = np.broadcast_to(raw[:,:, -1:], raw.shape)
        encoded = np.log1p(abs(raw) / scale.cpu().numpy()).clip(0,5)
        actual = head.encode_memory(memory[:2])
        encoded_error = float(abs(actual.cpu().numpy() - encoded).max())
        assert encoded_error < 1e-6
        values = actual.reshape(2,40,32,64)
        weight = head.memory_projection.weight[:,:,0,0].cpu().numpy().astype(np.float64)
        bias = head.memory_projection.bias.cpu().numpy().astype(np.float64)
        projected = np.einsum('oi,bihw->bohw', weight, values.cpu().numpy().astype(np.float64)) + bias[None,:,None,None]
        projection_error = float(abs(head.memory_projection(values).cpu().numpy() - projected).max())
        assert projection_error < 2e-6
        temporal = weight.reshape(4,20,2,20)
        total = float((temporal ** 2).sum())
        contrast = float(((temporal - temporal.mean(-1,keepdims=True)) ** 2).sum())
        if mode == "none":
            assert np.count_nonzero(weight) == 0
        if mode == "last":
            assert np.allclose(temporal, temporal[:,:,:, -1:], atol=1e-12)
        projection_checks.append(dict(mode=mode, encoded_numpy_max_error=encoded_error,
            projection_numpy_max_error=projection_error,
            learned_input_time_contrast_fraction=None if total == 0 else contrast/total))
    calibration = load(OUT / "calibration.pt")["frozen_rich"]["separable_integral"]
    original_calibration = load(SOURCE / "calibration.pt")["expanded_rich"]["separable_integral"]
    assert all(torch.equal(value, original_calibration[key]) for key,value in calibration.items())
    dump(OUT / "training_independent_audit.json", dict(passed=True, training_identities_disjoint_from_heldout=True,
        actual_prelude_forecasts_exact=92, all819previous_error_inputs_used_in_independent_scale_check=True,
        numpy_memory_scale_max_error=error, projection_checks=projection_checks,
        minibatches_and_availability_masks_identical=True, allocated_parameters=48340,
        frozen_rich_calibration_exact=True, training_feature_supervision_order_pairs=18,
        limits="Same allocated capacity, deliberately different temporal information rank. Temporal weight contrast is not a transport-quality or causal-mechanism proof. Later offline packets contain labels, but labels are excluded from features."))
    print("FULLBLOCK_MEMORY_TRAINING_AUDIT_PASS", error, projection_checks, flush=True)


if __name__ == "__main__":
    main()
