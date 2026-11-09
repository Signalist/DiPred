"""Synthetic inference/causality smoke check; no raw dataset required."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .predict import ReliabilityPredictor, CHECKPOINT, PRESETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    provenance = json.loads((package.parent / "metadata/reliability/provenance.json").read_text())
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == provenance["checkpoint_sha256"]
    context = json.loads((package.parent / "metadata/reliability/context_provenance.json").read_text())
    assert hashlib.sha256(CHECKPOINT.with_name("context.pt").read_bytes()).hexdigest() == context["checkpoint_sha256"]
    for preset in PRESETS:
        predictor = ReliabilityPredictor([0.0034, -0.0034, 0.02], preset, args.device)
        # Synthetic histories only validate execution, never empirical quality.
        observed = predictor.norm["mean"].expand(1, 2, 20, 32, 64).clone()
        result = predictor.predict(observed)
        assert predictor.last_call_labels == [99, 66, 33, 0]
        assert all(torch.isfinite(result[k]).all() for k in ("prediction", "lower", "upper", "risk_half_width"))
        assert (result["lower"] <= result["upper"]).all()
        try:
            predictor.predict(observed)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Repeated forecast accepted")
        try:
            predictor.reveal(observed[:, :, :10])
        except ValueError:
            pass
        else:
            raise AssertionError("Partial target accepted")
        predictor.reveal(observed)
        next_result = predictor.predict(observed)
        assert next_result["previous_feedback_windows"] == 1
        # Reach the third issue, where context-scale corrections can activate.
        predictor.reveal(observed)
        third = predictor.predict(observed)
        assert third["previous_feedback_windows"] == 2
        assert all(torch.isfinite(third[k]).all() for k in ("prediction", "lower", "upper"))
        if preset.startswith("context_"):
            assert torch.isfinite(third["global_log_scale_correction"]).all()
        predictor.reset()
        assert predictor.pending is None and predictor.calibrator is None
        print("PASS", preset, args.device)


if __name__ == "__main__":
    main()
