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
        predictor.reset()
        assert predictor.pending is None and predictor.calibrator is None
        print("PASS", preset, args.device)


if __name__ == "__main__":
    main()
