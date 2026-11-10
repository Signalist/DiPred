"""Dataset-free checkpoint, execution and complete-feedback smoke checks."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .dynamic_predict import DynamicRiskMemoryPredictor, DYNAMIC_CHECKPOINT
from .memory_predict import SHARED_CHECKPOINT, STRATEGIES


def expect_error(exception, action):
    try:
        action()
    except exception:
        return
    raise AssertionError("Invalid feedback/stream action was accepted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    provenance = json.loads((root / "metadata/reliability/dynamic_memory/provenance.json").read_text())
    for path, field in ((SHARED_CHECKPOINT, "shared_checkpoint_sha256"),
                        (DYNAMIC_CHECKPOINT, "checkpoint_sha256")):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance[field]
    for memory in ("warp", "modulate"):
        for strategy in STRATEGIES:
            model = DynamicRiskMemoryPredictor([0.0034, -0.0034, 0.02], memory, strategy, args.device)
            assert sum(p.numel() for p in model.head.parameters()) == 58770
            history = model.norm["mean"].expand(1, 2, 20, 32, 64).clone()
            for step in range(3):
                result = model.predict(history)
                assert model.last_call_labels == [99, 66, 33, 0]
                assert result["memory_available"] == (step > 0)
                assert result["previous_feedback_windows"] == step
                assert all(torch.isfinite(result[k]).all() for k in ("prediction", "lower", "upper", "risk_half_width"))
                assert (result["upper"] >= result["lower"]).all()
                expect_error(RuntimeError, lambda: model.predict(history))
                expect_error(ValueError, lambda: model.reveal(history[:, :, :10]))
                expect_error(ValueError, lambda: model.reveal(torch.full_like(history, float("nan"))))
                expected_error = (history - model.norm["mean"]) / model.norm["std"] - model.pending
                model.reveal(history)
                assert torch.equal(model.previous_error, expected_error)
                expect_error(RuntimeError, lambda: model.predict(history + 1))
                expect_error(RuntimeError, lambda: model.reveal(history))
            model.reset()
            assert all(getattr(model, name) is None for name in (
                "pending", "previous_error", "previous_truth", "calibrator"))
            print("PASS", memory, strategy, args.device, flush=True)


if __name__ == "__main__":
    main()
