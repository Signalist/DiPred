"""Compare actual four-call forecasts and complete online states for all arms."""

import torch

from clean_reliability_common import configure, dump, manifest
from experiment_dynamic_risk_memory_followup import OUT, NAMES, MODES, Provider, controller
from experiment_reliability_transport_e12 import normalization, geometry, history, target
from experiment_fullblock_memory_risk import load
from conditional_online_risk import risk_features
from infer_dynamic_risk_memory import DynamicRiskMemoryPredictor


def expect_error(exception, action):
    try:
        action()
    except exception:
        return
    raise AssertionError("Invalid online action was accepted.")


@torch.no_grad()
def main():
    configure()
    assert (OUT / "complete.json").exists()
    provider = Provider(normalization())
    initial = load(OUT / "calibration.pt")
    geo = geometry(manifest()["files"]["test"])
    cases = []
    for name in NAMES:
        for mode in MODES:
            actual = DynamicRiskMemoryPredictor(geo, name, mode)
            reference = controller(mode, initial[name]["separable_integral"], len(geo))
            for start in (0, 20, 40, 60):
                observed = history("test", start)
                result = actual.predict(observed)
                point, bounds = provider.bounds("test", start)
                lower, upper = bounds[name]
                h = (observed - provider.norm["mean"]) / provider.norm["std"]
                lo, hi = reference.issue(lower, upper, risk_features(h, point, .5 * (upper - lower)))
                physical = point * provider.norm["std"] + provider.norm["mean"]
                expected = dict(prediction=physical, lower=physical + lo * provider.norm["std"],
                                upper=physical + hi * provider.norm["std"],
                                risk_half_width=.5 * (hi - lo) * provider.norm["std"])
                assert all(torch.equal(result[key], value) for key, value in expected.items()), (name, mode, start)
                assert actual.last_call_labels == [99, 66, 33, 0]
                assert result["memory_available"] == (start > 0)
                assert result["previous_feedback_windows"] == start // 20
                expect_error(RuntimeError, lambda: actual.predict(observed))
                truth = target("test", start)
                expect_error(ValueError, lambda: actual.reveal(truth[:, :, :10]))
                expect_error(ValueError, lambda: actual.reveal(torch.full_like(truth, float("nan"))))
                actual.reveal(truth)
                residual = (truth - provider.norm["mean"]) / provider.norm["std"] - point
                reference.reveal(residual)
                assert torch.equal(actual.previous_error, residual)
                for key in ("beta", "shape", "amplitude", "bias"):
                    assert torch.equal(getattr(actual.calibrator, key), getattr(reference, key))
                if hasattr(reference, "memory"):
                    assert len(actual.calibrator.memory) == len(reference.memory)
                    for left, right in zip(actual.calibrator.memory, reference.memory):
                        assert all(torch.equal(a, b) for a, b in zip(left, right))
                if hasattr(reference, "gram"):
                    assert torch.equal(actual.calibrator.gram, reference.gram)
                    assert torch.equal(actual.calibrator.cross, reference.cross)
                expect_error(RuntimeError, lambda: actual.predict(torch.zeros_like(truth)))
                expect_error(RuntimeError, lambda: actual.reveal(truth))
                cases.append(dict(objective=name, strategy=mode, start=start, batch=len(geo),
                                  point_intervals_previous_error_and_complete_states_exact=True,
                                  native_four_calls=True))
                print("ACTUAL_DYNAMIC_RISK_INFERENCE", name, mode, start, flush=True)
            actual.reset()
            assert all(getattr(actual, key) is None for key in ("pending", "previous_error", "calibrator"))
    dump(OUT / "direct_inference_audit.json", dict(
        passed=True, cases=cases, actual_forecasts=sum(row["batch"] for row in cases),
        native_labels=[99, 66, 33, 0], full_deque_and_moment_states_exact=True,
        duplicate_partial_nonfinite_noncontiguous_and_reset_guards=True,
        no_sampling_or_issued_forecast_cache_needed_for_actual_interface=True,
        caveat="Implementation reproduction on historical inputs, not independent statistical confirmation or coverage guarantee.",
    ))
    print("DYNAMIC_RISK_MEMORY_ACTUAL_INFERENCE_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()
