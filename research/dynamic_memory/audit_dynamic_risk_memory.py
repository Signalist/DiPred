"""Independent pixel transport, frozen-risk fitting and causal-pilot audit."""

import argparse
import hashlib
import json
from statistics import mean

import numpy as np
import torch

from clean_reliability_common import ROOT, configure, dump, digest
from experiment_dynamic_risk_memory import OUT, BEST, ARMS, STEPS, new_head, METRICS
from experiment_fullblock_memory_risk import OUT as MEMORY, load
from experiment_reliability_transport_e12 import normalization
from temporal_mean_memory_control import training_data
from dynamic_risk_memory import warp_risk_features


def numpy_warp(value, displacement):
    batch, channels, time, height, width = value.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    result = np.zeros_like(value)
    for b in range(batch):
        for t in range(time):
            x = np.clip(xx - displacement[b, 0, t], 0, width - 1)
            y = np.clip(yy - displacement[b, 1, t], 0, height - 1)
            x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
            x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
            wx, wy = x - x0, y - y0
            field = value[b, :, t]
            result[b, :, t] = (
                field[:, y0, x0] * (1 - wx) * (1 - wy)
                + field[:, y0, x1] * wx * (1 - wy)
                + field[:, y1, x0] * (1 - wx) * wy
                + field[:, y1, x1] * wx * wy
            )
    return result


def numerical_audit():
    torch.manual_seed(42)
    value = torch.randn(1, 4, 2, 4, 8, dtype=torch.float64)
    displacement = torch.empty(1, 2, 2, 4, 8, dtype=torch.float64)
    displacement[:, 0] = .27
    displacement[:, 1] = -.31
    displacement.requires_grad_(True)
    output = warp_risk_features(value, displacement)
    array, flow = value.numpy(), displacement.detach().numpy().copy()
    expected = numpy_warp(array, flow)
    error = float(np.max(np.abs(output.detach().numpy() - expected)))
    assert error < 1e-12
    output.square().mean().backward()
    checks = []
    for coordinate in ((0, 0, 1, 2, 3), (0, 1, 1, 2, 3)):
        perturbation = np.zeros_like(flow)
        perturbation[coordinate] = 1e-6
        plus = np.mean(numpy_warp(array, flow + perturbation) ** 2)
        minus = np.mean(numpy_warp(array, flow - perturbation) ** 2)
        finite_difference = (plus - minus) / 2e-6
        actual = float(displacement.grad[coordinate])
        assert abs(actual - finite_difference) < 1e-9
        checks.append(dict(coordinate=coordinate, autograd=actual,
                           numpy_finite_difference=float(finite_difference)))
    identity = warp_risk_features(value, torch.zeros_like(displacement))
    assert torch.equal(identity, value)
    impulse = torch.zeros(1, 4, 2, 4, 8, dtype=torch.float64)
    impulse[0, 0, 0, 1, 2] = 1
    right = torch.zeros(1, 2, 2, 4, 8, dtype=torch.float64)
    right[:, 0] = 1
    moved = warp_risk_features(impulse, right)
    assert float(moved[0, 0, 0, 1, 3]) == 1
    assert float(moved[0, 0, 0, 1, 2]) == 0
    dump(OUT / "independent_warp_numerics.json", dict(
        passed=True, numpy_bilinear_output_max_error=error, finite_difference_checks=checks,
        zero_displacement_identity_exact=True, positive_x_moves_features_right=True,
        border_replication_and_no_periodic_wrap=True,
        limitation="Numerical feature transport, not evidence of a physical error velocity or useful reliability gains.",
    ))
    print("DYNAMIC_RISK_WARP_NUMERICS_PASS", flush=True)


def fitting_audit():
    data = training_data(normalization())
    base, last, history, memory, residual = data
    scale = load(MEMORY / "statistics.pt")["memory_scale"]
    frozen = new_head("frozen_best", scale)
    frozen_state = frozen.state_dict()
    initial_modifiers = {}
    checks = []
    expected = frozen.bounds(base[:8], last[:8], history[:8], memory[:8], True)
    for name in ARMS:
        head = new_head(name, scale)
        initial = load(OUT / (name + "_initial.pt"))["model"]
        final = load(OUT / (name + ".pt"))["model"]
        assert all(torch.equal(value, initial[key]) and torch.equal(value, final[key])
                   for key, value in frozen_state.items())
        initial_modifiers[name] = {key: value for key, value in initial.items() if key.startswith("modifier.")}
        head.load_state_dict(initial)
        actual = head.bounds(base[:8], last[:8], history[:8], memory[:8], True)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        first = load(OUT / (name + "_first_update.pt"))
        optimizer = torch.optim.AdamW(head.modifier.parameters(), lr=3e-4, weight_decay=1e-5)
        indices, available = first["indices"], first["available"]
        lower, upper = head.bounds(base[indices], last[indices], history[indices], memory[indices], available)
        a, b = residual[indices] - lower, residual[indices] - upper
        loss = (torch.where(a >= 0, .05 * a, -.95 * a)
                + torch.where(b >= 0, .95 * b, -.05 * b)).mean()
        assert float(loss) == first["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.modifier.parameters(), 1)
        gradient_error = max(float((parameter.grad - first["gradients"][key]).abs().max())
                             for key, parameter in head.modifier.named_parameters())
        assert all(torch.allclose(parameter.grad, first["gradients"][key], rtol=1e-5, atol=1e-8)
                   for key, parameter in head.modifier.named_parameters())
        assert all(parameter.grad is None for key, parameter in head.named_parameters()
                   if not key.startswith("modifier."))
        optimizer.step()
        update_error = max(float((value - first["model"][key]).abs().max())
                           for key, value in head.state_dict().items())
        assert all(torch.allclose(value, first["model"][key], rtol=1e-6, atol=1e-7)
                   for key, value in head.state_dict().items())
        head.load_state_dict(final)
        with torch.no_grad():
            cold = head.bounds(base[:8], last[:8], history[:8], memory[:8], False)
            cold_changed_memory = head.bounds(base[:8], last[:8], history[:8], memory[:8] + 100, False)
            source_cold = frozen.bounds(base[:8], last[:8], history[:8], memory[:8], False)
            assert all(torch.equal(a, b) and torch.equal(a, c)
                       for a, b, c in zip(cold, cold_changed_memory, source_cold))
        checks.append(dict(name=name, loss=float(loss), gradient_max_error=gradient_error,
                           first_update_max_error=update_error,
                           original48340parameters_and_buffers_exact=True,
                           raw_cold_bounds_ignore_unavailable_memory=True))
    assert all(torch.equal(value, initial_modifiers["warp"][key])
               for key, value in initial_modifiers["modulate"].items())
    return checks


def complete_audit():
    assert (OUT / "complete.json").exists()
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert digest(ROOT / "experiment_dynamic_risk_memory.py") == protocol["source_sha256"]
    assert digest(ROOT / "dynamic_risk_memory.py") == protocol["module_sha256"]
    assert all(digest(ROOT / path) == sha for path, sha in protocol["frozen_artifacts"].items())
    frozen = json.loads((OUT / "heads_frozen.json").read_text())
    assert all(digest(OUT / (name + ".pt")) == sha for name, sha in frozen["artifacts"].items())
    sampler = torch.Generator(device="cuda").manual_seed(42)
    schedule = hashlib.sha256()
    for _ in range(STEPS):
        indices = torch.randint(819, (4,), device="cuda", generator=sampler)
        available = (torch.rand(4, device="cuda", generator=sampler) >= .2).float()
        schedule.update(indices.cpu().numpy().tobytes())
        schedule.update(available.cpu().numpy().tobytes())
    assert set(frozen["minibatches_and_availability_masks_sha256"].values()) == {schedule.hexdigest()}
    first = fitting_audit()
    rows = json.loads((OUT / "per_case.json").read_text())
    reference = json.loads((BEST / "per_case.json").read_text())
    keys = ("file", "start", "channel", "variant")
    lookup = {tuple(row[key] for key in keys): row for row in reference if row["map"] == "density_quantile"}
    controls = 0
    for row in rows:
        if row["map"] == "frozen_best":
            previous = lookup[tuple(row[key] for key in keys)]
            assert all(row[key] == value for key, value in previous.items() if isinstance(value, float))
            controls += 1
    assert controls == 1760 and len(rows) == 5280
    events = [json.loads(line) for line in (OUT / "pilot_events.jsonl").read_text().splitlines()]
    assert len(events) == 20
    for issue, reveal in zip(events[::2], events[1::2]):
        assert issue["event"] == "issued" and reveal["event"] == "revealed"
        assert issue["start"] == reveal["start"] and issue["time"] <= reveal["time"]
    values = json.loads((OUT / "pilot_macro.json").read_text())
    for mode, methods in values.items():
        for name, expected in methods.items():
            groups = {}
            for row in rows:
                if row["map"] == name and row["variant"] == mode and row["start"] >= 600:
                    groups.setdefault((row["group"], row["channel"]), []).append(row)
            actual = {metric: (mean(abs(mean(r["coverage"] for r in group) - .9) for group in groups.values())
                               if metric == "coverage_error"
                               else mean(mean(r[metric] for r in group) for group in groups.values()))
                      for metric in METRICS}
            assert actual == expected
    decision = json.loads((OUT / "gate.json").read_text())
    for mode, result in decision["results"].items():
        successes = []
        for reference, expected in result["comparisons"].items():
            a, b = values[mode]["warp"], values[mode][reference]
            gains = {key: direction * (a[key] - b[key]) for key, direction in METRICS.items()}
            failures = [key for key, value in gains.items() if value < -1e-6]
            strict = gains["standardized_interval_score"] > 1e-6
            assert expected == dict(gains=gains, regressions=failures, strict_score_gain=strict)
            successes.append(not failures and strict)
        assert result["passed"] == all(successes)
    assert decision["proceed_full_evaluation"] == any(row["passed"] for row in decision["results"].values())
    dump(OUT / "independent_training_pilot_audit.json", dict(
        passed=True, first_update_checks=first, original_risk_weights_and_norms_exact=True,
        initial_raw_bounds_equal_current_best=True, cold_raw_bounds_independent_of_memory=True,
        identical_initial_module_parameters=True, same2000step_sampler_regenerated=True,
        original_reference_pilot_rows_exact=controls, all_pilot_rows=len(rows),
        complete_pilot_issue_reveal_pairs=10, all_eight_macro_metrics_and_gates_exact=True,
        no_heldout_or_calibration_labels_fit_parameters=True,
        limitation="Latent risk features and learned statistical displacement only; no physical velocity, transport benefit, independent generalization or joint coverage proven by implementation checks.",
    ))
    print("DYNAMIC_RISK_MEMORY_TRAINING_PILOT_AUDIT_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numerics-only", action="store_true")
    args = parser.parse_args()
    configure()
    numerical_audit()
    if not args.numerics_only:
        complete_audit()


if __name__ == "__main__":
    main()
