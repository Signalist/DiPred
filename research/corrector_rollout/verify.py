"""Checks against historical scores and interventions on the rollout inputs."""
import csv
import json
from collections import defaultdict

import numpy as np
import torch


def verify_scores(rows, prior):
    trajectories = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = row["split"], row["trajectory"], row["model"]
        trajectories[key][row["channel"]].append(row["residual_mse"])
    with (prior / "per_trajectory.csv").open(newline="") as handle:
        historical = list(csv.DictReader(handle))
    expected = {(r["split"], r["trajectory"], r["model"]): r for r in historical}
    assert set(trajectories) == set(expected) and len(expected) == len(historical)
    summaries = defaultdict(list)
    errors = []
    for key, channels in trajectories.items():
        assert len(channels["u"]) == len(channels["v"]) == 38
        u, v = (np.mean(channels[channel]) for channel in ("u", "v"))
        actual = [(u + v) / 2, u, v]
        reference = [float(expected[key][metric]) for metric in ("mse", "mse_u", "mse_v")]
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-10, err_msg=str(key))
        errors.extend(np.abs(np.subtract(actual, reference)))
        summaries[key[0], key[2]].append(actual)

    aggregate = json.loads((prior / "summary.json").read_text())
    assert {(r["split"], r["model"]) for r in aggregate} == set(summaries)
    reproduced = []
    for row in aggregate:
        key = row["split"], row["model"]
        assert len(summaries[key]) == row["trajectories"]
        actual = np.mean(summaries[key], axis=0)
        reference = [row[metric] for metric in ("mse", "mse_u", "mse_v")]
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-10, err_msg=str(key))
        reproduced.append(dict(split=key[0], model=key[1], historical_mse=reference[0],
                               mse=actual[0], mse_u=actual[1], mse_v=actual[2]))
    return dict(trajectory_model_rows=len(trajectories), aggregate_rows=len(aggregate),
                max_trajectory_mse_abs_error=max(errors), aggregates=reproduced)


def verify_rollout(data, models, device, decode, rollout, one_step, advance, free_window):
    def get_window(start):
        truth, cached = data["development", start]
        return decode(truth[0], cached[0], device)

    truth, backbone, residual = get_window(400)
    second_window = get_window(420)
    changed_truth = truth.clone()
    changed_truth[:, 1:] += 0.25
    changed_residual = residual.clone()
    changed_residual[:, :, 1:] += 0.25

    # Exercise the same prediction-and-scoring entry point as the actual experiment.
    original, scores = free_window(models, backbone, truth, residual)
    intervened, changed_scores = free_window(models, backbone, changed_truth, changed_residual)
    reference = one_step(models, backbone, truth, residual)
    initial = residual[:, :, 0]
    lead_2_max_abs_delta = {}
    for name in original:
        assert torch.equal(original[name][:, :, 0], initial)
        assert torch.equal(original[name], intervened[name])
        for metric, changed_metric in zip(scores[name], changed_scores[name]):
            assert not np.array_equal(metric, changed_metric)
        free_lead_2 = original[name][:, :, 1]
        oracle_lead_2 = reference[name][:, :, 0]
        lead_2_max_abs_delta[name] = float((free_lead_2 - oracle_lead_2).abs().max())
        # Reconstructing the conditioned state incurs float32 cancellation. Rare
        # values then round to adjacent bins at the historical TF32 conv input.
        atol = 2e-6 if name == "State-conditioned" else 0
        torch.testing.assert_close(free_lead_2, oracle_lead_2, rtol=0, atol=atol)
        target = residual[:, :, 1].double()
        torch.testing.assert_close((target - free_lead_2.double()).square().mean(),
                                   (target - oracle_lead_2.double()).square().mean(),
                                   rtol=1e-4, atol=1e-10)
    torch.testing.assert_close(original["Persistence"], initial.unsqueeze(2).expand_as(backbone),
                               rtol=0, atol=0)

    # Compare every step against a direct legacy-sign invocation on the same input.
    # The reference is deliberately independent of the sign-converting adapter.
    for name, (model, scale) in models.items():
        assert not model.training and all(not p.requires_grad for p in model.parameters())
        for lead in range(2, 21):
            current = original[name][:, :, lead - 2]
            predicted = backbone[:, :, lead - 2]
            if name == "R-only":
                predicted = torch.zeros_like(current)
            state = predicted + current if name == "State-conditioned" else predicted
            labels = torch.full((32,), lead, device=device)
            legacy = model(state / scale, predicted / scale, (-current) / scale, labels) * scale
            torch.testing.assert_close(original[name][:, :, lead - 1], -legacy, rtol=0, atol=0)

    # Nonzero residuals, a non-unit normalization, and a non-identity legacy network
    # make input and output signs observable separately.
    class LegacyProbe:
        def __call__(self, state, forecast, old_residual, labels):
            torch.testing.assert_close(state, torch.full_like(state, 4.0))
            torch.testing.assert_close(forecast, torch.full_like(forecast, 3.0))
            torch.testing.assert_close(old_residual, torch.full_like(old_residual, -1.0))
            assert labels.tolist() == [2, 2]
            return old_residual * 2

    current = torch.full((2, 2, 2, 2), 0.125, device=device)
    prediction = current * 3
    state = prediction + current
    torch.testing.assert_close(advance((LegacyProbe(), 0.125), current, prediction, state, 2),
                               2 * current, rtol=0, atol=0)
    torch.testing.assert_close(backbone.double() + residual.double(),
                               truth.double().unsqueeze(0).expand_as(backbone), rtol=0, atol=1.2e-7)

    # After evaluating and intervening on one window, the next window retains its
    # own first-frame input. A concatenated window is rejected by the public API.
    y2, b2, r2 = second_window
    later, _ = free_window(models, b2, y2, r2)
    # Perturb the first window's seed while keeping the subsequent window fixed.
    changed_first_truth = truth.clone()
    changed_first_truth[:, 0] += 0.25
    changed_first_residual = residual.clone()
    changed_first_residual[:, :, 0] += 0.25
    first_again, _ = free_window(models, backbone, changed_first_truth, changed_first_residual)
    for name in first_again:
        assert not torch.equal(first_again[name][:, :, -1], original[name][:, :, -1])
    later_again, _ = free_window(models, b2, y2, r2)
    for name in later:
        assert torch.equal(later[name][:, :, 0], r2[:, :, 0])
        assert torch.equal(later[name], later_again[name])
    assert not torch.equal(r2[:, :, 0], original["Persistence"][:, :, -1])
    try:
        rollout(models, torch.cat((backbone, b2), dim=2), initial)
    except ValueError as exc:
        assert "one 20-frame" in str(exc)
    else:
        raise AssertionError("Accepted concatenated windows")

    return dict(sign_probe=True, legacy_sign_and_lead_all_steps=True,
                shared_initial=True, lead_2_matches_oracle_within_tolerance=True,
                lead_2_max_abs_output_difference=lead_2_max_abs_delta, persistence_constant=True,
                future_truth_and_residual_mutation_predictions_bitwise_identical=True,
                score_responds_to_mutation=True, windows_reset_and_concatenation_rejected=True,
                physical_residual_identity=True, models_frozen=True)
