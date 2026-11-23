"""CPU checks for the fixed A feature layout and weighted normal equations."""
import numpy as np

from research.residual_information_study.information_probe.probe import (
    FEATURES, FIELD, LAYOUT, MODELS, OUTPUTS, active_slots, features, fit, labels, points,
)


def run():
    constants = [np.full(FIELD, float(j), dtype=np.float32) for j in range(1, 6)]
    sample = points(11, 2)
    assert len(set(sample.tolist())) == 128
    assert np.array_equal(sample, points(11, 2))
    assert not np.array_equal(sample, points(11, 3))
    assert not np.array_equal(sample, points(12, 2))
    full = features(*constants, sample)
    assert full.shape == (128, FEATURES)
    for i, slot in enumerate(LAYOUT.values(), 1):
        if i <= 5:
            assert np.all(full[:, slot] == i)
    assert np.all(np.abs(full[:, -2:]) <= 1)
    assert [int(active_slots(name).sum()) for name in MODELS] == [122, 162, 162, 202]
    target = labels(constants[0], constants[2], sample)
    assert np.all(target[:, :40] == -2.) and np.all(target[:, 40:] == 4.)

    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(36, FEATURES))
    signed = rng.normal(size=(36, 40))
    targets = np.column_stack([signed, signed**2])
    tracks = ['a'] * 12 + ['b'] * 24
    trained = fit(matrix, targets, tracks)
    assert trained.coefficients.shape == (4, FEATURES, OUTPUTS)
    assert np.allclose(trained.target_mean, (targets[:12].mean(0) + targets[12:].mean(0)) / 2)
    weights = np.array([1.5] * 12 + [.75] * 24)
    standardized = (matrix - trained.feature_mean) / trained.feature_scale
    standardized_targets = (targets - trained.target_mean) / trained.target_scale
    for i, name in enumerate(MODELS):
        mask = active_slots(name)
        x = standardized[:, mask]
        y = standardized_targets
        b = trained.coefficients[i, mask]
        assert np.all(trained.coefficients[i, ~mask] == 0.)
        # Optimality condition for D-weighted standardized normal equations.
        assert np.max(np.abs(x.T @ (weights[:, None] * (x @ b - y)) + b)) < 1e-10
        prediction = trained.predict(name, matrix)
        assert np.all(prediction[:, 40:] >= 0.)
        altered = matrix.copy()
        altered[:, ~mask] += 10**5
        assert np.array_equal(prediction, trained.predict(name, altered))


if __name__ == '__main__':
    run()
    print('A: feature, label, seed and ridge checks passed')
