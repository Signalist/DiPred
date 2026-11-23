"""Fixed per-grid-point features and weighted multioutput ridge probes."""
from dataclasses import dataclass

import numpy as np


MODELS = ('A0', 'AR', 'AH', 'AHR')
BASELINES = ('zero', 'train_lead_channel_mean')
COMPARISONS = (('AR', 'A0'), ('AH', 'A0'), ('AHR', 'AH'))
FIELD = (2, 20, 32, 64)
FEATURES = 202
OUTPUTS = 80
HEIGHT, WIDTH = FIELD[-2:]
SIZE = HEIGHT * WIDTH
LAYOUT = {
    'last_observation': slice(0, 40),
    'native_mean': slice(40, 80),
    'native_std': slice(80, 120),
    'previous_residual': slice(120, 160),
    'earlier_observation': slice(160, 200),
    'xy': slice(200, 202),
}
Y, X = np.meshgrid(np.linspace(-1., 1., HEIGHT), np.linspace(-1., 1., WIDTH), indexing='ij')
COORDS = np.column_stack((X.ravel(), Y.ravel()))


def points(episode_index, window):
    """Pre-fixed sample of 128 distinct positions from one training warm window."""
    assert window in (2, 3)
    rng = np.random.default_rng(42600000 + 2 * episode_index + window - 2)
    return rng.choice(SIZE, 128, replace=False)


def flatten_field(field, index):
    assert field.shape == FIELD
    return field.reshape(40, SIZE)[:, index].T


def features(last_observation, current_native_mean, current_native_std,
             previous_native_residual, earlier_observation, index=slice(None)):
    """The feature interface only accepts quantities available before the current reveal."""
    fields = (last_observation, current_native_mean, current_native_std,
              previous_native_residual, earlier_observation)
    return np.concatenate([*(flatten_field(field, index) for field in fields), COORDS[index]], axis=1)


def labels(current_observation, current_native_mean, index=slice(None)):
    signed = flatten_field(current_observation - current_native_mean, index).astype(np.float64)
    return np.concatenate((signed, signed**2), axis=1)


def active_slots(model):
    assert model in MODELS
    mask = np.ones(FEATURES, dtype=bool)
    if model not in ('AR', 'AHR'):
        mask[LAYOUT['previous_residual']] = False
    if model not in ('AH', 'AHR'):
        mask[LAYOUT['earlier_observation']] = False
    return mask


def weighted_location_scale(matrix, weights):
    location = np.average(matrix, axis=0, weights=weights)
    variance = np.average((matrix - location)**2, axis=0, weights=weights)
    return location, np.where(variance > 0., np.sqrt(variance), 1.)


@dataclass(frozen=True)
class Probe:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, model, matrix):
        standardized = (matrix - self.feature_mean) / self.feature_scale
        if model == 'zero':
            return np.zeros((len(matrix), OUTPUTS), dtype=np.float64)
        if model == 'train_lead_channel_mean':
            return np.broadcast_to(self.target_mean, (len(matrix), OUTPUTS))
        mask = active_slots(model)
        standardized[:, ~mask] = 0.
        standardized_prediction = standardized @ self.coefficients[MODELS.index(model)]
        estimate = standardized_prediction * self.target_scale + self.target_mean
        estimate[:, 40:] = np.maximum(estimate[:, 40:], 0.)
        return estimate

    def arrays(self):
        return dict(feature_mean=self.feature_mean, feature_scale=self.feature_scale,
                    target_mean=self.target_mean, target_scale=self.target_scale,
                    coefficients=self.coefficients)


def fit(matrix, target, trajectory_ids):
    """Each trajectory has the same total weight, while row weights sum to N.

    The centered standardized weighted normal equation uses lambda=1 in
    sum-of-squared-errors units. The intercept is the weighted target mean.
    """
    assert matrix.shape == (len(target), FEATURES) and target.shape[1:] == (OUTPUTS,)
    _, inverse, counts = np.unique(trajectory_ids, return_inverse=True, return_counts=True)
    assert len(inverse) == len(matrix)
    weights = len(matrix) / (len(counts) * counts[inverse])
    assert np.allclose(np.bincount(inverse, weights=weights), len(matrix) / len(counts))
    feature_mean, feature_scale = weighted_location_scale(matrix, weights)
    target_mean, target_scale = weighted_location_scale(target, weights)
    scale = weights[:, None]**.5
    x = scale * ((matrix - feature_mean) / feature_scale)
    y = scale * ((target - target_mean) / target_scale)
    gram = x.T @ x
    cross = x.T @ y
    coefficients = np.zeros((len(MODELS), FEATURES, OUTPUTS), dtype=np.float64)
    for index, model in enumerate(MODELS):
        mask = active_slots(model)
        lhs = gram[np.ix_(mask, mask)] + np.eye(mask.sum())
        coefficients[index, mask] = np.linalg.solve(lhs, cross[mask])
    return Probe(feature_mean, feature_scale, target_mean, target_scale, coefficients)


def losses(estimated, target):
    squared = (estimated - target)**2
    return dict(signed_mse=float(squared[:, :40].mean()),
                second_moment_mse=float(squared[:, 40:].mean()))


def paired_differences(first, second):
    ids = sorted(first)
    assert set(ids) == set(second)
    offsets = np.asarray([first[track] - second[track] for track in ids])
    rng = np.random.default_rng(123)
    samples = rng.integers(len(ids), size=(10000, len(ids)))
    return dict(delta=float(offsets.mean()), ci95=np.quantile(offsets[samples].mean(axis=1), [.025, .975]).tolist(),
                n=len(ids), improved=int(np.sum(offsets < 0)), trajectory_delta=dict(zip(ids, offsets.tolist())))
