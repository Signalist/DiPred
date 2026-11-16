"""Small independent checks for noise-path and quantile calculations."""
import sys
import torch
from common import ROOT, sample_predict
sys.path.insert(0, str(ROOT / 'xpf16_02grid_2048'))
from distribution_grid import quantiles_and_pair
from calibration_grid import GridCalibrator
from evaluate import sorted_cdf_sum, quantile_crps


class ZeroDenoiser(torch.nn.Module):
    def forward(self, value, labels, history):
        return 0.25 * value + 0.1 * history


class Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = ZeroDenoiser()
        self.register_buffer('abar', torch.tensor([.98, .8, .5, .2]))


def main():
    model = Diffusion()
    x = torch.arange(80., dtype=torch.float32).reshape(1, 2, 20, 1, 2) / 100
    mask = torch.ones(1, 2, 1, 1, 1)
    paths = sample_predict(model, x, mask, 32)
    assert torch.equal(paths[:8], sample_predict(model, x, mask, 8))
    levels = torch.tensor([.002, .05, .5, .95, .998])
    actual, _ = quantiles_and_pair(paths, torch.zeros_like(x), levels, chunk=17)
    expected = torch.quantile(paths, levels, dim=0, interpolation='linear')
    assert torch.allclose(actual, expected, atol=2e-6)
    alpha = torch.tensor([.05])
    controller = GridCalibrator(alpha, torch.zeros(1, 2, 2), 1)
    lo = torch.zeros(1, 1, 2, 20, 4, 4)
    hi = lo + 1
    issued = controller.issue(lo, hi)
    assert torch.equal(issued[0], lo) and torch.equal(issued[1], hi)
    try:
        controller.issue(lo, hi)
        raise AssertionError('A second forecast before reveal must fail')
    except RuntimeError:
        pass
    assert controller.updates == 0
    controller.reveal(torch.ones(1, 2, 20, 4, 4) * 2)
    assert controller.updates == 1
    assert not torch.equal(controller.issue(lo, hi)[0], lo)
    # Independently sum a two-point predictive CDF, including a quantile atom.
    values = torch.tensor([[-1., -.5], [0., 0.], [0., 1.], [2., 2.]])
    probs = torch.tensor([.1, .3, .7, .9], dtype=torch.float64)
    grid = torch.tensor([-.75, 0., .5, 1.5], dtype=torch.float64)
    actual_cdf = sorted_cdf_sum(values, probs, grid)
    assert torch.allclose(actual_cdf, torch.tensor([.15, 1., 1.25, 1.65], dtype=torch.float64))
    # Trapezoidal quantile integration, with clamped tails, against NumPy.
    levels = torch.arange(1, 500, dtype=torch.float32) / 500
    q = torch.linspace(-1, 1, 499).reshape(499, 1, 1, 1, 1, 1)
    y = torch.full((1, 1, 1, 1, 1), .3)
    result = quantile_crps(q, y, levels)
    points = torch.cat((q[:1], q, q[-1:])).flatten().numpy()
    import numpy as np
    tau = np.linspace(0, 1, 501)
    error = .3 - points
    reference = 2 * np.trapz(np.maximum(tau * error, (tau - 1) * error), tau)
    assert abs(float(result) - reference) < 1e-6
    print('PASS: path prefixes, empirical quantiles, causal feedback, CDF and CRPS')


if __name__ == '__main__':
    main()
