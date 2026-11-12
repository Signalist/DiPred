"""Independent small-array checks of CDF averaging, atoms and crossing."""
import numpy as np
import torch
from cdf_stream import quantile_cdf_sum, ensemble_cdf_sum

grid = torch.linspace(-2, 2, 41, dtype=torch.float64)
probabilities = torch.tensor([.1, .3, .7, .9], dtype=torch.float64)
# Columns include crossed predictions, endpoint atoms and zero-width segments.
q = torch.tensor([[-1., .5, 0.], [-.4, -.5, 0.], [.6, .5, 0.], [1., 1., .2]], dtype=torch.float64)
actual = quantile_cdf_sum(q, probabilities, grid, chunk=2).numpy()
expected = np.zeros(len(grid))
for column in q.numpy().T:
    values = np.sort(column)
    for i, x in enumerate(grid.numpy()):
        total = probabilities[0].item() * (x >= values[0]) + (1-probabilities[-1].item()) * (x >= values[-1])
        for j in range(len(values)-1):
            a,b=values[j:j+2]
            fraction = float(x >= a) if a == b else np.clip((x-a)/(b-a), 0, 1)
            total += (probabilities[j+1]-probabilities[j]).item()*fraction
        expected[i] += total
np.testing.assert_allclose(actual, expected, atol=1e-12)
samples = torch.tensor([[-1., 0., .5], [.4, -.4, .5], [1., 1., -.5]], dtype=torch.float64)
actual = ensemble_cdf_sum(samples, grid, chunk=2).numpy()
expected = (samples.numpy()[:,:,None] <= grid.numpy()).mean(axis=0).sum(axis=0)
np.testing.assert_allclose(actual, expected, atol=1e-12)
print('CDF_STREAM_TEST_PASSED')

# Exercise all six arms, selection of 98/498 levels, scaling and cold start.
from cdf_stream import CDFAccumulator
for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
    acc = CDFAccumulator(device, maximum=1, points=5)
    qx = torch.zeros(499, 1, 2, 1, 1, 2, device=device)
    paths = torch.zeros(2048, 1, 2, 1, 1, 2, device=device)
    point = torch.zeros(1, 2, 1, 1, 2, device=device)
    residual = torch.tensor([0., .5], device=device).reshape(1,1,1,1,2).expand(1,2,1,1,2)
    std = torch.ones(1, 2, 1, 1, 1, device=device)
    scales = {n:torch.ones_like(point) for n in (8,32,256,2048)}
    acc.add(0, qx, paths, point, residual, scales, std)
    assert acc.count == 0
    acc.add(20, qx, paths, point, residual, scales, std)
    assert acc.count == 2
    expected = np.array([0., .5, 1., 1., 1.])
    np.testing.assert_allclose((acc.sums / acc.count).cpu().numpy(), np.broadcast_to(expected, (6,2,5)), atol=1e-9)
print('CDF_ACCUMULATOR_TEST_PASSED')
