"""Evaluate the requested compact16 against real32-path predictive spread.

The32-path reference uses the same frozen DiPred and common zero-noise point
center, independently recalibrated. This isolates error-uncertainty quality.
It is not the original ensemble-mean point forecast and does not fabricate
sample quantiles from a standard-deviation cache. No extra learned controls.
"""
import json
import time
import torch
from clean_reliability_common import ROOT, configure, dump, digest
from experiment_selected16_quantile import OUT as TRAIN, Provider as OriginalProvider, load
from experiment_reliability_transport_e12 import normalization, issued, target, PREDICTOR
from causal_shape_calibration import initial_state
from experiment_continue_rich_uncertainty import controller
from replay_conditional_risk import replay

OUT = ROOT / 'runs/selected16_noise32_requested_20261110'
NAMES = ('selected16', 'noise32')
MODES = ('memory_dynamic',)


class Provider(OriginalProvider):
    def __init__(self, norm):
        super().__init__(norm)
        self.heads = {'selected16': self.heads['compact16']}

    @torch.no_grad()
    def bounds(self, split, start):
        point, bounds = super().bounds(split, start)
        saved = issued('diffusion', split, start)
        physical = point * self.norm['std'] + self.norm['mean']
        assert torch.allclose(physical, saved['zero'], rtol=0, atol=1e-6)
        # These are actual32-path moments from the original sampler, not
        # network-predicted sigma, synthetic targets or new training seeds.
        std = (saved['std32'] / self.norm['std']).clamp_min(1e-6)
        radius = 1.6448536269514722 * std
        bounds['noise32'] = (-radius, radius)
        return point, bounds


@torch.no_grad()
def main():
    configure()
    assert json.loads((TRAIN / 'independent_audit.json').read_text())['passed']
    OUT.mkdir(exist_ok=False)
    paths = [TRAIN / 'compact16.pt', TRAIN / 'calibration.pt', PREDICTOR / 'augmented_diffusion/model.pt']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    dump(OUT / 'protocol.json', dict(time=time.time(), source_sha256=digest(ROOT / 'evaluate_selected16_noise32.py'),
        predictor_sha256=digest(paths[-1]), frozen_artifacts=hashes, names=NAMES, modes=MODES,
        user_scope='Onlyselected16versusoriginal24and32noise; this file executes the unambiguousselected16/32noise portion, original24identity awaitsconfirmation. No masked16or othernewbaseline evaluation.',
        point_center='Common original4call zero-noise P.32-noise variance estimated from32actual paths, scaled1.64485thenindependentcalibration. Not32pathmean pointcenter andnotempirical5/95quantiles.',
        calibration='Same10independentidentities, samefixedmemory_dynamic onlinecontroller for both, rawheads frozen. All complete20issuedbeforereveal.',
        native_calls=dict(selected16=4, fresh32sampling_reference_including_common_center=132),
        runtime='Cached32statistics reused for evaluation, cache IO isnot deploymenttiming. No extra32path sampling orneuraltraining.',
        caveat='Exploratory reused historicalidentities; userselectedsubset afterpreviousstudies. No independentstatistical or formalcoverage guarantee.'))
    provider = Provider(normalization())
    scores = {name: [] for name in NAMES}
    for start in range(0, 161, 20):
        point, bounds = provider.bounds('calibration', start)
        residual = (target('calibration', start) - provider.norm['mean']) / provider.norm['std'] - point
        for name, (lower, upper) in bounds.items():
            ratio = (residual - .5 * (lower + upper)).abs() / (.5 * (upper - lower)).clamp_min(1e-6)
            scores[name].append(ratio.cpu())
    initial = {name: {'separable_integral': initial_state(torch.cat(values).cuda(), 'separable_integral')}
               for name, values in scores.items()}
    reference = load(TRAIN / 'calibration.pt')['compact16']['separable_integral']
    assert all(torch.equal(value, reference[key]) for key, value in initial['selected16']['separable_integral'].items())
    torch.save({name: {mode: {key: value.cpu() for key, value in state.items()} for mode, state in modes.items()}
                for name, modes in initial.items()}, OUT / 'calibration.pt')
    replay(OUT, provider, NAMES, MODES, controller, initial)
    assert all(digest(ROOT / path) == value for path, value in hashes.items())
    dump(OUT / 'complete.json', dict(time=time.time(), frozen_artifacts_unchanged=True,
                                    calibration_selected16_reproduced_exact=True, original24_not_yet_resolved=True))
    print('SELECTED16_NOISE32_REQUESTED_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
