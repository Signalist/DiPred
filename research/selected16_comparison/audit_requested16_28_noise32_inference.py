"""Actual three-method inference, actual32sampling and complete online states."""
import torch
from clean_reliability_common import configure, dump, manifest
from report_requested16_28_noise32 import OUT, NAMES, Provider
from experiment_reliability_transport_e12 import normalization, geometry, history, target, issued
from experiment_continue_rich_uncertainty import controller
from experiment_fullblock_memory_risk import load
from conditional_online_risk import risk_features
from infer_requested16_28_noise32 import RequestedThreePredictor


def expect_error(kind, action):
    try:
        action()
    except kind:
        return
    raise AssertionError('Invalid stream action accepted.')


@torch.no_grad()
def main():
    configure()
    provider = Provider(normalization())
    initial = load(OUT / 'calibration.pt')
    geo = geometry(manifest()['files']['test'])
    cases = []
    for name in NAMES:
        actual = RequestedThreePredictor(geo, name)
        reference = controller('memory_dynamic', initial[name]['separable_integral'], len(geo))
        for start in (0, 20):
            observed = history('test', start)
            calls = []
            hook = actual.model.net.register_forward_pre_hook(lambda module, args: calls.append(int(args[1][0])))
            try:
                result = actual.predict(observed)
            finally:
                hook.remove()
            expected_calls = 132 if name == 'noise32' else 4
            assert len(calls) == expected_calls and calls == [99, 66, 33, 0] * (expected_calls // 4)
            assert actual.last_call_labels == [99, 66, 33, 0]
            if name == 'noise32':
                cached = issued('diffusion', 'test', start)['std32']
                assert torch.equal(actual.head.last_physical_std, cached), 'Original32-path sampler/cache mismatch.'
            point, bounds = provider.bounds('test', start)
            lower, upper = bounds[name]
            h = (observed - provider.norm['mean']) / provider.norm['std']
            lo, hi = reference.issue(lower, upper, risk_features(h, point, .5 * (upper - lower)))
            physical = point * provider.norm['std'] + provider.norm['mean']
            expected = dict(prediction=physical, lower=physical + lo * provider.norm['std'],
                            upper=physical + hi * provider.norm['std'], risk_half_width=.5 * (hi - lo) * provider.norm['std'])
            assert all(torch.equal(result[key], value) for key, value in expected.items()), (name, start)
            assert result['memory_available'] == (start > 0)
            assert result['previous_feedback_windows'] == start // 20
            expect_error(RuntimeError, lambda: actual.predict(observed))
            truth = target('test', start)
            expect_error(ValueError, lambda: actual.reveal(truth[:, :, :10]))
            expect_error(ValueError, lambda: actual.reveal(torch.full_like(truth, float('nan'))))
            actual.reveal(truth)
            residual = (truth - provider.norm['mean']) / provider.norm['std'] - point
            reference.reveal(residual)
            assert torch.equal(actual.previous_error, residual)
            for key in ('beta', 'shape', 'amplitude', 'bias'):
                assert torch.equal(getattr(actual.calibrator, key), getattr(reference, key))
            for left, right in zip(actual.calibrator.memory, reference.memory):
                assert all(torch.equal(a, b) for a, b in zip(left, right))
            assert len(actual.calibrator.memory) == len(reference.memory)
            expect_error(RuntimeError, lambda: actual.predict(torch.zeros_like(truth)))
            expect_error(RuntimeError, lambda: actual.reveal(truth))
            cases.append(dict(method=name, start=start, batch=len(geo), actual_backbone_calls=expected_calls,
                              point_intervals_previous_error_and_all_states_exact=True,
                              original32path_std_exact=name == 'noise32'))
            print('REQUESTED_NATIVE_INFERENCE', cases[-1], flush=True)
        actual.reset()
        assert all(getattr(actual, key) is None for key in ('pending', 'previous_error', 'calibrator'))
    dump(OUT / 'direct_inference_audit.json', dict(passed=True, cases=cases, actual_forecasts=sum(r['batch'] for r in cases),
        original32sample_std_regenerated_exact=True, original_fourcall_point_labels=[99, 66, 33, 0],
        complete_online_states_and_stream_guards_exact=True,
        caveat='Implementation validation on historicalinputs, not independent statistical or formal coverage proof.'))
    print('REQUESTED_THREE_NATIVE_INFERENCE_AUDIT_PASS', flush=True)


if __name__ == '__main__':
    main()
