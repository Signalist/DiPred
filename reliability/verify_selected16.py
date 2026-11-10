"""Dataset-free three-method checkpoint and complete-window stream checks."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .selected16_predict import RequestedThreePredictor, CHECKPOINT, METHODS
from .memory_predict import SHARED_CHECKPOINT


def expect_error(kind, action):
    try:
        action()
    except kind:
        return
    raise AssertionError('Invalid stream action was accepted.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    provenance = json.loads((root / 'metadata/reliability/selected16_comparison/provenance.json').read_text())
    for path, key in ((CHECKPOINT, 'checkpoint_sha256'), (SHARED_CHECKPOINT, 'shared_checkpoint_sha256')):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance[key]
    for method in METHODS:
        model = RequestedThreePredictor([0.0034, -0.0034, .02], method, args.device)
        if method != 'noise32':
            assert sum(p.numel() for p in model.head.parameters()) == provenance['risk_parameters'][method]
        if method == 'selected16':
            assert model.head.net[0].in_channels == 16
            assert not any(k.startswith(('modifier.', 'history_projection.', 'memory_projection.')) for k in model.head.state_dict())
        history = model.norm['mean'].expand(1, 2, 20, 32, 64).clone()
        for step in range(2):
            calls = []
            hook = model.model.net.register_forward_pre_hook(lambda module, values: calls.append(int(values[1][0])))
            try:
                result = model.predict(history)
            finally:
                hook.remove()
            assert calls == [99, 66, 33, 0] * (33 if method == 'noise32' else 1)
            assert result['memory_available'] == (step > 0)
            assert result['previous_feedback_windows'] == step
            assert all(torch.isfinite(result[k]).all() for k in ('prediction', 'lower', 'upper', 'risk_half_width'))
            assert (result['upper'] >= result['lower']).all()
            expect_error(RuntimeError, lambda: model.predict(history))
            expect_error(ValueError, lambda: model.reveal(history[:, :, :10]))
            expect_error(ValueError, lambda: model.reveal(torch.full_like(history, float('nan'))))
            expected_error = (history - model.norm['mean']) / model.norm['std'] - model.pending
            model.reveal(history)
            assert torch.equal(model.previous_error, expected_error)
            expect_error(RuntimeError, lambda: model.predict(history + 1))
            expect_error(RuntimeError, lambda: model.reveal(history))
        model.reset()
        assert all(getattr(model, k) is None for k in ('pending', 'previous_error', 'previous_truth', 'calibrator'))
        print('PASS', method, args.device, flush=True)


if __name__ == '__main__':
    main()
