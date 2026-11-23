"""Issue native windows before disclosure and store only compact ensemble fields."""
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from research.observed_residual_feedback.core import Forecast, Publication
from research.observed_residual_feedback.run import Observations, noise_for, sha
from research.residual_information_study.information_probe.probe import FIELD, features


ARRAYS = {'observed': (4, *FIELD), 'mean': (3, *FIELD), 'std': (3, *FIELD),
          'first8': (3, *FIELD), 'last8': (3, *FIELD)}


def as_array(value):
    return value.detach().cpu().contiguous().numpy()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def feature_matrices(arrays, window, index=slice(None)):
    known = arrays['observed']
    means = arrays['mean']
    previous = known[window - 1]
    prev_slot = window - 2
    earlier = known[window - 2]
    common = (previous, means[window - 1], arrays['std'][window - 1])
    full_residual = previous - means[prev_slot]
    first_residual = previous - arrays['first8'][prev_slot]
    last_residual = previous - arrays['last8'][prev_slot]
    matrices = {label: features(*common, residual, earlier, index) for label, residual in
                [('full', full_residual), ('first8', first_residual), ('last8', last_residual)]}
    return matrices, {'split_input_mse': float(np.mean((first_residual.astype(np.float64) - last_residual)**2)),
                      'full_residual_mse': float(np.mean(full_residual.astype(np.float64)**2)),
                      'half_average_identity_max_abs': float(np.max(np.abs(
                          ((first_residual.astype(np.float64) + last_residual) / 2) - full_residual)))}


def previous_native_events(path, split):
    """Select only prior original native release identities for this split."""
    assert split in ('development', 'test')
    wanted_stage = 0 if split == 'development' else 'selected'
    events = {}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if f'"split":"{split}"' not in line or '"arm":"N"' not in line:
                continue
            event = json.loads(line)
            if event['stage'] == wanted_stage:
                key = (event['trajectory'], event['start'], event['window'], event['event'])
                assert key not in events
                events[key] = event
    assert len(events) == 72
    return events


class NativeCache:
    def __init__(self, native, norm, root, data_root, cache_root, frozen_sha):
        self.model = Forecast(native, 'N').eval().requires_grad_(False)
        self.norm = norm
        self.data_root = Path(data_root)
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_sha = sha(root / 'reliability/checkpoints/current.pt')
        self.frozen_sha = frozen_sha
        self.generated_groups = 0
        self.reused_groups = 0

    def case(self, split, index, entry, predict=None, archived=None):
        old = archived or {}
        identity = dict(split=split, index=index, entry=entry, model='native N', members=16,
                        checkpoint_sha256=self.checkpoint_sha, frozen_sha256=self.frozen_sha,
                        std_ddof=0)
        path = self.cache_root / f'{split}_{index:03d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as cache:
                receipt = json.loads(cache['receipt'].tobytes())
                assert receipt['identity'] == identity, (path, 'wrong native source or schedule')
                arrays = {name: cache[name] for name in ARRAYS}
            issued_predictions = {}
            self.reused_groups += 3
            if predict is not None:
                for window in (2, 3):
                    matrices, stability = feature_matrices(arrays, window)
                    estimates = predict(matrices)
                    assert prediction_hashes(estimates) == receipt['windows'][window - 1]['issued']['probe_sha256']
                    issued_predictions[window] = (estimates, stability)
        else:
            arrays, receipt, issued_predictions = self.generate(identity, predict)
            path_tmp = path.with_suffix('.npz.tmp')
            with path_tmp.open('wb') as stream:
                np.savez(stream, **arrays, receipt=np.frombuffer(json.dumps(receipt).encode(), dtype=np.uint8))
            path_tmp.replace(path)
            self.generated_groups += 3
        validate_case(arrays, receipt)
        receipt['archive_identity'] = compare_archive(receipt, old) if archived is not None else None
        receipt['cache_sha256'] = sha(path)
        receipt['cache_bytes'] = path.stat().st_size
        return arrays, receipt, issued_predictions

    @torch.inference_mode()
    def generate(self, identity, predict):
        entry = identity['entry']
        obs = Observations(entry, self.norm, self.data_root)
        arrays = {key: np.empty(shape, dtype=np.float32) for key, shape in ARRAYS.items()}
        first_history = obs.read_window(0)
        arrays['observed'][0] = as_array(first_history)
        publisher = Publication(first_history)
        windows = []
        predictions = {}
        for window, seed in enumerate(entry['noise_seeds'], 1):
            noise = noise_for(publisher.history, seed)
            paths, correction = publisher.issue(self.model, noise)
            assert len(paths) == 16 and not torch.count_nonzero(correction)
            slot = window - 1
            arrays['mean'][slot] = as_array(paths.mean(dim=0))
            arrays['std'][slot] = as_array(paths.std(dim=0, unbiased=False))
            arrays['first8'][slot] = as_array(paths[:8].mean(dim=0))
            arrays['last8'][slot] = as_array(paths[8:].mean(dim=0))
            issued = dict(history_sha256=array_hash(as_array(publisher.history)), noise_sha256=array_hash(as_array(noise)),
                          backbone_sha256=array_hash(as_array(paths)), correction_sha256=array_hash(as_array(correction)),
                          previous_residual_sha256=array_hash(as_array(publisher.memory.residual)),
                          mean_sha256=array_hash(arrays['mean'][slot]), std_sha256=array_hash(arrays['std'][slot]),
                          first8_sha256=array_hash(arrays['first8'][slot]), last8_sha256=array_hash(arrays['last8'][slot]))
            if predict is not None and window > 1:
                matrices, stability = feature_matrices(arrays, window)
                estimates = predict(matrices)
                predictions[window] = (estimates, stability)
                issued['probe_sha256'] = prediction_hashes(estimates)
            issued['utc_ns'] = time.time_ns()
            current = obs.read_window(window)
            reveal_time = time.time_ns()
            publisher.reveal(current)
            arrays['observed'][window] = as_array(current)
            revealed = dict(utc_ns=reveal_time, truth_sha256=array_hash(arrays['observed'][window]),
                            residual_sha256=array_hash(as_array(publisher.memory.residual)))
            windows.append(dict(window=window, seed=seed, issued=issued, revealed=revealed))
        return arrays, dict(identity=identity, windows=windows), predictions


def prediction_hashes(estimates):
    return {name: array_hash(value) for name, value in estimates.items()}


def validate_case(arrays, receipt):
    assert all(arrays[key].shape == shape and arrays[key].dtype == np.float32
               and np.isfinite(arrays[key]).all() for key, shape in ARRAYS.items())
    assert (arrays['std'] >= 0.).all()
    known, means = arrays['observed'], arrays['mean']
    assert len(receipt['windows']) == 3
    previous = np.zeros_like(known[0])
    for slot, window in enumerate(receipt['windows']):
        issued, reveal = window['issued'], window['revealed']
        assert window['window'] == slot + 1
        assert window['seed'] == receipt['identity']['entry']['noise_seeds'][slot]
        assert issued['utc_ns'] < reveal['utc_ns']
        if slot > 0:
            assert receipt['windows'][slot - 1]['revealed']['utc_ns'] < issued['utc_ns']
        assert issued['history_sha256'] == array_hash(known[slot])
        assert issued['previous_residual_sha256'] == array_hash(previous)
        for name in ('mean', 'std', 'first8', 'last8'):
            assert issued[f'{name}_sha256'] == array_hash(arrays[name][slot])
        assert reveal['truth_sha256'] == array_hash(known[slot + 1])
        previous = known[slot + 1] - means[slot]
        assert reveal['residual_sha256'] == array_hash(previous)


def compare_archive(receipt, old):
    entry = receipt['identity']['entry']
    native_matches = []
    for window in receipt['windows']:
        key = (entry['trajectory'], entry['start'], window['window'])
        archive_issue = old[(*key, 'issued')]
        archive_reveal = old[(*key, 'revealed')]
        assert archive_issue['seed'] == window['seed'] == archive_reveal['seed']
        for name in ('history_sha256', 'noise_sha256', 'correction_sha256'):
            assert window['issued'][name] == archive_issue[name], (key, name)
        assert window['revealed']['truth_sha256'] == archive_reveal['truth_sha256'], key
        assert archive_issue['utc_ns'] < archive_reveal['utc_ns']
        native_matches.append(archive_issue['backbone_sha256'] == window['issued']['backbone_sha256'])
    return native_matches
