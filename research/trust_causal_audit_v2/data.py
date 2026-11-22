"""Keyed causal preparation, independent of current-future labels.

All intervals are zero-based half-open. Cache row lookup uses the identity's
file list, never a prepared table's row number. No cache generation is allowed.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
from .common import MEMBERS, tensor_hash


@dataclass(frozen=True)
class Key:
    split: str
    trajectory_id: str
    start: int


@dataclass
class Issued:
    key: Key
    paths: torch.Tensor  # [32,2,20,H,W], physical values
    history_interval: Tuple[int, int]
    target_interval: Tuple[int, int]
    member_ids: tuple
    cache_row: int
    source: str

    def validate(self):
        assert self.history_interval == (self.key.start, self.key.start + 20)
        assert self.target_interval == (self.key.start + 20, self.key.start + 40)
        assert self.member_ids == MEMBERS
        assert self.paths.shape[:3] == (32, 2, 20)
        assert torch.isfinite(self.paths).all()
        return self


class CacheIndex:
    def __init__(self, payload, split, start, files, source):
        identity = payload['identity']
        assert identity['split'] == split and identity['start'] == start
        assert identity['files'] == list(files), 'cache file order differs from locked split'
        ids = identity.get('members', identity.get('seeds'))
        assert ids == list(MEMBERS), 'member identity mismatch'
        self.paths = payload['paths']
        assert self.paths.shape == (len(files), 32, 2, 20, 32, 64)
        assert torch.isfinite(self.paths).all()
        self.files = tuple(files)
        assert len(set(self.files)) == len(self.files)
        self.split, self.start, self.source = split, start, source

    def lookup(self, key):
        assert key.split == self.split and key.start == self.start
        index = self.files.index(key.trajectory_id)
        return Issued(key, self.paths[index], (key.start, key.start+20),
                      (key.start+20, key.start+40), MEMBERS, index, self.source).validate()


def window_catalog(split, files, lengths, starts):
    """Each window owns its valid list, derived from actual data lengths."""
    rows = []
    for start in starts:
        for original_index, name in enumerate(files):
            if lengths[name] >= start + 40:
                rows.append(dict(key=Key(split, name, start), original_index=original_index,
                                 frames=lengths[name]))
    return rows


def raw_risk(history, previous, std):
    previous.validate()
    assert history.shape == previous.paths.shape[1:]
    # Prediction target of the previous release equals *observed* current H.
    error = (previous.paths.mean(0) - history) / std.reshape(2, 1, 1, 1)
    return error.square().mean((0, 1)).unsqueeze(0)


def fit_scale(raw_maps, split):
    assert split == 'train', 'risk scale must be frozen using training only'
    values = torch.stack(raw_maps).flatten()
    q = float(torch.quantile(torch.log1p(values), .99).clamp_min(1e-6))
    return dict(log1p_p99=q, quantile=.99, clip=[0., 1.], split='train',
                maps=len(raw_maps), source='previous_prediction_minus_current_observed_history')


def prepare_inputs(reader, key, previous, norm, scale):
    """Real publication entry: reader is never asked for [s+20,s+40)."""
    assert previous.key == Key(key.split, key.trajectory_id, key.start - 20)
    assert previous.target_interval == (key.start, key.start+20)
    history = reader(key.trajectory_id, key.start, key.start+20)
    assert history.shape == (2, 20, 32, 64) and torch.isfinite(history).all()
    mean, std = norm['mean'].reshape(2,1,1,1), norm['std'].reshape(2,1,1,1)
    R = raw_risk(history, previous, std)
    risk = (torch.log1p(R) / scale['log1p_p99']).clamp(0, 1)
    return dict(key=key, history=(history-mean)/std,
                previous_mean=(previous.paths.mean(0)-mean)/std,
                raw_risk=R, risk=risk, history_sha256=tensor_hash(history),
                previous_path_sha256=tensor_hash(previous.paths), member_ids=MEMBERS,
                observed_interval=(key.start,key.start+20),
                feedback_interval=previous.target_interval, target_interval=(key.start+20,key.start+40))


def model_inputs(prepared, noise):
    assert prepared['member_ids'] == MEMBERS
    assert noise.shape == (32, 2, 20, 32, 64)
    device, dtype = noise.device, noise.dtype
    def expand(name):
        return prepared[name].to(device=device,dtype=dtype)[None].expand(32, *prepared[name].shape)
    return dict(history=expand('history'), previous_mean=expand('previous_mean'),
                risk=expand('risk'), noise=noise)


def batch_noise(key, files, member_ids=MEMBERS, device='cuda', member_slice=None):
    """Reconstruct the *original CUDA batch draw* before selecting one trajectory.

    Old RNG identity was seed + complete split batch shape/order, not just seed.
    Reusing that exact draw permits strict pairing with the old native cache.
    We record this convention explicitly rather than asserting independent keys.
    """
    files = tuple(files)
    index = files.index(key.trajectory_id)
    shape = (len(files), 2, 20, 32, 64)
    chosen = member_ids if member_slice is None else member_ids[member_slice]
    chunks, identities = [], []
    for seed in chosen:
        generator = torch.Generator(device=device).manual_seed(seed)
        full = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        value = full[index]
        chunks.append(value)
        identities.append(dict(split=key.split, trajectory_id=key.trajectory_id, history_start=key.start,
             member_id=seed, noise_seed=seed, noise_tensor_hash=tensor_hash(value),
             generator_device=str(device), batch_shape=list(shape), batch_row=index,
             batch_files=list(files), convention='original_full_split_batch_draw_then_identity_slice'))
    return torch.stack(chunks), identities


class H5Reader:
    def __init__(self, root):
        self.root = root
        self.reads = []

    def __call__(self, name, begin, end):
        import h5py
        self.reads.append(dict(trajectory_id=name, interval=[begin,end]))
        with h5py.File(self.root / name, 'r') as f:
            value = np.stack([f[c][begin:end, ::2, ::2] for c in ('u','v')]).astype('float32')
        assert value.shape == (2, end-begin, 32, 64) and np.isfinite(value).all()
        return torch.from_numpy(value)
