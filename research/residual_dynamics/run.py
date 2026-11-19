"""Frozen DiPred signed residual dynamics: alignment audit, cache, A/B/C.

Run from repository root; see README.md beside this file. No network access.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reliability.backbone import configure
from reliability.networks import DynamicsDiffusion20


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def table(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def raw(data, records, start):
    values = []
    for r in records:
        with h5py.File(data / r['file'], 'r') as f:
            a = np.stack([f[c][start:start+20, ::2, ::2] for c in ('u', 'v')]).astype('float32')
        assert a.shape == (2, 20, 32, 64) and np.isfinite(a).all()
        values.append(torch.from_numpy(a))
    return torch.stack(values).cuda()


def load_backbone(args):
    p = REPO / 'reliability/checkpoints/current.pt'
    ck = torch.load(p, map_location='cuda', weights_only=False)
    model = DynamicsDiffusion20(ck['dynamics_scale']).cuda().eval().requires_grad_(False)
    model.load_state_dict(ck['predictor'], strict=True)
    original = args.workspace / 'runs/physical20_augmented_dynamics_20261109/augmented_diffusion/model.pt'
    old = torch.load(original, map_location='cuda', weights_only=False)['model']
    assert old.keys() == ck['predictor'].keys()
    assert all(torch.equal(old[k], ck['predictor'][k]) for k in old)
    norm = {k: v.cuda() for k, v in ck['normalization'].items()}
    return model, norm, dict(packaged_checkpoint_sha256=sha(p), original_checkpoint_sha256=sha(original),
                            predictor_tensors_exact=True)


@torch.no_grad()
def audit(args):
    """Any uncertain frame/member alignment fails before affected training."""
    model, norm, frozen = load_backbone(args)
    source = args.workspace / 'runs/compact_test_distribution_20261111'
    rows = list(csv.DictReader((source / 'snapshot_index.csv').open()))
    y = np.load(source / 'snapshot_Y.npy')
    samples = np.load(source / 'snapshot_diffusion_samples32.npy')
    assert y.shape == (3, 2, 20, 32, 64)
    assert samples.shape == (3, 32, 2, 20, 32, 64)
    records = json.loads(args.manifest.read_text())['files']['test']
    checks = []
    for i, r in enumerate(rows):
        start, target = int(r['history_start']), int(r['target_start'])
        assert int(r['snapshot']) == i and target == start + 20
        # Original sampler noise depends on the complete ordered batch shape.
        j = int(r['global_index']) % len(records)
        assert records[j]['file'] == r['trajectory_id']
        truth = raw(args.data, records, target)
        assert np.array_equal(truth[j].cpu().numpy(), y[i])
        h = (raw(args.data, records, start) - norm['mean']) / norm['std']
        maxerr = 0.
        for member, seed in enumerate(range(500000, 500032)):
            p = model.rollout(h, seed, sign=1) * norm['std'] + norm['mean']
            error = float(np.max(np.abs(p[j].cpu().numpy() - samples[i, member])))
            maxerr = max(maxerr, error)
            # Numerical tolerance, not a statistical member-matching heuristic.
            assert error <= 2e-6, (r, member, error)
        e = samples[i] - y[i][None]
        assert e[:, :, :-1].shape[2] == e[:, :, 1:].shape[2] == 19
        checks.append(dict(trajectory=r['trajectory_id'], history_indices=[start, start+19],
                           prediction_truth_indices=[target, target+19],
                           paired_indices=[[target+t, target+t+1] for t in range(19)],
                           all_32_members_all_20_frames_max_abs_error=maxerr,
                           truth_exact=True, members=list(range(500000, 500032))))
        print('AUDIT', r['trajectory_id'], maxerr, flush=True)
    dump(args.out / 'alignment.json', dict(passed=True, frozen=frozen, snapshots=checks,
         source_files={str(source / n): sha(source / n) for n in
                       ('snapshot_index.csv', 'snapshot_Y.npy', 'snapshot_diffusion_samples32.npy')},
         rule='e=prediction-truth; member seed labels the entire jointly generated 20-frame path; no sorting'))


@torch.no_grad()
def cache(args):
    assert json.loads((args.out / 'alignment.json').read_text())['passed']
    model, norm, frozen = load_backbone(args)
    manifest = json.loads(args.manifest.read_text())
    starts = [400, 420]
    all_records, exclusions = {}, []
    for split in ('train', 'development', 'test'):
        all_records[split] = [r for r in manifest['files'][split] if r['frames'] >= max(starts)+40]
        exclusions += [dict(split=split, **r) for r in manifest['files'][split] if r not in all_records[split]]
    ids = [{r['file'] for r in rs} for rs in all_records.values()]
    assert not any(ids[i] & ids[j] for i in range(3) for j in range(i))
    protocol = dict(starts=starts, member_seeds=list(range(500000, 500032)), records=all_records,
        exclusions=exclusions, manifest_sha256=sha(args.manifest), data_root=str(args.data), frozen=frozen,
        residual='prediction minus truth, original data scale',
        grouping='original HDF5 file; all windows and all members stay on one side',
        snapshot_usage='alignment smoke test only; starts 0/100/200 never enter fitting or scoring')
    dump(args.out / 'protocol.json', protocol)
    args.cache.mkdir(parents=True, exist_ok=True)
    entries = []
    for split, records in all_records.items():
        for start in starts:
            dest = args.cache / f'{split}_{start}.pt'
            identity = dict(split=split, start=start, files=[r['file'] for r in records],
                            seeds=protocol['member_seeds'], frozen=frozen)
            if dest.exists():
                value = torch.load(dest, map_location='cpu', weights_only=False)
                assert value['identity'] == identity
                print('REUSE', dest, flush=True)
            else:
                observed, truth = raw(args.data, records, start), raw(args.data, records, start+20)
                h = (observed - norm['mean']) / norm['std']
                paths = []
                for seed in protocol['member_seeds']:
                    p = model.rollout(h, seed, sign=1) * norm['std'] + norm['mean']
                    paths.append((p - truth).cpu())
                # [trajectory, member, channel, frame, y, x]
                e = torch.stack(paths).transpose(0, 1).contiguous()
                assert e.shape == (len(records), 32, 2, 20, 32, 64) and torch.isfinite(e).all()
                value = dict(identity=identity, residual=e,
                    history_sha256=hashlib.sha256(observed.cpu().numpy().tobytes()).hexdigest(),
                    truth_sha256=hashlib.sha256(truth.cpu().numpy().tobytes()).hexdigest())
                torch.save(value, dest)
                print('CACHE', split, start, list(e.shape), flush=True)
            entries.append(dict(path=str(dest), sha256=sha(dest), identity=identity,
                                history_sha256=value['history_sha256'], truth_sha256=value['truth_sha256']))
    dump(args.out / 'cache_manifest.json', entries)


class ResidualPredictor(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.net = nn.Sequential(nn.Conv2d(3 if arm == 'A' else 5, 32, 1), nn.SiLU(),
                                 nn.Conv2d(32, 32, 1), nn.SiLU(), nn.Conv2d(32, 2, 1))
        if arm != 'A':
            k = 1 if arm == 'B' else 3
            self.mix = nn.Conv2d(2, 2, k, groups=2, bias=False)
            nn.init.zeros_(self.mix.weight)

    def forward(self, previous, lead):
        n, _, h, w = previous.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=previous.device),
                                torch.linspace(-1, 1, w, device=previous.device), indexing='ij')
        coords = torch.stack([xx, yy])[None].expand(n, -1, -1, -1)
        step = ((lead.float()-2)/18 * 2-1)[:, None, None, None].expand(n, 1, h, w)
        x = torch.cat([coords, step], 1)
        if self.arm == 'A':
            return self.net(x)
        # Direct local skip retains amplitude freedom in both arms.
        mixed = previous + self.mix(F.pad(previous, (1, 1, 1, 1), mode='replicate')
                                   if self.arm == 'C' else previous)
        return previous + self.net(torch.cat([x, mixed], 1))


def dataset(args, split):
    arrays = []
    for start in (400, 420):
        value = torch.load(args.cache / f'{split}_{start}.pt', map_location='cpu', weights_only=False)
        arrays.append(value['residual'])
    # [trajectory, window, member, frame, channel, y, x]
    return torch.stack(arrays, 1).permute(0, 1, 2, 4, 3, 5, 6).contiguous().cuda()


@torch.no_grad()
def evaluate(model, data, scale, shuffle_seed=None):
    """Score each trajectory separately; shuffle whole member fields per window/time."""
    rows = []
    generator = torch.Generator(device='cuda').manual_seed(shuffle_seed or 0)
    for trajectory in data:
        sums = torch.zeros(2, device='cuda', dtype=torch.float64)
        for window in trajectory:
            for t in range(19):
                previous, target = window[:, t], window[:, t+1]
                if shuffle_seed is not None:
                    previous = previous[torch.randperm(32, generator=generator, device='cuda')]
                estimate = model(previous / scale, torch.full((32,), t+2, device='cuda')) * scale
                sums += (estimate - target).double().square().mean((0, 2, 3))
        rows.append((sums / (data.shape[1]*19)).cpu().numpy())
    return np.array(rows)


def integrity_checks():
    """Verify information boundary by perturbing a neighboring grid cell."""
    torch.manual_seed(123)
    x = torch.randn(1, 2, 32, 64, device='cuda')
    lead = torch.tensor([5], device='cuda')
    effects = {}
    for arm in 'ABC':
        m = ResidualPredictor(arm).cuda()
        if arm == 'C':
            with torch.no_grad():
                m.mix.weight.fill_(.1)
        changed = x.clone()
        changed[:, :, 10, 11] += 10
        delta = float((m(x, lead)[:, :, 10, 10] - m(changed, lead)[:, :, 10, 10]).abs().max())
        assert (delta == 0) if arm in 'AB' else (delta > 0)
        if arm == 'A':
            assert torch.equal(m(x, lead), m(torch.zeros_like(x), lead))
        effects[arm] = delta
    return effects


def fit(args):
    assert json.loads((args.out / 'alignment.json').read_text())['passed']
    checks = integrity_checks()
    train, dev = dataset(args, 'train'), dataset(args, 'development')
    scale = float(train.square().mean().sqrt())  # single train-only scalar; preserves raw-MSE objective
    assert scale > 0
    log, counts, bests = [], {}, {}
    for arm in 'ABC':
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = ResidualPredictor(arm).cuda()
        counts[arm] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        rng = torch.Generator(device='cuda').manual_seed(20261119)
        best = float('inf')
        for step in range(1, args.steps+1):
            tr = torch.randint(len(train), (16,), device='cuda', generator=rng)
            wi = torch.randint(2, (16,), device='cuda', generator=rng)
            me = torch.randint(32, (16,), device='cuda', generator=rng)
            ti = torch.randint(19, (16,), device='cuda', generator=rng)
            previous, target = train[tr, wi, me, ti], train[tr, wi, me, ti+1]
            prediction = model(previous/scale, ti+2)
            loss = (prediction - target/scale).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 250 == 0 or step == args.steps:
                model.eval()
                mse = float(evaluate(model, dev, scale).mean())
                log.append(dict(arm=arm, step=step, batch_loss=float(loss), development_mse=mse))
                if mse < best:
                    best = mse
                    bests[arm] = step
                    torch.save(dict(arm=arm, model=model.state_dict(), scale=scale, step=step), args.out / f'{arm}.pt')
                print('FIT', arm, step, mse, flush=True)
                model.train()
    dump(args.out / 'training.json', dict(seed=42, batch_seed=20261119, steps=args.steps, batch_size=16,
        optimizer='Adam lr=.001', selection='lowest trajectory-macro development MSE, every250steps plus endpoint',
        scalar_train_rms=scale, parameters=counts, selected_steps=bests,
        neighbor_perturbation_effect=checks, log=log))


def paired_ci(values, seed=123):
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def score(args):
    protocol = json.loads((args.out / 'protocol.json').read_text())
    rows, summaries, contrasts = [], [], []
    for split in ('development', 'test'):
        data = dataset(args, split)
        results = {}
        for arm in 'ABC':
            ck = torch.load(args.out / f'{arm}.pt', map_location='cuda', weights_only=False)
            m = ResidualPredictor(arm).cuda().eval()
            m.load_state_dict(ck['model'])
            conditions = ['aligned'] if arm == 'A' else ['aligned', 'member_shuffled']
            for condition in conditions:
                # Ten fixed permutations; same donor fields for B/C, targets unchanged.
                trials = [evaluate(m, data, ck['scale'], s) for s in range(7100, 7110)] if condition != 'aligned' else [evaluate(m, data, ck['scale'])]
                values = np.mean(trials, axis=0)
                results[arm, condition] = values.mean(1)
                for i, r in enumerate(protocol['records'][split]):
                    rows.append(dict(split=split, trajectory=r['file'], arm=arm, condition=condition,
                        mse=float(values[i].mean()), mse_u=float(values[i, 0]), mse_v=float(values[i, 1]),
                        windows=2, members=32, adjacent_pairs_per_member_per_window=19))
                summaries.append(dict(split=split, arm=arm, condition=condition, trajectories=len(data),
                    mse=float(values.mean()), mse_u=float(values[:, 0].mean()), mse_v=float(values[:, 1].mean()),
                    shuffle_trials=len(trials), trial_macro_mse=[float(v.mean()) for v in trials]))
                print('SCORE', split, arm, condition, float(values.mean()), flush=True)
        for a, ac, b, bc, name in [('B','aligned','A','aligned','B-A'), ('C','aligned','B','aligned','C-B'),
                                  ('B','member_shuffled','B','aligned','B_shuffle-B'),
                                  ('C','member_shuffled','C','aligned','C_shuffle-C')]:
            delta = results[a, ac] - results[b, bc]
            contrasts.append(dict(split=split, contrast=name, mean_delta_mse=float(delta.mean()),
                trajectory_bootstrap_95=paired_ci(delta), trajectories_improved=int((delta<0).sum()),
                trajectories=len(delta)))
        del data
    table(args.out / 'per_trajectory.csv', rows)
    dump(args.out / 'summary.json', summaries)
    dump(args.out / 'contrasts.json', contrasts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['audit', 'cache', 'fit', 'score', 'all'])
    p.add_argument('--workspace', type=Path, default=REPO.parents[1])
    p.add_argument('--data', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--out', type=Path, default=REPO / 'results/residual_dynamics')
    p.add_argument('--steps', type=int, default=2000)
    args = p.parse_args()
    args.data = args.data or args.workspace.parent / 'OLD/REALPDE/realpdebench-competition/data/train_real'
    args.manifest = args.workspace / 'runs/clean_direct20_20261107/manifest.json'
    args.cache = args.cache or args.workspace / 'runs/residual_dynamics_20261119'
    args.out.mkdir(parents=True, exist_ok=True)
    configure()
    for stage in ('audit', 'cache', 'fit', 'score') if args.stage == 'all' else (args.stage,):
        globals()[stage](args)


if __name__ == '__main__':
    main()
