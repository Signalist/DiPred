"""Prepare, check, train and evaluate the fixed observed residual experiment."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import torch

from reliability.backbone import configure
from reliability.networks import DynamicsDiffusion20
from research.observed_residual_feedback.core import (
    FIELD, MEMBERS, Forecast, Memory, Publication, field_metrics, objective, score_components,
)

OUT = ROOT / 'results/observed_residual_feedback'
SOURCE = Path(__file__).resolve().parent
REFERENCE = '5c971748381ca99f35643cea8b1c8c1c5106d612'
ARMS = ('L', 'J')
CANDIDATES = (0, 250, 500)
BUDGET = 3600


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def jsonl(path, row):
    with path.open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')


def old_score_repair(std):
    """Repair the archived new-forward V rows only, using their paired U fields."""
    old = ROOT / 'results/coupled_feedback_audit/window_metrics.csv'
    scales = np.asarray(std, dtype=float)
    index = {}
    with old.open() as handle:
        for row in csv.DictReader(handle):
            if row['estimator'] != 'V':
                continue
            assert row['prediction_sha256'] not in ('', 'reused')
            channel = row['channel']
            squared, linear = 1., 1.
            if channel in ('u', 'v'):
                scale = scales['uv'.index(channel)]
                squared = scale**2 / np.mean(scales**2)
                linear = scale / scales.mean()
            corrected = dict(row)
            for metric, factor in [('mse', squared), ('variance', squared), ('width90', linear)]:
                corrected[metric] = float(row[metric]) * factor
            assert np.isclose(corrected['mse'], float(row['smse']) *
                              (scales['uv'.index(channel)]**2 if channel in 'uv' else
                               corrected['mse'] / float(row['smse'])), atol=1e-12)
            identity = (row['trajectory'], row['start'], row['checkpoint'], channel)
            index[(*identity, row['method'])] = corrected
    pairs = []
    for key, on in index.items():
        if key[-1] != 'C_on':
            continue
        off = index[(*key[:-1], 'C_off')]
        row = {k: on[k] for k in ('trajectory', 'Re', 'start', 'checkpoint', 'channel')}
        row.update(on_prediction_sha256=on['prediction_sha256'], off_prediction_sha256=off['prediction_sha256'])
        for metric in ('crps_u', 'es_u', 'crps_v', 'es_v', 'mse', 'smse', 'variance', 'width90', 'coverage90'):
            row[metric] = float(on[metric]) - float(off[metric])
        for estimator in ('u', 'v'):
            row[f'score_{estimator}'] = row[f'crps_{estimator}'] + .1 * row[f'es_{estimator}']
        pairs.append(row)
    assert len(pairs) == 24 * 3 * 3
    with (OUT / 'old_audit_repaired.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(pairs)


def prepare(data_root):
    assert git('rev-parse', 'HEAD') == REFERENCE
    assert not (OUT / 'FROZEN.json').exists()
    base = read(ROOT / 'results/residual_dynamics/protocol.json')['records']
    manifest = read(ROOT / 'metadata/manifest.json')['files']
    assert {key: len(rows) for key, rows in base.items()} == dict(train=46, development=12, test=12)
    re_groups = [{row['group'] for row in rows} for rows in base.values()]
    assert all(not group & other for i, group in enumerate(re_groups) for other in re_groups[:i])
    windows = {}
    exclusions = []
    for split, records in base.items():
        windows[split] = []
        for record in records:
            assert record in manifest[split]
            file = record['file']
            with h5py.File(data_root / file, 'r') as handle:
                assert len(handle['u']) == len(handle['v']) == record['frames']
            starts = (200, 300, 400, 500) if split == 'train' else (400,)
            legal = [start for start in starts if start + 80 <= record['frames']]
            if legal:
                windows[split].append(dict(trajectory=file, Re=record['group'], starts=legal))
            exclusions.extend(dict(split=split, trajectory=file, start=start, frames=record['frames'])
                              for start in starts if start not in legal)
    assert len(windows['development']) == len(windows['test']) == 12
    dump(OUT / 'windows.json', dict(windows=windows, excluded=exclusions))
    rng = np.random.default_rng(42)
    schedule = {}
    for split, rows in windows.items():
        if split == 'train':
            schedule[split] = []
            for index in range(500):
                row = rows[int(rng.integers(len(rows)))]
                start = int(rng.choice(row['starts']))
                schedule[split].append(dict(trajectory=row['trajectory'], Re=row['Re'], start=start,
                    noise_seeds=[42000000 + index * 3 + offset for offset in range(3)]))
        else:
            root = 16042000 if split == 'development' else 32042000
            schedule[split] = [dict(trajectory=row['trajectory'], Re=row['Re'], start=400,
                noise_seeds=[root + index * 3 + offset for offset in range(3)]) for index, row in enumerate(rows)]
    seeds = [seed for rows in schedule.values() for row in rows for seed in row['noise_seeds']]
    assert len(seeds) == len(set(seeds))
    dump(OUT / 'schedule.json', schedule)
    dependencies = ('reliability/checkpoints/current.pt', 'reliability/networks.py',
                    'reliability/backbone.py', 'metadata/manifest.json', 'metadata/normalization.json',
                    'results/residual_dynamics/protocol.json',
                    'research/coupled_state_feedback/core.py', 'research/residual_dynamics_phase2/causal.py',
                    'results/residual_dynamics_stream/README.md', 'research/coupled_feedback_audit/run.py',
                    'results/coupled_feedback_audit/window_metrics.csv',
                    'results/coupled_feedback_audit/DECISION_REPORT.md')
    paths = [ROOT / name for name in dependencies] + [SOURCE / name for name in ('run.py', 'core.py', 'checks.py')]
    paths += [OUT / name for name in ('PROTOCOL.md', 'windows.json', 'schedule.json')]
    settings = dict(reference=REFERENCE, actual_head=git('rev-parse', 'HEAD'),
                    data_root=str(data_root.resolve()), seed=42, members=MEMBERS, arms=ARMS,
                    updates_per_arm=500, candidates=CANDIDATES, gpu_budget_seconds=BUDGET,
                    lr_backbone=1e-5, lr_new=1e-4, clip_grad_norm=1.,
                    selection='mean of warm final S_U by trajectory, earliest exact tie',
                    sha256={str(path.relative_to(ROOT)): sha(path) for path in paths})
    previous = read(ROOT / 'results/coupled_state_feedback/config.json')
    assert settings['sha256']['reliability/checkpoints/current.pt'] == previous['dependencies']['reliability/checkpoints/current.pt']
    dump(OUT / 'FROZEN.json', settings)
    old_score_repair(read(ROOT / 'metadata/normalization.json')['std'])
    print('Frozen development and test counts:', {key: len(rows) for key, rows in windows.items()}, flush=True)


def load():
    settings = read(OUT / 'FROZEN.json')
    for path, expected in settings['sha256'].items():
        assert sha(ROOT / path) == expected, path
    checkpoint = torch.load(ROOT / 'reliability/checkpoints/current.pt', map_location='cpu', weights_only=False)
    native = DynamicsDiffusion20(checkpoint['dynamics_scale'])
    native.load_state_dict(checkpoint['predictor'])
    norm = checkpoint['normalization']
    metadata = read(ROOT / 'metadata/normalization.json')
    for key in ('mean', 'std'):
        assert torch.equal(norm[key].flatten(), torch.tensor(metadata[key]))
        norm[key] = norm[key].reshape(2, 1, 1, 1).cuda()
    return native.requires_grad_(False).eval().cuda(), norm, settings


class Observations:
    def __init__(self, entry, norm, data_root):
        self.entry, self.norm, self.data_root = entry, norm, data_root

    def read_window(self, n):
        start = self.entry['start'] + 20 * n
        with h5py.File(Path(self.data_root) / self.entry['trajectory'], 'r') as handle:
            array = np.stack([handle[key][start:start + 20, ::2, ::2] for key in 'uv']).astype(np.float32)
        assert array.shape == FIELD and np.isfinite(array).all()
        return (torch.from_numpy(array).cuda() - self.norm['mean']) / self.norm['std']


def noise_for(history, seed):
    return torch.randn((MEMBERS, *FIELD), device=history.device, dtype=history.dtype,
                       generator=torch.Generator(device=history.device).manual_seed(seed))


def trainable(model):
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def frozen_tensors(model):
    state = model.state_dict()
    return {name: tensor_hash(value) for name, value in state.items() if name not in trainable(model)}


def optimize(model):
    parameters = trainable(model)
    backbone = [parameter for name, parameter in parameters.items() if name.startswith('base.')]
    new = [parameter for name, parameter in parameters.items() if not name.startswith('base.')]
    return torch.optim.Adam([dict(params=backbone, lr=1e-5), dict(params=new, lr=1e-4)])


def fit_update(model, optimizer, observation):
    optimizer.zero_grad(set_to_none=True)
    publication = Publication(observation.read_window(0))
    losses = []
    for n, seed in enumerate(observation.entry['noise_seeds'], 1):
        noise = noise_for(publication.history, seed)
        with torch.set_grad_enabled(n > 1):
            backbone, correction = publication.issue(model, noise)
            truth = observation.read_window(n)
            publication.reveal(truth)
            if n > 1:
                loss = objective(backbone, correction, truth)
                (loss / 2).backward()
                losses.append(loss.detach())
    grad = torch.nn.utils.clip_grad_norm_(list(trainable(model).values()), 1., error_if_nonfinite=True)
    optimizer.step()
    return [float(loss) for loss in losses], float(grad)


def checkpoint(model, step):
    assert step in CANDIDATES
    path = OUT / 'weights' / f'{model.arm}_{step}.pt'
    assert not path.exists()
    path.parent.mkdir(exist_ok=True)
    torch.save(dict(arm=model.arm, step=step, state={k: p.detach().cpu() for k, p in trainable(model).items()},
                    config_sha256=sha(OUT / 'FROZEN.json')), path)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def score_rows(paths, truth, norm):
    paths, truth = paths.double(), truth.double()
    rows = []
    for channel, position in [('overall', slice(None)), ('u', slice(0, 1)), ('v', slice(1, 2))]:
        samples, target, scale = paths[:, position], truth[position], norm['std'][position].double()
        absolute, pair, first, es_pair = score_components(samples, target)
        result = field_metrics(samples, target, scale)
        for lead, sl in [(0, slice(None))] + [(n, slice(n - 1, n)) for n in range(1, 21)]:
            values = dict(result) if lead == 0 else dict(
                mse=float(((samples.mean(0)-target)[:, sl] * scale).square().mean()),
                smse=float((samples.mean(0)-target)[:, sl].square().mean()),
                variance=float((samples.var(0, unbiased=False)[:, sl] * scale.square()).mean()))
            values.update(channel=channel, lead=lead)
            for estimator, denominator in [('u', MEMBERS * (MEMBERS - 1)), ('v', MEMBERS**2)]:
                values['crps_' + estimator] = float((absolute[:, sl] - pair[:, sl] / denominator).mean())
                if lead == 0:
                    values['es_' + estimator] = float(first - es_pair / denominator)
                    values['score_' + estimator] = values['crps_' + estimator] + .1 * values['es_' + estimator]
            rows.append(values)
    return rows


def write_scores(path, rows):
    # Deterministic compressed JSON lines, one member-derived metric record per group.
    with path.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')


@torch.no_grad()
def evaluate(models, aliases, split, stage, entries, norm, data_root):
    assert split in ('development', 'test')
    assert all(not model.training and not any(p.requires_grad for p in model.parameters())
               for model in models.values())
    path = OUT / f'{split}_{stage}.jsonl.gz'
    assert not path.exists()
    rows = []
    for entry in entries:
        obs = Observations(entry, norm, data_root)
        initial = obs.read_window(0)
        streams = {arm: Publication(initial) for arm in models}
        for n, seed in enumerate(entry['noise_seeds'], 1):
            noise = noise_for(initial, seed)
            common = dict(split=split, stage=stage, trajectory=entry['trajectory'], Re=entry['Re'],
                          start=entry['start'], window=n, seed=seed)
            issued = {}
            for arm, model in models.items():
                publication = streams[arm]
                backbone, correction = publication.issue(model, noise)
                issued[arm] = (backbone, correction)
                jsonl(OUT / 'publication.jsonl', dict(**common, arm=arm, event='issued', utc_ns=time.time_ns(),
                    history_sha256=tensor_hash(publication.history), noise_sha256=tensor_hash(noise),
                    availability=publication.memory.available, backbone_sha256=tensor_hash(backbone),
                    correction_sha256=tensor_hash(correction),
                    previous_residual_sha256=tensor_hash(publication.memory.residual),
                    previous_correction_sha256=tensor_hash(publication.memory.correction)))
            truth = obs.read_window(n)
            truth_sha = tensor_hash(truth)
            reveal_ns = time.time_ns()
            for arm, (backbone, correction) in issued.items():
                publication = streams[arm]
                publication.reveal(truth)
                jsonl(OUT / 'publication.jsonl', dict(**common, arm=arm, event='revealed', utc_ns=reveal_ns,
                    truth_sha256=truth_sha, residual_sha256=tensor_hash(publication.memory.residual),
                    mean_correction_sha256=tensor_hash(publication.memory.correction)))
                for field, values in [('backbone', backbone), ('final', backbone + correction)]:
                    # Duplicate scoring is unnecessary for the zero correction native reference.
                    if arm == 'N' and field == 'final':
                        metrics = backbone_metrics
                    else:
                        metrics = score_rows(values, truth, norm)
                    if field == 'backbone':
                        backbone_metrics = metrics
                    for label in (arm, *[alias for alias, source in aliases.items() if source == arm]):
                        rows.extend(dict(**common, arm=label, field=field, **metric) for metric in metrics)
    write_scores(path, rows)
    return rows


def snapshot(model):
    frozen_model = model
    frozen_model.eval().requires_grad_(False)
    return frozen_model


def restore(native, arm, step):
    if arm == 'N' or step == 0:
        return snapshot(Forecast(native, 'N'))
    model = Forecast(native, arm)
    path = OUT / 'weights' / f'{arm}_{step}.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert (state['arm'], state['step'], state['config_sha256']) == (arm, step, sha(OUT / 'FROZEN.json'))
    assert set(state['state']) == set(trainable(model))
    model.load_state_dict(state['state'], strict=False)
    return snapshot(model)


def selection(rows):
    values = [row['score_u'] for row in rows if row['channel'] == 'overall'
              and row['lead'] == 0 and row['window'] > 1 and row['field'] == 'final']
    assert len(values) == 24  # exactly two windows per each of 12 trajectories
    return float(np.mean(values))


def train():
    start = time.monotonic()
    assert not (OUT / 'TRAIN_STARTED.json').exists(), 'Formal run already started'
    preflight = read(OUT / 'PREFLIGHT.json')
    assert preflight['passed'] and preflight['config_sha256'] == sha(OUT / 'FROZEN.json')
    native, norm, settings = load()
    schedule = read(OUT / 'schedule.json')
    models = {arm: Forecast(native, arm).train() for arm in ARMS}
    initial = {name: tensor_hash(value) for name, value in trainable(models['L']).items()}
    assert initial == {name: tensor_hash(value) for name, value in trainable(models['J']).items()}
    checks = {arm: frozen_tensors(model) for arm, model in models.items()}
    optimizers = {arm: optimize(model) for arm, model in models.items()}
    paths = {arm: [checkpoint(model, 0)] for arm, model in models.items()}
    dump(OUT / 'TRAIN_STARTED.json', dict(utc_ns=time.time_ns(), config_sha256=sha(OUT / 'FROZEN.json'),
        parameters={arm: dict(trainable=sum(p.numel() for p in trainable(model).values()),
                             total=sum(p.numel() for p in model.parameters())) for arm, model in models.items()}))
    baseline = evaluate({'N': restore(native, 'N', 0)}, {'L': 'N', 'J': 'N'}, 'development', 0,
                        schedule['development'], norm, settings['data_root'])
    candidates = {arm: [(selection([row for row in baseline if row['arm'] == arm]), 0)] for arm in ARMS}
    common = 0
    slowest_pair = 1.
    for step, entry in enumerate(schedule['train'], 1):
        elapsed = preflight['seconds'] + time.monotonic() - start
        if elapsed + 120 + 2 * slowest_pair >= BUDGET:
            break
        pair_start = time.monotonic()
        for arm in ARMS:
            losses, grad = fit_update(models[arm], optimizers[arm], Observations(entry, norm, settings['data_root']))
            jsonl(OUT / 'training.jsonl', dict(step=step, arm=arm, warm_losses=losses,
                                             gradient_norm=grad, utc_ns=time.time_ns(), **entry))
        common = step
        slowest_pair = max(slowest_pair, time.monotonic() - pair_start)
        if step % 25 == 0:
            print('Common updates:', step, 'elapsed seconds:', round(elapsed, 1), flush=True)
        if step in CANDIDATES:
            for arm, model in models.items():
                paths[arm].append(checkpoint(model, step))
            rows = evaluate({arm: restore(native, arm, step) for arm in ARMS}, {}, 'development', step,
                            schedule['development'], norm, settings['data_root'])
            for arm in ARMS:
                candidates[arm].append((selection([row for row in rows if row['arm'] == arm]), step))
    changes = {}
    for arm, model in models.items():
        assert checks[arm] == frozen_tensors(model)
        changes[arm] = [name for name, value in trainable(model).items() if tensor_hash(value) != initial[name]]
        if common:
            for prefix in ('base.net.up2', 'base.net.out', 'encoder', 'projection', 'head'):
                assert any(name.startswith(prefix) for name in changes[arm])
    selected = {arm: min(choices)[1] for arm, choices in candidates.items()}
    dump(OUT / 'SELECTION.json', dict(utc_ns=time.time_ns(), selected=selected, candidates=candidates,
        config_sha256=sha(OUT / 'FROZEN.json')))
    evaluate(dict(N=restore(native, 'N', 0), **{arm: restore(native, arm, step)
                                               for arm, step in selected.items() if step}),
             {arm: 'N' for arm, step in selected.items() if not step},
             'test', 'selected', schedule['test'], norm, settings['data_root'])
    torch.cuda.synchronize()
    spent = preflight['seconds'] + time.monotonic() - start
    dump(OUT / 'RUN.json', dict(common_updates=common, stop='completed' if common == 500 else 'budget guard',
        seconds=spent, budget_met=spent <= BUDGET, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        weights=paths, frozen_backbone_unchanged=True, changed_parameters=changes,
        native_calls_per_forecast=4, evaluation_prediction_tensor_cache_bytes=0))
    print('Completed', common, 'updates per arm, selected', selected, 'seconds', spent, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'preflight', 'train', 'report'))
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    configure()
    if args.stage == 'prepare':
        if args.data is None:
            parser.error('prepare requires --data')
        prepare(args.data)
    elif args.stage == 'preflight':
        from research.observed_residual_feedback.checks import preflight
        preflight()
    elif args.stage == 'train':
        train()
    else:
        from research.observed_residual_feedback.reporting import report
        report()


if __name__ == '__main__':
    # When launched as a script, use this module for checks without re-importing it.
    sys.modules['research.observed_residual_feedback.run'] = sys.modules[__name__]
    main()
