"""Train 129 XPF levels with 0.25% tails and frozen 19-level anchors."""
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from experiment_extended_train_error_adapters import LATE_RECORDS
from experiment_reliability_transport_e12 import FIT_RECORDS, normalization
from experiment_xpf16_quantiles19 import OrderedInterior, OUT as OLD, load
from selected16_quantile_head import BASE_INDICES
from temporal_mean_memory_control import training_data
from xpf_quantiles129_head import Quantiles129Head, quantile_levels


OUT = ROOT / 'runs/xpf_quantiles129_calibrated_20261111'
STEPS = 2000


@torch.no_grad()
def frozen_features(old, base, last):
    anchors = old(base, last)
    batch, _, _, height, width = base.shape
    latent = old.endpoint.feature_projection(last).reshape(batch, 8, 20, height, width)
    y = torch.linspace(-1, 1, height, device=base.device, dtype=base.dtype)
    x = torch.linspace(-1, 1, width, device=base.device, dtype=base.dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coordinates = torch.stack((xx, yy))[None, :, None].expand(batch, 2, 20, height, width)
    features = torch.cat((base[:, list(BASE_INDICES)], latent, coordinates), dim=1)
    return features, anchors


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    fit_ids = {r['file'] for r in FIT_RECORDS + LATE_RECORDS}
    held = {r['file'] for split, records in manifest()['files'].items() if split != 'train' for r in records}
    assert not fit_ids & held
    sources = [OLD / 'interior_quantiles.pt', ROOT / 'runs/selected16_quantile_20261110/compact16.pt']
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    dump(OUT / 'training_protocol.json', dict(source_sha256=digest(Path(__file__)),
        head_source_sha256=digest(ROOT / 'xpf_quantiles129_head.py'), frozen_sources=frozen,
        train_ids=sorted(fit_ids), training_pairs=819, steps=STEPS, batch=4, seed=42,
        scope='Freeze19 anchors; learn110 levels, including tails0.25..4.75 and95.25..99.75 step0.25 percent.',
        objective='Spacing-weighted mean pinball at110 new levels. Outer tails below0.25 and above99.75 not specified.',
        caution='Train trajectories disjoint from calibration/dev/test; historical dev/test reused, exploratory.'))
    norm = normalization()
    base, last, h, memory, residual = training_data(norm)
    del h, memory
    assert len(base) == 819
    old = OrderedInterior().cuda()
    old.gaps.load_state_dict(load(OLD / 'interior_quantiles.pt')['model'])
    old.eval().requires_grad_(False)
    torch.manual_seed(42)
    model = Quantiles129Head().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sampler = torch.Generator(device='cuda').manual_seed(42)
    tau = quantile_levels()
    anchor_indices = set(range(19, 110, 5))
    indices = [i for i in range(129) if i not in anchor_indices]
    assert len(indices) == 110
    which = torch.tensor(indices, device='cuda')
    levels = torch.tensor([tau[i] for i in indices], device='cuda').reshape(1, 1, 110, 1, 1, 1)
    spacing = torch.tensor([(tau[min(i + 1, 128)] - tau[max(i - 1, 0)]) / 2 for i in indices], device='cuda').reshape(1, 1, 110, 1, 1, 1)
    history_hash = hashlib.sha256()
    log = []
    for step in range(1, STEPS + 1):
        ix = torch.randint(819, (4,), device='cuda', generator=sampler)
        available = (torch.rand(4, device='cuda', generator=sampler) >= .2).float()
        history_hash.update(ix.cpu().numpy().tobytes())
        history_hash.update(available.cpu().numpy().tobytes())
        optimizer.param_groups[0]['lr'] = 1e-6 + (3e-4 - 1e-6) * .5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
        optimizer.zero_grad(set_to_none=True)
        features, anchors = frozen_features(old, base[ix], last[ix])
        prediction = model(features, anchors)
        error = residual[ix].unsqueeze(2) - prediction.index_select(2, which)
        loss = (torch.maximum(levels * error, (levels - 1) * error) * spacing).mean() / spacing.mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step == 1 or step % 250 == 0:
            row = dict(step=step, loss=float(loss), time=time.time())
            log.append(row)
            print('TRAIN129', row, flush=True)
    reference = json.loads((ROOT / 'runs/selected16_quantile_20261110/heads_frozen.json').read_text())
    assert history_hash.hexdigest() == reference['same_sampler_as_full28']['compact16']
    torch.save(dict(model={k: v.cpu() for k, v in model.state_dict().items()}, steps=STEPS, seed=42,
                    training_pairs=819, parameters=sum(p.numel() for p in model.parameters())), OUT / 'quantiles129.pt')
    dump(OUT / 'training.json', log)
    dump(OUT / 'training_complete.json', dict(passed=True, checkpoint_sha256=digest(OUT / 'quantiles129.pt'),
        frozen_inputs_unchanged=all(digest(ROOT / k) == v for k, v in frozen.items()),
        added_parameters=sum(p.numel() for p in model.parameters()), minibatches_match_original=True))
    print('TRAIN129_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
