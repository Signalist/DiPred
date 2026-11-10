"""Independent training/control audit for the physically pruned16-feature head."""
import hashlib
import json
from collections import defaultdict
from statistics import mean
import torch

from clean_reliability_common import ROOT, configure, dump, digest, manifest
from experiment_selected16_quantile import OUT, SOURCE, ABLATION, MEMORY, ARMS, NAMES, STEPS, new_head
from experiment_fullblock_memory_risk import load
from experiment_reliability_transport_e12 import normalization
from temporal_mean_memory_control import training_data
from selected16_quantile_head import CORE_INDICES, BASE_INDICES
from experiment_global_operator_risk import METRICS


def read(path):
    return json.loads(path.read_text())


@torch.no_grad()
def independent_compact_bounds(head, base, last):
    batch, _, _, height, width = base.shape
    projected = torch.nn.functional.conv2d(last, head.feature_projection.weight, head.feature_projection.bias)
    features = projected.reshape(batch, 8, 20, height, width)
    yy = torch.linspace(-1, 1, height, device=base.device, dtype=base.dtype)
    xx = torch.linspace(-1, 1, width, device=base.device, dtype=base.dtype)
    y, x = torch.meshgrid(yy, xx, indexing='ij')
    coordinates = torch.stack([x, y])[None, :, None].expand(batch, 2, 20, height, width)
    value = torch.cat([base[:, [4, 5]], base[:, [8, 9]], base[:, [10, 11]], features, coordinates], dim=1)
    assert value.shape[1:] == (16, 20, height, width)
    result = head.net(value)
    radius = torch.nn.functional.softplus(result[:, 2:]) + 1e-4
    return result[:, :2] - radius, result[:, :2] + radius


def main():
    configure()
    assert (OUT / 'complete.json').exists()
    protocol, frozen = read(OUT / 'protocol.json'), read(OUT / 'heads_frozen.json')
    assert protocol['source_sha256'] == digest(ROOT / 'experiment_selected16_quantile.py')
    assert protocol['module_sha256'] == digest(ROOT / 'selected16_quantile_head.py')
    assert all(digest(ROOT / path) == value for path, value in protocol['frozen_artifacts'].items())
    held = {r['file'] for split, rows in manifest()['files'].items() if split != 'train' for r in rows}
    assert not set(protocol['fit_files']) & held
    sampler = torch.Generator(device='cuda').manual_seed(42)
    signature = hashlib.sha256()
    for step in range(STEPS):
        ix = torch.randint(819, (4,), device='cuda', generator=sampler)
        available = (torch.rand(4, device='cuda', generator=sampler) >= .2).float()
        signature.update(ix.cpu().numpy().tobytes())
        signature.update(available.cpu().numpy().tobytes())
    assert set(frozen['same_sampler_as_full28'].values()) == {signature.hexdigest()}
    old_schedule = read(ABLATION / 'heads_frozen.json')['minibatch_and_masks_sha256']['full']
    assert signature.hexdigest() == old_schedule
    original = load(SOURCE / 'warp.pt')['model']
    scale = load(MEMORY / 'statistics.pt')['memory_scale']
    inputs = training_data(normalization())
    updates = []
    for name in ARMS:
        head = new_head(name, scale).requires_grad_(True)
        state = load(OUT / (name + '_initial.pt'))['model']
        for key, value in state.items():
            expected = original[key]
            if name == 'compact16' and key == 'net.0.weight':
                expected = expected[:, list(CORE_INDICES)]
            assert torch.equal(value, expected), (name, key)
        head.load_state_dict(state)
        if name == 'compact16':
            assert head.net[0].in_channels == 16
            assert sum(p.numel() for p in head.parameters()) == 34004
            assert not any(key.startswith(('modifier.', 'memory_projection.', 'history_projection.')) for key in state)
        saved = load(OUT / (name + '_first_update.pt'))
        values = [item[saved['indices']] for item in inputs[:4]]
        residual = inputs[4][saved['indices']]
        with torch.no_grad():
            before = head.bounds(*values, saved['available'])
            altered = [item.clone() for item in values]
            removed = [i for i in range(14) if i not in BASE_INDICES]
            altered[0][:, removed] -= 29
            altered[2] += 19
            altered[3] -= 37
            after = head.bounds(*altered, torch.zeros_like(saved['available']))
            assert all(torch.equal(a, b) for a, b in zip(before, after))
            if name == 'compact16':
                independent = independent_compact_bounds(head, values[0], values[1])
                assert all(torch.equal(a, b) for a, b in zip(before, independent))
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-5)
        lower, upper = head.bounds(*values, saved['available'])
        left, right = residual - lower, residual - upper
        lower_loss = torch.where(left == 0, -.45 * left, torch.where(left > 0, .05 * left, -.95 * left))
        upper_loss = torch.where(right == 0, .45 * right, torch.where(right > 0, .95 * right, -.05 * right))
        loss = (lower_loss + upper_loss).mean()
        assert abs(float(loss) - saved['loss']) < 1e-8
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
        grad_error = 0.
        for key, parameter in head.named_parameters():
            expected = saved['gradients'][key]
            assert (parameter.grad is None) == (expected is None)
            if expected is not None:
                grad_error = max(grad_error, float((parameter.grad - expected).abs().max()))
                assert torch.allclose(parameter.grad, expected, rtol=1e-5, atol=1e-8), (name, key)
        optimizer.step()
        update_error = 0.
        for key, tensor in head.state_dict().items():
            expected = saved['model'][key]
            update_error = max(update_error, float((tensor - expected).abs().max()))
            assert torch.allclose(tensor, expected, rtol=1e-6, atol=1e-7), (name, key)
        updates.append(dict(name=name, gradient_max_error=grad_error, update_max_error=update_error))
        print('SELECTED16_FIRST_UPDATE_AUDIT', updates[-1], flush=True)
    rows = read(OUT / 'per_case.json')
    old = read(ABLATION / 'per_case.json')
    key = lambda r: tuple(r[k] for k in ('file', 'start', 'channel', 'variant'))
    old_lookup = {(r['map'], *key(r)): r for r in old if r['map'] in ('published_warp', 'full')}
    checks = 0
    for row in rows:
        name = {'published_warp': 'published_warp', 'matched_full28': 'full'}.get(row['map'])
        if name is not None:
            previous = old_lookup[(name, *key(row))]
            assert all(row[k] == value for k, value in previous.items() if k != 'map')
            checks += 1
    assert checks == 3520 and len(rows) == 7040
    events = [json.loads(line) for line in (OUT / 'pilot_events.jsonl').read_text().splitlines()]
    assert len(events) == 20
    for issued, revealed in zip(events[::2], events[1::2]):
        assert issued['event'] == 'issued' and revealed['event'] == 'revealed'
        assert issued['start'] == revealed['start'] and issued['time'] <= revealed['time']
    macro = read(OUT / 'pilot_macro.json')
    for mode, methods in macro.items():
        for name, expected in methods.items():
            buckets = defaultdict(list)
            for row in rows:
                if row['map'] == name and row['variant'] == mode and row['start'] >= 600:
                    buckets[row['group'], row['channel']].append(row)
            actual = {metric: mean(mean(row[metric] for row in group) for group in buckets.values())
                      for metric in METRICS if metric != 'coverage_error'}
            actual['coverage_error'] = mean(abs(mean(row['coverage'] for row in group) - .9) for group in buckets.values())
            assert all(abs(value - expected[metric]) < 1e-12 for metric, value in actual.items())
    dump(OUT / 'independent_audit.json', dict(passed=True, updates=updates,
        real16_input_and34004parameters=True, unused_projections_and_modifier_removed=True,
        initial_retained_weights_exact=True, omitted_inputs_and_availability_invariant=True,
        compact_forward_matches_independent_feature_construction=True,
        full28_reused_run_exact_same_sampler_and_budget=True, old_control_rows_exact=checks,
        all_rows=len(rows), all_macro_metrics_verified=True, issue_before_full_reveal=True,
        no_calibration_or_heldout_labels_fit_weights=True, frozen_artifacts_unchanged=True,
        caveat='Userselectedsubset from previously studieddata; exploratory conditional comparison, not independent confirmation or formal coverage.'))
    print('SELECTED16_INDEPENDENT_TRAINING_AUDIT_PASS', flush=True)


if __name__ == '__main__':
    main()
