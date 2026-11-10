"""Train the requested selected16 head and a same-information masked control.

Reuse the audited full28 continuation checkpoint: its exact initialization,
819pairs,2000steps, optimizer and minibatches match this experiment. This
avoids spending another identical full28 training run. No test-based refit.
"""
import hashlib
import json
import math
import time
from pathlib import Path
import torch

from clean_reliability_common import ROOT, configure, dump, digest, manifest
from experiment_dynamic_risk_memory import Provider as SourceProvider, OUT as SOURCE, BEST
from experiment_feature_ablation28 import OUT as ABLATION, new_head as previous_head
from experiment_fullblock_memory_risk import OUT as MEMORY, load
from experiment_reliability_transport_e12 import FIT_RECORDS, PREDICTOR, normalization
from experiment_extended_train_error_adapters import LATE_RECORDS
from temporal_mean_memory_control import training_data
from selected16_quantile_head import Selected16QuantileHead, MaskedSelected16Head, CORE_INDICES, BASE_INDICES, SELECTED_GROUPS
from risk_head_pilot_protocol import calibrate, pilot
from experiment_continue_rich_uncertainty import MODES
from experiment_global_operator_risk import METRICS

OUT = ROOT / 'runs/selected16_quantile_20261110'
ARMS = ('compact16', 'masked16')
NAMES = ('published_warp', 'matched_full28', *ARMS)
STEPS = 2000


def new_head(name, scale):
    dynamic = load(SOURCE / 'warp.pt')['model']
    if name == 'compact16':
        return Selected16QuantileHead(dynamic).cuda().eval()
    if name == 'masked16':
        original = load(BEST / 'density_quantile.pt')['model']
        return MaskedSelected16Head(scale, original, dynamic).cuda().eval()
    if name == 'matched_full28':
        head = previous_head('full', scale)
        head.load_state_dict(load(ABLATION / 'full.pt')['model'])
        return head.eval().requires_grad_(False)
    raise ValueError(name)


@torch.no_grad()
def source_independence(head, values):
    base, last, history, memory = [item[:4].clone() for item in values[:4]]
    before = head.bounds(base, last, history, memory, True)
    omitted = [i for i in range(14) if i not in BASE_INDICES]
    base[:, omitted] += 31
    after = head.bounds(base, last, history + 17, memory + 19, False)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    return dict(omitted_base_history_memory_and_availability_invariant=True)


def train(norm):
    values = training_data(norm)
    base, last, history, memory, residual = values
    assert len(base) == 819
    scale = load(MEMORY / 'statistics.pt')['memory_scale']
    initial = {name: new_head(name, scale) for name in ARMS}
    with torch.no_grad():
        a = initial['compact16'].bounds(base[:8], last[:8], history[:8], memory[:8], True)
        b = initial['masked16'].bounds(base[:8], last[:8], history[:8], memory[:8], True)
        difference = max(float((x - y).abs().max()) for x, y in zip(a, b))
        assert all(torch.allclose(x, y, rtol=1e-5, atol=2e-6) for x, y in zip(a, b)), difference
    dump(OUT / 'initial_function_check.json', dict(passed=True, maximum_absolute_bound_error=difference,
        tolerance='float32 rtol1e-5/atol2e-6 for sliced-convolution reduction, not relaxed primary metric tolerance1e-6'))
    schedules, checks = {}, {}
    for name in ARMS:
        torch.manual_seed(42)
        head = initial[name].requires_grad_(True)
        checks[name] = source_independence(head, values)
        torch.save(dict(model=head.state_dict()), OUT / (name + '_initial.pt'))
        params = list(head.parameters())
        optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-5)
        sampler = torch.Generator(device='cuda').manual_seed(42)
        schedule = hashlib.sha256()
        logs, begin = [], time.time()
        for step in range(1, STEPS + 1):
            ix = torch.randint(819, (4,), device='cuda', generator=sampler)
            available = (torch.rand(4, device='cuda', generator=sampler) >= .2).float()
            schedule.update(ix.cpu().numpy().tobytes())
            schedule.update(available.cpu().numpy().tobytes())
            optimizer.param_groups[0]['lr'] = 1e-6 + (3e-4 - 1e-6) * .5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
            optimizer.zero_grad(set_to_none=True)
            loss = head.loss(base[ix], last[ix], history[ix], memory[ix], residual[ix], available)
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(params, 1))
            if step == 1:
                gradients = {key: p.grad.detach().cpu().clone() if p.grad is not None else None
                             for key, p in head.named_parameters()}
            optimizer.step()
            if step == 1:
                torch.save(dict(model={key: tensor.detach().cpu() for key, tensor in head.state_dict().items()},
                                gradients=gradients, indices=ix.cpu(), available=available.cpu(), loss=float(loss)),
                           OUT / (name + '_first_update.pt'))
            if step == 1 or step % 500 == 0:
                logs.append(dict(step=step, loss=float(loss), elapsed=time.time() - begin))
                print('SELECTED16_TRAIN', name, logs[-1], flush=True)
        schedules[name] = schedule.hexdigest()
        torch.save(dict(model=head.state_dict(), steps=STEPS, seed=42, examples=819,
                        parameters=sum(p.numel() for p in params), effective_core_channels=16), OUT / (name + '.pt'))
        dump(OUT / (name + '_training.json'), logs)
    matched = json.loads((ABLATION / 'heads_frozen.json').read_text())['minibatch_and_masks_sha256']['full']
    assert set(schedules.values()) == {matched}
    dump(OUT / 'heads_frozen.json', dict(time=time.time(), parameters=dict(compact16=34004, masked16=58770, matched_full28=58770),
        source_checks=checks, steps=STEPS, fit_pairs=819, same_sampler_as_full28=schedules,
        full28_reused_checkpoint_sha256=digest(ABLATION / 'full.pt'),
        artifacts={name: digest(OUT / (name + '.pt')) for name in ARMS}))


class Provider(SourceProvider):
    def __init__(self, norm):
        super().__init__(norm)
        self.heads = {'published_warp': self.heads['warp']}
        scale = load(MEMORY / 'statistics.pt')['memory_scale']
        for name in NAMES[1:]:
            head = new_head(name, scale)
            if name in ARMS:
                head.load_state_dict(load(OUT / (name + '.pt'))['model'])
            self.heads[name] = head.eval().requires_grad_(False)


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    assert json.loads((ABLATION / 'independent_audit.json').read_text())['passed']
    fit_ids = {r['file'] for r in FIT_RECORDS + LATE_RECORDS}
    held = {r['file'] for split, rows in manifest()['files'].items() if split != 'train' for r in rows}
    assert not fit_ids & held
    paths = [SOURCE / 'warp.pt', BEST / 'density_quantile.pt', ABLATION / 'full.pt',
             ABLATION / 'calibration.pt', PREDICTOR / 'augmented_diffusion/model.pt']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    dump(OUT / 'protocol.json', dict(time=time.time(), source_sha256=digest(Path(__file__)),
        module_sha256=digest(ROOT / 'selected16_quantile_head.py'), frozen_artifacts=hashes,
        groups=SELECTED_GROUPS, zero_based_core_indices=CORE_INDICES, effective_channels=16, names=NAMES, modes=MODES,
        fit_pairs=819, steps=STEPS, seed=42, fit_files=sorted(fit_ids),
        architecture='Compact head keeps6normalizedbase channels,8projecteddenoising features,2coordinates. Real16->24convolution. Remove physical-historyprojection,error-memoryprojection,anddisplacementmodule entirely.34004params versus58770full riskhead. Masked16control keepssame16information but originalallocatedparams.',
        fitting='Both16arms initialized by retaining/slicing the original publishedwarp riskweights before previous2000step ablation. Same819pairs,2000updates,AdamW3e-4cosine1e-6,batch4,clip1,seed42,dedicatedbatch/masks. Full28 is the previously audited EXACT matched continuation run, reused not newly refitted. DiPred frozen.',
        calibration='Freeze bothnewheads, then independent10IDcalibration perhead. Publishedwarp andmatchedfull28 states mustexactly reproduce their frozen priorstates. Parentcalibrators retain identicaloriginalfeature builders, so selected16describes neuralhead not every known-fieldinput in wholepipeline.',
        evaluation='Run both fixedonlinepolicies on all originaltrain-IDpilot and full devtest trajectories for ALL4methods, regardless of pilot outcome. No test-driven reruns, changes, coefficient selection, folds, distillation or trainingseedscan. Preserve all8metrics/96cells/tolerance1e-6/all87referencepairs.',
        caveat='Subset userselected after previous repeatedhistoricaltests, not a fresh independent generalization confirmation. No universal necessity, formal coverage or latency gain inferred from parameter count alone.'))
    norm = normalization()
    train(norm)
    provider = Provider(norm)
    reference = load(SOURCE / 'calibration.pt')['warp']['separable_integral']
    initial = calibrate(provider, OUT, NAMES, 'published_warp', reference)
    prior = load(ABLATION / 'calibration.pt')['full']['separable_integral']
    assert all(torch.equal(v, prior[k]) for k, v in initial['matched_full28']['separable_integral'].items())
    values = pilot(provider, OUT, NAMES, initial)
    gains = {}
    for mode, methods in values.items():
        gains[mode] = {name: {k: direction * (methods[name][k] - methods['matched_full28'][k])
                              for k, direction in METRICS.items()} for name in ARMS}
    dump(OUT / 'pilot_comparisons.json', gains)
    assert all(digest(ROOT / path) == sha for path, sha in hashes.items())
    dump(OUT / 'complete.json', dict(time=time.time(), original_artifacts_unchanged=True,
                                    development_test_not_evaluated=True, full_evaluation_planned_for_all_arms=True))
    print('SELECTED16_QUANTILE_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
