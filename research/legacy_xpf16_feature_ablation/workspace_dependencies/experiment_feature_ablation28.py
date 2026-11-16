"""Matched 11 source-group deletions plus full-input continuation control.

Start all arms at the identical published warp checkpoint. Fit risk parameters
only on the original 819 training pairs, keeping the PDE predictor frozen.
This tests removal after fixed-budget adaptation, not fresh-training superiority.
"""
import hashlib
import json
import math
import time
from pathlib import Path
import torch

from clean_reliability_common import ROOT, configure, dump, digest, manifest
from experiment_dynamic_risk_memory import Provider as SourceProvider, OUT as SOURCE, BEST, new_head as source_head
from experiment_fullblock_memory_risk import OUT as MEMORY, load
from experiment_reliability_transport_e12 import FIT_RECORDS, PREDICTOR, normalization
from experiment_extended_train_error_adapters import LATE_RECORDS
from temporal_mean_memory_control import training_data
from feature_ablation_risk_head import AblatedRiskHead, GROUPS, BASE_GROUPS, ARMS
from experiment_global_operator_risk import METRICS
from experiment_continue_rich_uncertainty import MODES
from risk_head_pilot_protocol import calibrate, pilot

OUT = ROOT / 'runs/feature_ablation28_20261110'
NAMES = ('published_warp',) + ARMS
STEPS = 2000


def new_head(name, scale):
    original = load(BEST / 'density_quantile.pt')['model']
    dynamic = load(SOURCE / 'warp.pt')['model']
    dropped = None if name == 'full' else name[len('drop_'):]
    return AblatedRiskHead(scale, original, dynamic, dropped).cuda().eval()


def independence_checks(head, inputs):
    """Verify omitted source and frozen full-control arithmetic before fitting."""
    base, last, history, memory, residual = inputs
    values = [item[:2].clone() for item in (base, last, history, memory)]
    dropped = head.dropped
    with torch.no_grad():
        expected = head.bounds(*values, True)
        altered = [item.clone() for item in values]
        if dropped in BASE_GROUPS:
            altered[0][:, list(BASE_GROUPS[dropped])] += 13
        elif dropped == 'history_memory':
            altered[2] += 13
            altered[3] += 13
        elif dropped == 'denoising':
            altered[1] += 13
        if dropped != 'coordinates':
            actual = head.bounds(*altered, True)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        captured = {}
        h1 = head.net.register_forward_pre_hook(lambda module, args: captured.update(core=args[0].clone()))
        h2 = head.modifier.register_forward_pre_hook(lambda module, args: captured.update(modifier=args[0].clone()))
        try:
            head.bounds(*values, True)
        finally:
            h1.remove()
            h2.remove()
        if dropped is not None:
            indices = list(GROUPS[dropped])
            assert torch.count_nonzero(captured['core'][:, indices]) == 0
            assert torch.count_nonzero(captured['modifier'][:, indices]) == 0
            if dropped == 'history_memory':
                assert torch.count_nonzero(captured['modifier'][:, 28:]) == 0
    return dict(dropped=dropped, masked_core_and_modifier_exact=True,
                omitted_source_perturbation_invariant=dropped != 'coordinates')


def train(norm):
    inputs = training_data(norm)
    base, last, history, memory, residual = inputs
    assert len(base) == 819
    scale = load(MEMORY / 'statistics.pt')['memory_scale']
    published = source_head('warp', scale)
    published.load_state_dict(load(SOURCE / 'warp.pt')['model'])
    with torch.no_grad():
        baseline = published.bounds(base[:8], last[:8], history[:8], memory[:8], True)
    schedules, checks, initial_hashes = {}, {}, {}
    for name in ARMS:
        torch.manual_seed(42)
        head = new_head(name, scale).requires_grad_(True)
        assert sum(p.numel() for p in head.parameters()) == 58770
        checks[name] = independence_checks(head, inputs)
        if name == 'full':
            with torch.no_grad():
                actual = head.bounds(base[:8], last[:8], history[:8], memory[:8], True)
                assert all(torch.equal(a, b) for a, b in zip(actual, baseline))
        torch.save(dict(model=head.state_dict()), OUT / (name + '_initial.pt'))
        # Serialize tensor bytes, not archive metadata, to verify common weights.
        fingerprint = hashlib.sha256()
        for key, tensor in head.state_dict().items():
            fingerprint.update(key.encode())
            fingerprint.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        initial_hashes[name] = fingerprint.hexdigest()
        parameters = list(head.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-5)
        sampler = torch.Generator(device='cuda').manual_seed(42)
        schedule = hashlib.sha256()
        logs, begin = [], time.time()
        for step in range(1, STEPS + 1):
            indices = torch.randint(819, (4,), device='cuda', generator=sampler)
            available = (torch.rand(4, device='cuda', generator=sampler) >= .2).float()
            schedule.update(indices.cpu().numpy().tobytes())
            schedule.update(available.cpu().numpy().tobytes())
            optimizer.param_groups[0]['lr'] = 1e-6 + (3e-4 - 1e-6) * .5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
            optimizer.zero_grad(set_to_none=True)
            loss = head.loss(base[indices], last[indices], history[indices], memory[indices], residual[indices], available)
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters, 1))
            if step == 1:
                gradients = {key: p.grad.detach().cpu().clone() if p.grad is not None else None
                             for key, p in head.named_parameters()}
            optimizer.step()
            if step == 1:
                torch.save(dict(model={key: value.detach().cpu() for key, value in head.state_dict().items()},
                                gradients=gradients, indices=indices.cpu(), available=available.cpu(), loss=float(loss)),
                           OUT / (name + '_first_update.pt'))
            if step == 1 or step % 500 == 0:
                logs.append(dict(step=step, loss=float(loss), elapsed=time.time() - begin))
                print('FEATURE_ABLATION_TRAIN', name, logs[-1], flush=True)
        schedules[name] = schedule.hexdigest()
        torch.save(dict(model=head.state_dict(), dropped=head.dropped, steps=STEPS, seed=42,
                        parameters=58770, retained_core_channels=28 - len(GROUPS.get(head.dropped, ()))), OUT / (name + '.pt'))
        dump(OUT / (name + '_training.json'), logs)
    assert len(set(schedules.values())) == 1
    assert len(set(initial_hashes.values())) == 1
    dump(OUT / 'heads_frozen.json', dict(time=time.time(), steps=STEPS, fit_pairs=819,
        identical_initial_state_sha256=initial_hashes, minibatch_and_masks_sha256=schedules,
        parameters_per_arm=58770, masking_checks=checks,
        all_neural_training_complete_before_calibration=True,
        artifacts={name: digest(OUT / (name + '.pt')) for name in ARMS}))


class Provider(SourceProvider):
    def __init__(self, norm):
        super().__init__(norm)
        self.heads = {'published_warp': self.heads['warp']}
        scale = load(MEMORY / 'statistics.pt')['memory_scale']
        for name in ARMS:
            head = new_head(name, scale)
            head.load_state_dict(load(OUT / (name + '.pt'))['model'])
            self.heads[name] = head.requires_grad_(False)


def decisions(values):
    result = {}
    for mode, methods in values.items():
        comparisons = {}
        for name in ARMS[1:]:
            gains = {key: direction * (methods[name][key] - methods['full'][key]) for key, direction in METRICS.items()}
            comparisons[name] = dict(gains=gains, regressions=[key for key, value in gains.items() if value < -1e-6],
                                    no_primary_regression=all(value >= -1e-6 for value in gains.values()),
                                    strict_score_improvement=gains['standardized_interval_score'] > 1e-6)
        result[mode] = comparisons
    return dict(comparisons=result, reference='same-budget full-input continuation', tolerance=1e-6,
                scope='Exploratory later times of original training identities; not development/test evidence.',
                no_claim_of_practical_or_statistical_equivalence=True)


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    fit_ids = {row['file'] for row in FIT_RECORDS + LATE_RECORDS}
    held = {row['file'] for split, records in manifest()['files'].items() if split != 'train' for row in records}
    assert not fit_ids & held
    paths = [SOURCE / 'warp.pt', BEST / 'density_quantile.pt', MEMORY / 'statistics.pt',
             PREDICTOR / 'augmented_diffusion/model.pt']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    dump(OUT / 'protocol.json', dict(time=time.time(), module_sha256=digest(ROOT / 'feature_ablation_risk_head.py'),
        source_sha256=digest(Path(__file__)), arms=ARMS, groups=GROUPS, modes=MODES,
        base_group_order=BASE_GROUPS, frozen_artifacts=hashes, fit_files=sorted(fit_ids),
        allocation='58770 parameters in every arm; input masking isolates information, not a reduced-parameter model.',
        fitting='All arms start from exactly the same published warp state. Fine-tune ALL risk parameters for2000updates on819 original training pairs; same seed42,batch4,AdamW3e-4cosine1e-6,clip1,20%availability masks. Point predictor never updated. Full-input continued arm controls additional training.',
        omission='After-normalization base channels, after-projection denoising and history channels. Mask all corresponding modifier inputs too. history_memory removes physical-history projection AND projected revealed-error memory because they share four core channels. Other base/history inputs may redundantly encode the same raw information; effects are conditional, not absence of all related physical information.',
        calibration='After ALL heads freeze, independently recompute same10calibration identities for every arm; published_warp calibration must reproduce existing state exactly. No calibration target trains weights.',
        pilot='Original44training identities, query580coldthen600..760warm; true target read after all intervals issue. Rank deletion effect versus matched continued-full, separately from published warp. No dev/test access for initial selection.',
        decision='Keep original8metrics/tolerance1e-6. A deletion is a non-regression candidate only if all8no worse vsmatchedfull under a fixed controller; retain all tradeoffs. Never call a finite one-seed experiment proof of unnecessary features. Combining deletions needs another experiment.',
        invariants='Native99,66,33,0/full20to20/frozenpoint/savedpreviousP/full20reveal/no32sampling/no folds/distillation/seedscan. All existingstrongrefs retained; no auto publication.',
        caveat='Repeatedly reused historical identities. Fixed-budget continuation ablation, not fresh-training or universal feature necessity.'))
    norm = normalization()
    train(norm)
    provider = Provider(norm)
    reference = load(SOURCE / 'calibration.pt')['warp']['separable_integral']
    initial = calibrate(provider, OUT, NAMES, 'published_warp', reference)
    values = pilot(provider, OUT, NAMES, initial)
    dump(OUT / 'gate.json', decisions(values))
    assert all(digest(ROOT / path) == sha for path, sha in hashes.items())
    dump(OUT / 'complete.json', dict(time=time.time(), original_artifacts_unchanged=True,
                                    development_test_not_evaluated=True, arms_trained=len(ARMS)))
    print('FEATURE_ABLATION28_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
