"""Independent, identical-budget XPF16 dual-quantile heads; resumable."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from clean_reliability_common import ROOT, configure, digest, dump, manifest
from experiment_selected16_quantile import new_head
from experiment_reliability_transport_e12 import FIT_RECORDS, normalization
from experiment_extended_train_error_adapters import LATE_RECORDS
from experiment_fullblock_memory_risk import OUT as MEMORY, load
from temporal_mean_memory_control import training_data
from calibration import TwoTailCalibrator
from experiment_reliability_transport_e12 import geometry, target
from experiment_temporal_error_uncertainty import pack, normalized
from experiment_fullblock_memory_risk import HEAD_SOURCE

OUT = ROOT / 'runs/xpf16_fullpercent_20261111'
STEPS = 2000
SEED = 42


def configs():
    return [dict(id=f'XPF16-Q{k:02d}', k=k, lower=k/100, upper=1-k/100,
                 target_coverage=1-2*k/100, seed=SEED, steps=STEPS,
                 batch=4, learning_rate=3e-4, weight_decay=1e-5,
                 grad_clip=1.0) for k in range(1, 50)]


def pinball(error, tau):
    return torch.maximum(tau * error, (tau - 1) * error)


def atomicsave(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def fit(cfg, data, pilot=False):
    base, last, history, memory, residual = data
    tag = cfg['id']
    folder = OUT / ('pilot' if pilot else 'models') / tag
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / 'model.pt'
    statepath = folder / 'resume.pt'
    if final.exists() and not pilot:
        print('SKIP_COMPLETE', tag, flush=True)
        return
    torch.manual_seed(SEED)
    scale = load(MEMORY / 'statistics.pt')['memory_scale']
    head = new_head('compact16', scale).requires_grad_(True)
    # This is the original common pretraining initialization, not Q05 weights.
    init = load(ROOT / 'runs/selected16_quantile_20261110/compact16_initial.pt')['model']
    head.load_state_dict(init, strict=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-5)
    sampler = torch.Generator(device='cuda').manual_seed(SEED)
    start, elapsed, logs = 0, 0.0, []
    if statepath.exists() and not pilot:
        saved = torch.load(statepath, map_location='cuda', weights_only=False)
        assert saved['config'] == cfg
        head.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        sampler.set_state(saved['sampler'].cpu())
        start, elapsed, logs = saved['step'], saved['elapsed'], saved['log']
    begin = time.monotonic()
    total = 8 if pilot else STEPS
    for step in range(start + 1, total + 1):
        ix = torch.randint(819, (4,), device='cuda', generator=sampler)
        available = (torch.rand(4, device='cuda', generator=sampler) >= .2).float()
        optimizer.param_groups[0]['lr'] = 1e-6 + (3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/STEPS))
        optimizer.zero_grad(set_to_none=True)
        lo, hi = head.bounds(base[ix], last[ix], history[ix], memory[ix], available)
        assert torch.isfinite(lo).all() and torch.isfinite(hi).all() and (lo < hi).all()
        loss = .5*(pinball(residual[ix]-lo, cfg['lower']) + pinball(residual[ix]-hi, cfg['upper'])).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{tag} step {step}: invalid loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
        if not torch.isfinite(grad):
            raise FloatingPointError(f'{tag} step {step}: invalid gradient')
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == total:
            logs.append(dict(step=step, loss=float(loss), seconds=elapsed+time.monotonic()-begin))
        if step % 250 == 0 and not pilot:
            atomicsave(dict(config=cfg, step=step, elapsed=elapsed+time.monotonic()-begin,
                            model=head.state_dict(), optimizer=optimizer.state_dict(),
                            sampler=sampler.get_state(), log=logs), statepath)
    if pilot:
        # Separate smoke-test model: exercise fit -> calibration -> issuance ->
        # complete feedback -> next issuance on development, before all 49 fits.
        norm = normalization()
        stats = load(HEAD_SOURCE / 'statistics.pt')['statistics']
        observed = []
        for split, block in (('calibration', 0), ('development', 0), ('development', 20)):
            point, baseval, lastval = pack(split, block, norm, geometry(manifest()['files'][split]))
            baseval, lastval = normalized(baseval, lastval, stats)
            with torch.no_grad():
                left, right = head.bounds(baseval, lastval, None, None, False)
            if split == 'calibration':
                residual_cal = (target(split,block)-norm['mean'])/norm['std']-point
                offsets = torch.stack((torch.quantile((residual_cal-left).flatten(2),cfg['lower'],dim=2).mean(0),
                                       torch.quantile((residual_cal-right).flatten(2),cfg['upper'],dim=2).mean(0)))
                control = None
            else:
                if control is None:
                    control = TwoTailCalibrator(cfg['lower'], offsets, len(point))
                issued_lo, issued_hi = control.issue(left, right)
                assert torch.isfinite(issued_lo).all() and torch.isfinite(issued_hi).all() and (issued_lo<=issued_hi).all()
                residual_dev = (target(split,block)-norm['mean'])/norm['std']-point
                coverage = ((residual_dev>=issued_lo)&(residual_dev<=issued_hi)).float().mean()
                score = (issued_hi-issued_lo+((issued_lo-residual_dev).clamp_min(0)+(residual_dev-issued_hi).clamp_min(0))/cfg['lower']).mean()
                assert torch.isfinite(score)
                observed.append(dict(block=block, coverage=float(coverage), score=float(score)))
                control.reveal(residual_dev)
        dump(folder / 'pilot.json', dict(config=cfg, loss=logs, ordered=True, smoke_development=observed,
             calibration_initialization=True, complete_window_updates=control.revealed_windows,
             parameters=sum(p.numel() for p in head.parameters())))
    else:
        atomicsave(dict(config=cfg, model={k:v.detach().cpu() for k,v in head.state_dict().items()},
                        elapsed=elapsed+time.monotonic()-begin, loss=logs,
                        parameters=sum(p.numel() for p in head.parameters()), training_pairs=819), final)
        dump(folder / 'training.json', dict(config=cfg, sha256=digest(final), loss=logs,
             elapsed=elapsed+time.monotonic()-begin, parameters=sum(p.numel() for p in head.parameters())))
        statepath.unlink(missing_ok=True)
    print('PILOT' if pilot else 'TRAINED', tag, 'seconds', elapsed+time.monotonic()-begin, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args()
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((ROOT/'runs/selected16_quantile_20261110/independent_audit.json').read_text())['passed']
    fit_ids = {r['file'] for r in FIT_RECORDS + LATE_RECORDS}
    held = {r['file'] for split, rows in manifest()['files'].items() if split != 'train' for r in rows}
    assert not fit_ids & held
    cfgs = configs()
    assert len(cfgs) == 49 and len({x['id'] for x in cfgs}) == 49
    dump(OUT/'protocol.json', dict(configs=cfgs, source_sha256=digest(Path(__file__)),
         parent_commit='2cf5377eb4e94cb0eaf793e4926b4fd14cfa889f',
         old_q05_audit='runs/selected16_quantile_20261110/independent_audit.json',
         old_q05_sha256=digest(ROOT/'runs/selected16_quantile_20261110/compact16.pt'),
         common_initial_sha256=digest(ROOT/'runs/selected16_quantile_20261110/compact16_initial.pt'),
         initial_strategy='same pretrained pre-Q05 XPF16 initial state in every independent head',
         training_ids=sorted(fit_ids), heldout_identity_disjoint=True,
         caveat='Historical dev/test trajectories used repeatedly in earlier work; no fresh test claim.'))
    data = training_data(normalization())
    assert len(data[0]) == 819
    for k in (1,5,25,49):
        fit(cfgs[k-1], data, pilot=True)
    if not args.pilot_only:
        for cfg in cfgs:
            fit(cfg, data)
        dump(OUT/'training_complete.json',dict(passed=True, models=49,
             all_checkpoints={c['id']:digest(OUT/'models'/c['id']/'model.pt') for c in cfgs}))

if __name__ == '__main__':
    main()
