"""Single entry: prepare, preflight, train, report, verify. See README.md."""
import os
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LOCAL_DEPS = ROOT/'.cache/coupled_state_feedback_dependencies'
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
import h5py
import numpy as np
import torch
from reliability.backbone import configure
from reliability.networks import DynamicsDiffusion20
from research.coupled_state_feedback.core import Forecast, objective, scores
from research.coupled_state_feedback.checks import real_checks
from research.residual_dynamics_phase2.metrics import empirical_crps, energy_score

OUT = ROOT/'results/coupled_state_feedback'
SOURCE = Path(__file__).resolve().parent


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read(p):
    return json.loads(p.read_text())


def event(name, **kw):
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(event=name, utc=time.time(), **kw), ensure_ascii=False, allow_nan=False)+'\n')


def cfg():
    return read(OUT/'config.json')


def code_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(SOURCE.glob('*.py'))}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def prepare(data):
    assert not (OUT/'config.json').exists(), 'Protocol already frozen; no overwrite.'
    original = read(ROOT/'results/residual_dynamics/protocol.json')
    manifest = read(ROOT/'metadata/manifest.json')
    records = original['records']
    assert {k: len(v) for k, v in records.items()} == dict(train=46, development=12, test=12)
    assert all(r in manifest['files'][split] for split, rs in records.items() for r in rs)
    sets = [set(r['file'] for r in records[k]) for k in ('train', 'development', 'test')]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    groups = [set(r['group'] for r in records[k]) for k in ('train', 'development', 'test')]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
    windows, excluded = {}, []
    for split, rs in records.items():
        windows[split] = []
        for r in rs:
            with h5py.File(data/r['file'], 'r') as f:
                assert len(f['u']) == len(f['v']) == r['frames']
            for start in ([200, 300, 400, 420, 500, 600] if split == 'train' else [400, 420]):
                row = dict(split=split, trajectory=r['file'], Re=r['group'], start=start)
                if start+40 <= r['frames']:
                    windows[split].append(row)
                else:
                    excluded.append(dict(**row, frames=r['frames']))
    rng = np.random.default_rng(42)
    schedule = [dict(update=i+1, window=int(rng.integers(len(windows['train']))), noise_seed=42000000+i)
                for i in range(1000)]
    dependencies = ['reliability/networks.py', 'reliability/backbone.py',
        'reliability/checkpoints/current.pt', 'metadata/normalization.json', 'metadata/manifest.json',
        'results/residual_dynamics/protocol.json', 'research/provenance_uncertainty_conditioning/core.py',
        'results/conditioning_failure_audit/DECISION_REPORT.md', 'research/residual_dynamics_phase2/metrics.py']
    configuration = dict(reference='fad8e7eb1af8b1f2768b4eaf6c883c654cb262b1', head=git('rev-parse', 'HEAD'),
        initial_status=git('status', '--short'), data_root=str(data.resolve()), records=records, windows=windows,
        excluded_windows=excluded, seed=42, schedule=schedule, arms=['F','O','C'], steps=1000, candidates=[0,500,1000],
        training_members=8, development_members=16, test_members=32, lr_backbone=1e-5, lr_new=1e-4,
        optimizer='Adam default betas/eps, weight_decay=0, clip_grad_norm=1',
        loss='CRPS_U + 0.1 ES_U, fixed train std units, full 20-frame RMS ES',
        max_gpu_seconds=7200, max_cache_bytes=4*1024**3, evaluation_reserve_seconds=240,
        development_noise_root=16042000, test_noise_root=32042000, bootstrap_seed=123, bootstrap_repeats=10000,
        replay_atol_standardized=2e-6, protocol_sha256=digest(OUT/'PROTOCOL.md'),
        dependencies={p:digest(ROOT/p) for p in dependencies}, initial_code=code_hashes())
    dump(OUT/'config.json', configuration)
    dump(OUT/'environment.json', dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), h5py=h5py.__version__, numpy=np.__version__,
        commands=['prepare','preflight','train','report','verify']))
    event('protocol_frozen', config_sha256=digest(OUT/'config.json'), windows={k:len(v) for k,v in windows.items()})
    print('FROZEN', {k:len(v) for k,v in windows.items()}, flush=True)


def check_dependencies():
    c = cfg()
    assert digest(OUT/'PROTOCOL.md') == c['protocol_sha256']
    for p, sha in c['dependencies'].items():
        assert digest(ROOT/p) == sha, p


def load():
    check_dependencies()
    checkpoint = torch.load(ROOT/'reliability/checkpoints/current.pt', map_location='cuda', weights_only=False)
    native = DynamicsDiffusion20(checkpoint['dynamics_scale']).cuda().eval().requires_grad_(False)
    native.load_state_dict(checkpoint['predictor'], strict=True)
    oldpath = ROOT.parents[1]/'runs/physical20_augmented_dynamics_20261109/augmented_diffusion/model.pt'
    old = torch.load(oldpath, map_location='cuda', weights_only=False)['model']
    assert old.keys() == checkpoint['predictor'].keys()
    assert all(torch.equal(old[k], checkpoint['predictor'][k]) for k in old)
    norm = {k:v.cuda() for k,v in checkpoint['normalization'].items()}
    metadata = read(ROOT/'metadata/normalization.json')
    for k, v in norm.items():
        assert torch.equal(v.flatten().cpu(), torch.tensor(metadata[k]))
    return native, norm


def frames(entry, future=False):
    begin = entry['start'] + (20 if future else 0)
    with h5py.File(Path(cfg()['data_root'])/entry['trajectory'], 'r') as f:
        a = np.stack([f[k][begin:begin+20, ::2, ::2] for k in ('u','v')]).astype('float32')
    assert a.shape == (2,20,32,64) and np.isfinite(a).all()
    return torch.from_numpy(a).cuda()


def condition(entry, norm, members):
    return ((frames(entry)[None]-norm['mean'])/norm['std']).expand(members,-1,-1,-1,-1).contiguous()


def target(entry, norm):
    return ((frames(entry, True)[None]-norm['mean'])/norm['std'])[0]


def noise(h, seed):
    # Independent sequential pseudorandom draws for all members; same schedule across arms.
    return torch.randn(h.shape, device=h.device, dtype=h.dtype,
        generator=torch.Generator(device=h.device).manual_seed(seed))


def tensor_hash(p):
    return hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def frozen_hash(model):
    result = hashlib.sha256()
    for k,p in model.named_parameters():
        if not p.requires_grad:
            result.update(k.encode()); result.update(p.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def preflight():
    assert not (OUT/'PREFLIGHT.json').exists()
    started = time.monotonic()
    native, norm = load()
    entry = cfg()['windows']['train'][0]
    h = condition(entry, norm, 8); y = target(entry, norm); z = noise(h, 42)
    torch.cuda.reset_peak_memory_stats()
    result = real_checks(native, h, y, z)
    torch.cuda.synchronize()
    previous_failures = sum(json.loads(s).get('conservative_gpu_seconds',0.)
        for s in (OUT/'events.jsonl').read_text().splitlines() if json.loads(s)['event']=='preflight_failed')
    result.update(passed=True, training_entry=entry, members=8,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        conservative_gpu_seconds=time.monotonic()-started+previous_failures, checkpointing_required=False,
        code_hashes=code_hashes(), config_sha256=digest(OUT/'config.json'))
    dump(OUT/'PREFLIGHT.json', result)
    event('preflight_passed', **result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def save(model, step):
    path = OUT/'weights'/f'{model.arm}_{step}.pt'
    assert step in (0,500,1000) and not path.exists()
    path.parent.mkdir(exist_ok=True)
    state = model.trainable_state()
    torch.save(dict(arm=model.arm, step=step, state=state,
        config_sha256=digest(OUT/'config.json')), path)
    result = dict(arm=model.arm, step=step, path=str(path.relative_to(ROOT)), sha256=digest(path),
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    event('checkpoint', **result)
    return result


@torch.no_grad()
def metric_rows(p, y, norm, unbiased, leads=False):
    p, y = p.double(), y.double()
    std = norm['std'][0].double()
    error = p.mean(0)-y
    var = p.var(0, unbiased=False)
    m = len(p)
    if unbiased:
        ordered = p.sort(0).values
        w = (2*torch.arange(m,device=p.device,dtype=p.dtype)-m+1).reshape(m,1,1,1,1)
        field_crps = (p-y).abs().mean(0) - (ordered*w).sum(0)/(m*(m-1))
    else:
        field_crps = empirical_crps(p,y)
    rows = []
    for ch, name in [(None,'overall'),(0,'u'),(1,'v')]:
        sl = slice(None) if ch is None else slice(ch,ch+1)
        crps = field_crps[sl].mean()
        es = scores(p[:,sl],y[sl],True)[1] if unbiased else energy_score(p[:,sl],y[sl])
        tm = scores(p[:,sl].mean(2),y[sl].mean(1),unbiased)[0]
        rows.append(dict(channel=name, lead=0, crps=float(crps), es=float(es),
            score=float(crps+.1*es), smse=float(error[sl].square().mean()),
            mse=float((error*std)[sl].square().mean()), variance=float((var*std.square())[sl].mean()),
            standardized_variance=float(var[sl].mean()), time_mean_crps=float(tm)))
    if leads:
        for ch, name in enumerate('uv'):
            for lead in range(20):
                rows.append(dict(channel=name, lead=lead+1, crps=float(field_crps[ch,lead].mean()),
                    smse=float(error[ch,lead].square().mean()), mse=float((error*std)[ch,lead].square().mean()),
                    variance=float((var*std.square())[ch,lead].mean()), standardized_variance=float(var[ch,lead].mean())))
    return rows


def write_scores(rows):
    with (OUT/'scores.jsonl').open('a') as f:
        for row in rows:
            f.write(json.dumps(row,allow_nan=False)+'\n')


@torch.no_grad()
def evaluate(models, norm, split, step, aliases=None):
    c = cfg()
    m = c['development_members'] if split == 'development' else c['test_members']
    for index, entry in enumerate(c['windows'][split]):
        h = condition(entry,norm,m)
        z = noise(h,c['development_noise_root' if split=='development' else 'test_noise_root']+index)
        # Finish generation without fetching the current truth.
        paths = {arm: model(h,z) for arm,model in models.items()}
        hashes = {arm:tensor_hash(p) for arm,p in paths.items()}
        event('generated', split=split, index=index, step=step, noise_sha256=tensor_hash(z),
            history_sha256=tensor_hash(h[0]), outputs=hashes, network_calls=4*len(models), members=m)
        y = target(entry,norm)
        for arm,p in paths.items():
            metrics = metric_rows(p,y,norm,split=='development',leads=split=='test')
            names = [arm]+([a for a, source in (aliases or {}).items() if source==arm])
            for name in names:
                write_scores([dict(**entry, arm=name, step=step if isinstance(step,int) else step[name],
                    members=m, estimator='U' if split=='development' else 'empirical', **r) for r in metrics])
        print('EVAL',split,step,index+1,len(c['windows'][split]),flush=True)


def rows():
    return [json.loads(s) for s in (OUT/'scores.jsonl').read_text().splitlines()]


def macro(values, metric):
    groups = {}
    for r in values:
        groups.setdefault(r['trajectory'],[]).append(r[metric])
    return {k:float(np.mean(v)) for k,v in groups.items()}


def select():
    data = [r for r in rows() if r['split']=='development' and r['channel']=='overall' and r['lead']==0]
    chosen = dict(N=0)
    candidates = []
    for arm in ('F','O','C'):
        available = sorted(set(r['step'] for r in data if r['arm']==arm))
        options = []
        for step in available:
            rr = [r for r in data if r['arm']==arm and r['step']==step]
            values = {metric:float(np.mean(list(macro(rr,metric).values())))
                for metric in ('score','crps','es','smse','mse','variance','time_mean_crps')}
            options.append((values['score'],step))
            candidates.append(dict(arm=arm,step=step,**values))
        chosen[arm] = min(options)[1]
    selection = dict(selected=chosen, candidates=candidates, config_sha256=digest(OUT/'config.json'),
        rule='trajectory mean of window mean CRPS_U+0.1ES_U; minimum; ties earlier; frozen before test',
        exploratory_old_test=True)
    dump(OUT/'selection.json',selection)
    event('selection_frozen',**selection)
    return chosen


def load_selected(native, selected):
    result = {'N':Forecast(native,'N')}
    for arm,step in selected.items():
        if step == 0:
            continue
        model = Forecast(native,arm)
        ck = torch.load(OUT/'weights'/f'{arm}_{step}.pt',map_location='cuda',weights_only=False)
        assert ck['config_sha256']==digest(OUT/'config.json')
        missing,unexpected = model.load_state_dict(ck['state'],strict=False)
        assert not unexpected and not set(ck['state']) & set(missing)
        result[arm] = model
    return result


def train():
    assert not (OUT/'TRAIN_STARTED.json').exists(), 'Formal run already started; no automatic restart.'
    pre = read(OUT/'PREFLIGHT.json')
    assert pre['passed']
    assert pre['code_hashes']==code_hashes(), 'Code changed after preflight.'
    started = time.monotonic()
    native,norm = load()
    models = {arm:Forecast(native,arm) for arm in ('F','O','C')}
    opts = {arm:model.optimizer() for arm,model in models.items()}
    initial = {arm:model.trainable_state() for arm,model in models.items()}
    frozen = {arm:frozen_hash(model) for arm,model in models.items()}
    counts = {arm:dict(total=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)) for arm,model in models.items()}
    dump(OUT/'TRAIN_STARTED.json',dict(code_hashes=code_hashes(), config_sha256=digest(OUT/'config.json'),
        utc=time.time(), counts=counts, preflight_seconds=pre['conservative_gpu_seconds']))
    checkpoints = [save(model,0) for model in models.values()]
    evaluate({'N':Forecast(native,'N')},norm,'development',0,aliases=dict(F='N',O='N',C='N'))
    common = 0
    durations = []
    stop = 'fixed_1000_updates_complete'
    for item in cfg()['schedule']:
        elapsed = pre['conservative_gpu_seconds']+time.monotonic()-started
        reserve = cfg()['evaluation_reserve_seconds']
        estimate = max(durations[-10:] or [60.])*1.5
        if elapsed+reserve+estimate >= cfg()['max_gpu_seconds']:
            stop='two_hour_budget_guard'; break
        round_start = time.monotonic()
        entry = cfg()['windows']['train'][item['window']]
        h = condition(entry,norm,8); z=noise(h,item['noise_seed']); y=target(entry,norm)
        for arm,model in models.items():
            torch.cuda.reset_peak_memory_stats()
            begin=time.monotonic()
            opts[arm].zero_grad(set_to_none=True)
            p=model(h,z,checkpointing=pre['checkpointing_required'])
            loss=objective(p,y)
            assert torch.isfinite(loss), (arm,item)
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            opts[arm].step()
            torch.cuda.synchronize()
            event('update',arm=arm,step=item['update'],window=item['window'],noise_seed=item['noise_seed'],
                loss=float(loss),gradient_norm_before_clip=float(grad),seconds=time.monotonic()-begin,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())
            del p,loss
        common=item['update']
        durations.append(time.monotonic()-round_start)
        if common % 25 == 0:
            print('COMMON',common,'SECONDS',pre['conservative_gpu_seconds']+time.monotonic()-started,flush=True)
        if common in (500,1000):
            checkpoints += [save(model,common) for model in models.values()]
            with torch.no_grad():
                evaluate(models,norm,'development',common)
    changed={}
    for arm,model in models.items():
        assert frozen_hash(model)==frozen[arm]
        delta={k:float((v-initial[arm][k]).abs().max()) for k,v in model.trainable_state().items()}
        for group in ('base.net.up2.','base.net.out.'):
            assert common==0 or any(v>0 for k,v in delta.items() if k.startswith(group))
        if arm in ('O','C') and common>1:
            assert any(v>0 for k,v in delta.items() if k.startswith('update.'))
        changed[arm]=dict(frozen_unchanged=True,max_abs_delta=delta)
    dump(OUT/'training_summary.json',dict(common_updates=common,stop=stop,checkpoints=checkpoints,
        parameter_updates=changed,counts=counts,conservative_gpu_seconds=pre['conservative_gpu_seconds']+time.monotonic()-started,
        formal_runs=3,checkpointing=pre['checkpointing_required']))
    selected=select()
    test_models=load_selected(native,selected)
    aliases={arm:'N' for arm,step in selected.items() if step==0 and arm!='N'}
    with torch.no_grad():
        evaluate(test_models,norm,'test',selected,aliases)
    # Optional intervention is fixed to first two development trajectories, start400.
    if selected['C']:
        c=test_models['C']; interventions=[]
        for index,entry in enumerate(cfg()['windows']['development']):
            if entry['start']!=400 or len(interventions)>=2:
                continue
            with torch.no_grad():
                h=condition(entry,norm,16); z=noise(h,cfg()['development_noise_root']+index)
                on=c(h,z); off=c(h,z,feedback=False); y=target(entry,norm)
                interventions.append(dict(**entry,on_score=float(objective(on,y)),off_score=float(objective(off,y)),
                    prediction_rms_delta=float((on-off).square().mean().sqrt())))
        dump(OUT/'feedback_intervention.json',interventions)
    torch.cuda.synchronize()
    elapsed=pre['conservative_gpu_seconds']+time.monotonic()-started
    assert elapsed <= cfg()['max_gpu_seconds'], elapsed
    total_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert total_bytes<=cfg()['max_cache_bytes']
    dump(OUT/'COMPLETE.json',dict(common_updates=common,stop=stop,selected=selected,
        conservative_gpu_seconds=elapsed,new_prediction_cache_bytes=0,total_artifact_bytes=total_bytes,
        four_backbone_calls_per_member=True,code_hashes=code_hashes(),checkpoint_hashes=checkpoints,
        test_inference_cases=len(cfg()['windows']['test'])*len(test_models),test_repeats=1,
        per_arm_formal_updates=dict.fromkeys(('F','O','C'),common)))
    print('COMPLETE',common,elapsed,selected,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','preflight','train','report','verify'])
    parser.add_argument('--data',type=Path)
    args=parser.parse_args()
    configure(); torch.manual_seed(42)
    OUT.mkdir(exist_ok=True)
    if args.stage=='prepare':
        assert args.data is not None
        prepare(args.data)
    elif args.stage=='preflight':
        begin=time.monotonic()
        try:
            preflight()
        except Exception as exc:
            torch.cuda.synchronize()
            event('preflight_failed',error=repr(exc),conservative_gpu_seconds=time.monotonic()-begin)
            raise
    elif args.stage=='train': train()
    else:
        from research.coupled_state_feedback.reporting import report, verify
        (report if args.stage=='report' else verify)()


if __name__=='__main__':
    main()
