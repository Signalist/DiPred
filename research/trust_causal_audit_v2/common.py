"""Audit-only utilities. No optimizer, model fitting, or forecast scoring job."""
import hashlib
import json
from pathlib import Path
import subprocess
import torch

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'research/trust_causal_audit_v2'
OUT = ROOT / 'results/trust_causal_audit_v2'
OLD_CODE = ROOT / 'research/error_guided_spatial_routing'
OLD_OUT = ROOT / 'results/error_guided_spatial_routing'
OLD_CACHE = ROOT / '.cache/error_guided_spatial_routing'
DATA = ROOT.parents[1] / 'OLD/REALPDE/realpdebench-competition/data/train_real'
REFERENCE = '3f5d3fd1e86a19bf33b78f35c861a0021553aa35'
MEMBERS = tuple(range(500000, 500032))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(module):
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode())
        h.update(tensor_hash(v).encode())
    return h.hexdigest()


def code_hashes():
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(CODE.glob('*.py'))}


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def records():
    return json.loads((ROOT / 'results/native_restart_consistency/config.json').read_text())['records']


def load_backbone(device='cuda'):
    from reliability.backbone import configure
    from reliability.networks import DynamicsDiffusion20
    configure()
    ck = torch.load(ROOT / 'reliability/checkpoints/current.pt', map_location=device, weights_only=False)
    native = DynamicsDiffusion20(ck['dynamics_scale']).to(device).eval().requires_grad_(False)
    native.load_state_dict(ck['predictor'], strict=True)
    norm = {k: v.to(device).reshape(2, 1, 1, 1) for k, v in ck['normalization'].items()}
    return native, norm


def protected_files():
    files = []
    for directory in (OLD_CODE, OLD_OUT, OLD_CACHE):
        files.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += [ROOT / 'reliability/checkpoints/current.pt']
    return sorted(files)


def freeze_originals():
    destination = OUT / 'PROTECTED_BEFORE.json'
    if destination.exists():
        raise RuntimeError('An audit has already started; refuse to overwrite its evidence')
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'trust'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == REFERENCE
    dump(destination, {str(p.relative_to(ROOT)): digest(p) for p in protected_files()})
    dump(OUT / 'RUN_STARTED.json', dict(reference=REFERENCE, optimizer_updates=0, training_runs=0,
         formal_test_comparisons=0, new_native_cache_generation=0))


def assert_protected():
    before = json.loads((OUT / 'PROTECTED_BEFORE.json').read_text())
    for name, sha in before.items():
        assert digest(ROOT / name) == sha, name
    assert {str(p.relative_to(ROOT)) for p in protected_files()} == set(before)
    return len(before)
