"""Single entry point for the staged residual-information study."""
import argparse
import fcntl
import importlib
import os
from pathlib import Path
import sys
import time
import uuid

os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from reliability.backbone import configure
from research.observed_residual_feedback.run import dump, jsonl
from research.residual_information_study.common.context import (
    CACHE, CACHE_BYTES, GPU_SECONDS, OUT, ROOT, SOURCE, budget_records, new_cache_bytes,
    read, remaining_gpu_seconds, settings, sha,
)
from research.residual_information_study.common.prepare import prepare

LINES = dict(A='information_probe', B='correction_diagnosis', C='representation_controls')
SELECTION_FILES = dict(probe_selection='information_probe/FROZEN_A.json',
                       diagnosis_summary='correction_diagnosis/summary.json',
                       representation_selection='representation_controls/SELECTION.json')


def freeze_selection():
    settings()
    path = OUT / 'FROZEN_SELECTION.json'
    assert not path.exists(), 'Unified selection already frozen'
    completed = {(r['line'], r['stage']) for r in budget_records() if r['event'] == 'finish' and r['passed']}
    assert {(line, 'development') for line in LINES} <= completed
    sealed = {name + '_sha256': sha(OUT / file) for name, file in SELECTION_FILES.items()}
    dump(path, dict(**sealed, utc_ns=time.time_ns(), config_sha256=sha(OUT / 'FROZEN.json'),
        representation_selected=read(OUT / SELECTION_FILES['representation_selection'])['selected']))
    print('All three development lines sealed for one historical test pass', flush=True)


def verify_selection():
    selection = read(OUT / 'FROZEN_SELECTION.json')
    assert selection['config_sha256'] == sha(OUT / 'FROZEN.json')
    for name, file in SELECTION_FILES.items():
        assert selection[name + '_sha256'] == sha(OUT / file)
    return selection


def run_line(line, stage):
    settings()
    if stage == 'test':
        assert line in ('A', 'C')
        verify_selection()
    else:
        assert not (OUT / 'FROZEN_SELECTION.json').exists()
    lock = CACHE / 'gpu.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = budget_records()
        completed = {(r['line'], r['stage']) for r in records if r['event'] == 'finish' and r['passed']}
        assert (line, stage) not in completed, f'{line}/{stage} already completed'
        assert sum(r['event'] == 'start' for r in records) == sum(r['event'] == 'finish' for r in records)
        assert remaining_gpu_seconds() > 0 and new_cache_bytes() <= CACHE_BYTES
        module = importlib.import_module(f'research.residual_information_study.{LINES[line]}.stage')
        files = [*SOURCE.joinpath(LINES[line]).rglob('*.py'), Path(__file__).resolve()]
        identities = {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}
        identifier = str(uuid.uuid4())
        started = time.monotonic()
        log = OUT / 'budget.jsonl'
        base = dict(id=identifier, line=line, stage=stage)
        jsonl(log, dict(**base, event='start', monotonic=started, utc_ns=time.time_ns(),
                        code_sha256=identities, cache_bytes=new_cache_bytes()))
        torch.cuda.reset_peak_memory_stats()
        passed = False
        try:
            getattr(module, stage)()
            torch.cuda.synchronize()
            assert time.monotonic() - started <= GPU_SECONDS
            assert remaining_gpu_seconds() > 0 and new_cache_bytes() <= CACHE_BYTES
            passed = True
        finally:
            jsonl(log, dict(**base, event='finish', utc_ns=time.time_ns(),
                seconds=time.monotonic() - started, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                cache_bytes=new_cache_bytes(), passed=passed))
        print(f'{line}/{stage} finished', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'A', 'B', 'C', 'freeze', 'test-A', 'test-C', 'report', 'all'))
    args = parser.parse_args()
    configure()
    if args.stage == 'prepare':
        prepare()
    elif args.stage in LINES:
        run_line(args.stage, 'development')
    elif args.stage == 'freeze':
        freeze_selection()
    elif args.stage.startswith('test-'):
        run_line(args.stage[-1], 'test')
    elif args.stage == 'report':
        from research.residual_information_study.common.reporting import report
        report(verify_selection())
    else:
        prepare()
        for line in LINES:
            run_line(line, 'development')
        freeze_selection()
        for line in ('A', 'C'):
            run_line(line, 'test')
        from research.residual_information_study.common.reporting import report
        report(verify_selection())


if __name__ == '__main__':
    main()
