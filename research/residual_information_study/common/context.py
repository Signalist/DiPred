"""Read-only experiment context and shared checks for the three work lines."""
from functools import lru_cache
import json
from pathlib import Path
import time

from research.observed_residual_feedback import run as observed

ROOT = observed.ROOT
OUT = ROOT / 'results/residual_information_study'
PREVIOUS = observed.OUT
CACHE = ROOT / '.cache/residual_information_study'
SOURCE = ROOT / 'research/residual_information_study'
REFERENCE = '098e2774054877f4b574d24706d038d394e723cc'
GPU_SECONDS = 7200
CACHE_BYTES = 8 * 1024**3


def read(path):
    return observed.read(path)


def sha(path):
    return observed.sha(path)


@lru_cache(maxsize=1)
def schedule():
    return read(PREVIOUS / 'schedule.json')


@lru_cache(maxsize=1)
def windows():
    return read(PREVIOUS / 'windows.json')


@lru_cache(maxsize=1)
def settings():
    config = read(OUT / 'FROZEN.json')
    assert config['reference'] == REFERENCE
    for path, checksum in config['sha256'].items():
        assert sha(ROOT / path) == checksum, path
    return config


def load_native():
    """Load the exact archived checkpoint and the original train-only norm."""
    config = settings()
    native, norm, original = observed.load()
    assert config['data_root'] == original['data_root']
    return native, norm, config


def budget_records():
    path = OUT / 'budget.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def remaining_gpu_seconds():
    elapsed = 0.
    starts = {}
    for row in budget_records():
        if row['event'] == 'start':
            starts[row['id']] = row['monotonic']
        else:
            elapsed += row['seconds']
            starts.pop(row['id'])
    # Includes the elapsed time of the currently running stage.
    elapsed += sum(max(0., time.monotonic() - start) for start in starts.values())
    return max(0., GPU_SECONDS - elapsed)


def new_cache_bytes():
    return sum(path.stat().st_size for path in CACHE.rglob('*') if path.is_file())
