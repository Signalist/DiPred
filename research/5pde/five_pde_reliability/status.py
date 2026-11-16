"""Show completion counts without opening test predictions."""
from pathlib import Path
from common import OUT

SCENARIOS = ('cylinder', 'controlled_cylinder', 'fsi', 'foil', 'combustion')
ARMS = ('X', 'XP', 'XPF')
NOISE = ('8-noise', '32-noise', '256-noise')
for scenario in SCENARIOS:
    heads = sum((OUT / 'heads' / scenario / f'seed_{seed}' / f'{arm}.pt').exists()
                for seed in (42, 123, 2026) for arm in ARMS)
    evaluated = sum((OUT / 'evaluated' / scenario / f'{arm}_{seed}.pt').exists()
                    for seed in (42, 123, 2026) for arm in ARMS)
    evaluated += sum((OUT / 'evaluated' / scenario / f'{arm}_500000.pt').exists()
                     for arm in NOISE)
    print(f'{scenario}: heads {heads}/9; full-test evaluations {evaluated}/12')
