"""Recompute quantile and ensemble metrics from released processed summaries."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent / 'results/reliability'


def read(directory, filename):
    return json.loads((ROOT / directory / filename).read_text())


def audit(directory, dimensions, metrics, expected_groups, expected_trajectories):
    groups = read(directory, 'group_summary.json')
    trajectories = read(directory, 'trajectory_summary.json')
    summary = read(directory, 'summary.json')
    assert len(groups) == expected_groups and len(trajectories) == expected_trajectories
    buckets = defaultdict(list)
    for row in trajectories:
        key = tuple(row[k] for k in ('split', 'group', 'channel') + dimensions)
        buckets[key].append(row)
    for row in groups:
        key = tuple(row[k] for k in ('split', 'group', 'channel') + dimensions)
        cells = buckets[key]
        total = sum(cell['windows'] for cell in cells)
        assert total == row['windows']
        for metric in metrics:
            expected = sum(cell[metric] * cell['windows'] for cell in cells) / total
            assert abs(row[metric] - expected) < 1e-10
    for row in summary:
        cells = [group for group in groups if group['split'] == row['split'] and
                 all(group[key] == row[key] for key in dimensions)]
        assert len(cells) == 6
        for metric in metrics:
            assert abs(row[metric] - mean(cell[metric] for cell in cells)) < 1e-10
    return dict(groups=len(groups), trajectories=len(trajectories), summary_rows=len(summary))


def main():
    quantiles = audit('quantiles19', ('mode', 'percentile'), ('empirical_cdf', 'pinball'), 456, 1824)
    native = audit('native_ensemble_crps', ('samples', 'estimator'),
                   ('crps', 'raw_crps', 'abs_term', 'half_pair_term', 'raw_abs_term', 'raw_half_pair_term'), 48, 192)
    for row in read('native_ensemble_crps', 'group_summary.json'):
        assert abs(row['crps'] - (row['abs_term'] - row['half_pair_term'])) < 2e-7
        assert abs(row['raw_crps'] - (row['raw_abs_term'] - row['raw_half_pair_term'])) < 2e-8
    print(json.dumps(dict(passed=True, quantiles=quantiles, ensemble_crps=native), indent=2))


if __name__ == '__main__':
    main()
