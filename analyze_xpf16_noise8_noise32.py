"""Standard-library recomputation for only the three requested methods."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

OUT = Path(__file__).resolve().parent / 'results/reliability/xpf16_noise8_noise32'
METRICS = dict(standardized_interval_score=-1, standardized_width=-1, relative_aurc=-1,
               top10_energy=1, per_frame_top10_energy=1, worst_frame_coverage=1,
               worst_patch_coverage=1, coverage_error=-1)


def key(row):
    return tuple(row[k] for k in ('split', 'group', 'channel', 'map', 'variant'))


def primary(row, metric):
    return abs(row['coverage'] - .9) if metric == 'coverage_error' else row[metric]


def main():
    rows = json.loads((OUT / 'per_case.json').read_text())
    assert len(rows) == 5616 and {r['map'] for r in rows} == {'XPF16', '8-noise', '32-noise'}
    assert {r['variant'] for r in rows} == {'memory_dynamic'}
    seen, points, buckets = set(), {}, defaultdict(list)
    for row in rows:
        identity = tuple(row[k] for k in ('split', 'file', 'start', 'channel', 'map', 'variant'))
        assert identity not in seen
        seen.add(identity)
        assert row['mse'] == points.setdefault(identity[:4], row['mse'])
        if row['start'] >= 20:
            buckets[key(row)].append(row)
    numeric = [k for k, v in rows[0].items() if isinstance(v, float)]
    computed = {}
    for identity, records in buckets.items():
        assert set(Counter(r['file'] for r in records).values()) == {38}
        computed[identity] = {k: mean(r[k] for r in records) for k in numeric}
    checks = 0
    for row in json.loads((OUT / 'summary.json').read_text()):
        if row['period'] != 'warm' or row['group'] == 'macro':
            continue
        for metric, value in computed[key(row)].items():
            assert abs(value - row[metric]) < 1e-10
            checks += 1
    comparisons = json.loads((OUT / 'regression_counts.json').read_text())
    assert len(comparisons) == 2 and {r['reference'] for r in comparisons} == {'8-noise', '32-noise'}
    failures = {}
    for row in comparisons:
        counts = dict(wins=0, ties=0, regressions=0)
        for identity, values in computed.items():
            if identity[-2:] != ('XPF16', 'memory_dynamic'):
                continue
            previous = computed[identity[:3] + (row['reference'], 'memory_dynamic')]
            for metric, direction in METRICS.items():
                a, b = primary(values, metric), primary(previous, metric)
                gain = direction * (a - b)
                category = 'wins' if gain > 1e-6 else 'regressions' if gain < -1e-6 else 'ties'
                counts[category] += 1
                if category == 'regressions':
                    failures[(row['reference'], *identity[:3], metric)] = (a, b, gain)
        assert sum(counts.values()) == 96 and all(counts[k] == row[k] for k in counts)
    saved = json.loads((OUT / 'counterexamples.json').read_text())
    assert len(saved) == len(failures) == 56
    for row in saved:
        identity = tuple(row[k] for k in ('reference', 'split', 'group', 'channel', 'metric'))
        expected = failures.pop(identity)
        assert all(abs(row[k] - value) < 1e-10 for k, value in zip(('value', 'baseline', 'directional_gain'), expected))
    assert not failures
    macros = json.loads((OUT / 'test_macro.json').read_text())
    for macro in macros:
        cells = [v for k, v in computed.items() if k[0] == 'test' and k[-2:] == (macro['method'], 'memory_dynamic')]
        assert len(cells) == 6
        for metric, value in macro.items():
            if isinstance(value, float):
                if metric == 'worst_group_coverage':
                    expected = mean(min(v['coverage'] for k, v in computed.items()
                        if k[0] == 'test' and k[2] == channel and
                        k[-2:] == (macro['method'], 'memory_dynamic')) for channel in ('u', 'v'))
                else:
                    expected = mean(v[metric] for v in cells)
                assert abs(value - expected) < 1e-10
    crps_groups = json.loads((OUT / 'crps_group_comparisons.json').read_text())
    assert len(crps_groups) == 48
    for row in crps_groups:
        prefix = tuple(row[k] for k in ('split', 'group', 'channel'))
        metric = row['distribution'] + '_crps'
        a = computed[prefix + ('XPF16', 'memory_dynamic')][metric]
        b = computed[prefix + (row['reference'], 'memory_dynamic')][metric]
        assert abs(a-row['XPF16']) < 1e-10 and abs(b-row['baseline']) < 1e-10
        assert abs(b-a-row['gain']) < 1e-10
    trajectories = defaultdict(list)
    for row in rows:
        if row['split'] == 'test' and row['start'] >= 20:
            trajectories[row['file'], row['channel'], row['map']].append(row)
    paired = []
    for (filename, channel, method), records in sorted(trajectories.items()):
        if method != 'XPF16':
            continue
        score = mean(r['standardized_interval_score'] for r in records)
        for reference in ('8-noise', '32-noise'):
            baseline = mean(r['standardized_interval_score'] for r in trajectories[filename, channel, reference])
            paired.append(dict(file=filename, channel=channel, reference=reference, score=score, baseline_score=baseline,
                               relative_score_change_percent=100 * (score / baseline - 1), improved=score < baseline - 1e-6))
    assert len(paired) == 48
    (OUT / 'paired_test_scores.json').write_text(json.dumps(paired, indent=2) + '\n')
    result = dict(passed=True, scope_exactly_three_methods=True, window_channel_rows=5616,
        independent_summary_numeric_checks=checks, primary_comparisons=2, cells_per_comparison=96,
        tolerance=1e-6, all_counterexamples_verified=len(saved), point_mse_unchanged=True,
        paired_score_improvements={name: sum(r['improved'] for r in paired if r['reference'] == name)
                                   for name in ('8-noise', '32-noise')},
        caveat='Repeated historicaldata; no independent statistical or formalcoverage confirmation.All three share the zero-noise point and online spatiotemporal calibrator; own calibration states. Gaussian/Laplace CRPS are assumed CDF completions.')
    (OUT / 'analysis_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
