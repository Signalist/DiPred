"""Summarize all 129 levels and calibrated ensemble comparisons."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from clean_reliability_common import ROOT, dump
from train_xpf_quantiles129 import OUT
from xpf_quantiles129_head import quantile_levels


def read(name):
    return json.loads((OUT / name).read_text())


def group_aggregate(rows, vector_fields, scalar_fields, method_key):
    buckets = defaultdict(list)
    for row in rows:
        if row['start'] < 20:
            continue
        periods = ('full_warm', 'historic_warm') if row['start'] <= 760 else ('full_warm',)
        for period in periods:
            key = (row['split'], period, row['group'], row['channel'], row['mode'], row.get(method_key, 'XPF'))
            buckets[key].append(row)
    result = []
    for key, records in buckets.items():
        row = dict(zip(('split', 'period', 'group', 'channel', 'mode', 'method'), key), windows=len(records))
        for metric in vector_fields:
            row[metric] = np.mean([r[metric] for r in records], axis=0).tolist()
        for metric in scalar_fields:
            values = [r[metric] for r in records if r[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
        result.append(row)
    return result


def main():
    assert read('evaluation_complete.json')['passed']
    qrows, erows = read('quantile_per_case.json'), read('ensemble_per_case.json')
    qgroups = group_aggregate(qrows, ('coverage', 'pinball'), ('approx_crps',), 'method')
    egroups = group_aggregate(erows, (), ('empirical_crps', 'fair_crps', 'empirical_interval_coverage',
        'proxy_interval_coverage', 'proxy_interval_width', 'proxy_interval_score'), 'method')
    dump(OUT / 'quantile_group_summary.json', qgroups)
    dump(OUT / 'ensemble_group_summary.json', egroups)
    levels = quantile_levels()
    summary, tables = [], []
    for split in ('development', 'test'):
        for period in ('historic_warm', 'full_warm'):
            for mode in ('raw', 'online'):
                cells = [r for r in qgroups if r['split'] == split and r['period'] == period and r['mode'] == mode]
                assert len(cells) == 6
                coverage = np.mean([r['coverage'] for r in cells], axis=0)
                pinball = np.mean([r['pinball'] for r in cells], axis=0)
                summary.append(dict(split=split, period=period, mode=mode, method='XPF',
                    approx_crps=float(np.mean([r['approx_crps'] for r in cells])),
                    mean_abs_calibration_error_pp=float(np.mean(np.abs(coverage - levels)) * 100)))
                for i, level in enumerate(levels):
                    tables.append(dict(split=split, period=period, mode=mode, percentile=100 * level,
                        empirical_cdf=float(coverage[i]), calibration_error_pp=float(100 * (coverage[i] - level)),
                        standardized_pinball=float(pinball[i])))
            for method in ('8-noise', '32-noise'):
                for mode in ('raw', 'online'):
                    cells = [r for r in egroups if r['split'] == split and r['period'] == period and
                             r['mode'] == mode and r['method'] == method]
                    assert len(cells) == 6
                    row = dict(split=split, period=period, method=method, mode=mode)
                    for metric in ('empirical_crps', 'fair_crps', 'empirical_interval_coverage',
                                   'proxy_interval_coverage', 'proxy_interval_width', 'proxy_interval_score'):
                        values = [r[metric] for r in cells if r[metric] is not None]
                        row[metric] = float(np.mean(values)) if values else None
                    summary.append(row)
    dump(OUT / 'macro_summary.json', summary)
    dump(OUT / 'quantile_summary.json', tables)
    with (OUT / 'quantile_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)
    raw = {r['percentile']: r for r in tables if r['split'] == 'test' and r['period'] == 'full_warm' and r['mode'] == 'raw'}
    online = {r['percentile']: r for r in tables if r['split'] == 'test' and r['period'] == 'full_warm' and r['mode'] == 'online'}
    lines = ['# XPF 129 个分位水平：测试集完整表', '',
             '测试窗口 2–42；Re 与 u/v 通道等权。覆盖率为 P(Y ≤ Qτ)，偏差单位为百分点；Pinball 按训练标准差标准化。', '',
             '|目标分位|原始覆盖率|在线覆盖率|在线偏差|在线 Pinball↓|', '|---:|---:|---:|---:|---:|']
    for level in levels:
        percentile = 100 * level
        a, b = raw[percentile], online[percentile]
        lines.append('|{:g}%|{:.3f}%|{:.3f}%|{:+.3f}|{:.6f}|'.format(percentile,
            100 * a['empirical_cdf'], 100 * b['empirical_cdf'], b['calibration_error_pp'], b['standardized_pinball']))
    (OUT / 'quantile_table.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps([r for r in summary if r['split'] == 'test' and r['period'] == 'full_warm'], indent=2), flush=True)


if __name__ == '__main__':
    main()
