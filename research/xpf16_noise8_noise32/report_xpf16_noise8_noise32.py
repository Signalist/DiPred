"""Audit and report the three methods under identical online calibration."""

import csv
import json
import math
from collections import defaultdict
from statistics import mean

import numpy as np
import torch
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from clean_reliability_common import ROOT, configure, dump, manifest
from evaluate_xpf16_noise8_noise32 import OUT, NAMES, Provider
from conditional_online_risk import risk_features
from memory_quantile_online_risk import MemoryQuantileCalibrator
from experiment_reliability_transport_e12 import target
from evaluate_reliability_transport_e12 import summarize
from audit_fourcall_uq_extensions import regression_tables, test_macro


FIGURES = ROOT / 'results/xpf16_noise8_noise32_20261111'


def read(path):
    return json.loads(path.read_text())


def key(row):
    return tuple(row[k] for k in ('split', 'file', 'start', 'channel'))


@torch.no_grad()
def audit(rows):
    """Check frozen controls, actual eight-path spread and causal updates."""
    assert read(OUT / 'complete.json')['passed'] and len(rows) == 5616
    old = read(ROOT / 'runs/requested16_28_noise32_20261110/per_case.json')
    lookup = {key(r) + (r['map'],): r for r in old}
    points = {key(r): r['mse'] for r in old}
    previous_crps = read(ROOT / 'runs/xpf16_crps_20261111/per_case_completions.json')
    crps_lookup = {key(r) + (r['method'], r['distribution']): r for r in previous_crps}
    aliases = {'XPF16': ('selected16', 'F0'), '32-noise': ('noise32', 'noise32')}
    controls = 0
    for row in rows:
        assert row['mse'] == points[key(row)]
        if row['map'] in aliases:
            interval_name, crps_name = aliases[row['map']]
            before = lookup[key(row) + (interval_name,)]
            for metric, value in before.items():
                if isinstance(value, float) and metric in row:
                    assert row[metric] == value, metric
            for family in ('gaussian', 'laplace'):
                reference = crps_lookup[key(row) + (crps_name, family)]
                assert row[family + '_crps'] == reference['crps']
                assert row['raw_' + family + '_crps'] == reference['raw_crps']
            controls += 1
    snapshot = np.load(OUT / 'audit_snapshot.npz')
    sample_std = snapshot['samples8_physical'].astype(np.float64).std(axis=0, ddof=1)
    std_error = float(np.max(np.abs(sample_std - snapshot['std8_physical'])))
    assert std_error < 1e-7
    discrepancies = []
    y = snapshot['residual'].astype(np.float64)
    for name in NAMES:
        lo, hi = [snapshot[name + '_' + field].astype(np.float64) for field in ('lower', 'upper')]
        mid, radius = (lo + hi) / 2, (hi - lo) / 2
        sigma = radius / 1.6448536269514722
        z = (y - mid) / sigma
        gaussian = sigma * (z * (2 * ndtr(z) - 1) + math.sqrt(2 / math.pi) * np.exp(-z * z / 2) - 1 / math.sqrt(math.pi))
        b = radius / math.log(10)
        laplace = np.abs(y - mid) + b * np.exp(-np.abs(y - mid) / b) - .75 * b
        for family, expected in (('gaussian', gaussian), ('laplace', laplace)):
            discrepancies.append(float(np.max(np.abs(expected - snapshot[name + '_' + family]))))
    assert max(discrepancies) < 2e-6
    events = [json.loads(line) for line in (OUT / 'events.jsonl').read_text().splitlines()]
    assert len(events) == 174
    for a, b in zip(events[::2], events[1::2]):
        expected = 'all_raw_forecasts_issued' if a['split'] == 'calibration' else 'all3_complete_intervals_issued'
        assert a['event'] == expected and b['event'] == 'full20_target_revealed'
        assert (a['split'], a['start']) == (b['split'], b['start']) and a['time'] <= b['time']

    initial = torch.load(OUT / 'calibration.pt', map_location='cuda', weights_only=False)
    provider = Provider()
    controls_by_name = {n: MemoryQuantileCalibrator('memory_dynamic', initial[n], 12) for n in NAMES}
    row_lookup = {key(r) + (r['map'],): r for r in rows}
    metric_checks = 0
    for start in (0, 20):
        h, point, bounds = provider.bounds('test', start)
        issued = {n: controls_by_name[n].issue(*bounds[n], risk_features(h, point, (bounds[n][1] - bounds[n][0]) / 2)) for n in NAMES}
        residual = (target('test', start) - provider.norm['mean']) / provider.norm['std'] - point
        for name in NAMES:
            lower, upper = [v.cpu().numpy().astype(np.float64) for v in issued[name]]
            truth = residual.cpu().numpy().astype(np.float64)
            for i, c in ((0, 0), (11, 1)):
                y, lo, hi = truth[i, c], lower[i, c], upper[i, c]
                inside = (y >= lo) & (y <= hi)
                expected = dict(coverage=inside.mean(), standardized_width=(hi - lo).mean(),
                    standardized_interval_score=(hi - lo + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)).mean(),
                    worst_frame_coverage=inside.mean((1, 2)).min(),
                    worst_patch_coverage=inside.reshape(20, 4, 8, 8, 8).mean((0, 2, 4)).min())
                row = row_lookup[('test', manifest()['files']['test'][i]['file'], start, ('u', 'v')[c], name)]
                assert all(abs(row[k] - value) < 2e-6 for k, value in expected.items())
                metric_checks += len(expected)
            try:
                controls_by_name[name].reveal(residual[:, :, :10])
            except RuntimeError:
                pass
            else:
                raise AssertionError('Partial feedback accepted')
            controls_by_name[name].reveal(residual)

        # Common inputs and common initialization must give identical outputs
        # and state updates, independently of the method label.
        identical = [MemoryQuantileCalibrator('memory_dynamic', initial['XPF16'], 12) for _ in NAMES]
        features = risk_features(h, point, (bounds['XPF16'][1] - bounds['XPF16'][0]) / 2)
        outputs = [c.issue(*bounds['XPF16'], features) for c in identical]
        assert all(torch.equal(a, b) for out in outputs[1:] for a, b in zip(outputs[0], out))
        for control in identical:
            control.reveal(residual)
        assert all(torch.equal(getattr(identical[0], field), getattr(control, field))
                   for control in identical[1:] for field in ('shape', 'amplitude', 'beta', 'bias'))
    result = dict(passed=True, unchanged_point_rows=len(rows), exact_old_interval_and_crps_rows=controls,
        actual8_sample_std_max_error=std_error, independent_crps_max_error=max(discrepancies),
        chronological_issue_reveal_pairs=87, real_two_window_numpy_metric_checks=metric_checks,
        identical_input_controllers_exact=True, partial_truth_rejected=True)
    dump(OUT / 'independent_audit.json', result)


def main():
    configure()
    rows = read(OUT / 'per_case.json')
    audit(rows)
    summary = summarize(rows)
    dump(OUT / 'summary.json', summary)
    counts = regression_tables(OUT, summary, summary, ['XPF16'], ['memory_dynamic'],
                              [('8-noise', 'memory_dynamic'), ('32-noise', 'memory_dynamic')])
    assert all(sum(r[k] for k in ('wins', 'ties', 'regressions')) == 96 for r in counts)
    # Preserve both distribution assumptions and every group-level reversal.
    groups = defaultdict(list)
    for row in rows:
        if row['start'] >= 20:
            groups[(row['split'], row['group'], row['channel'], row['map'])].append(row)
    crps_comparisons = []
    for reference in ('8-noise', '32-noise'):
        for family in ('gaussian', 'laplace'):
            for k, records in groups.items():
                if k[-1] == 'XPF16':
                    a = mean(r[family + '_crps'] for r in records)
                    b = mean(r[family + '_crps'] for r in groups[k[:-1] + (reference,)])
                    crps_comparisons.append(dict(split=k[0], group=k[1], channel=k[2], reference=reference,
                        distribution=family, XPF16=a, baseline=b, gain=b-a))
    dump(OUT / 'crps_group_comparisons.json', crps_comparisons)
    columns = ['method', 'coverage', 'standardized_width', 'standardized_interval_score', 'gaussian_crps', 'laplace_crps', 'relative_aurc']
    macros = [dict(method=n, **test_macro(summary, n, 'memory_dynamic')) for n in NAMES]
    dump(OUT / 'test_macro.json', macros)
    FIGURES.mkdir(exist_ok=True)
    with (FIGURES / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(macros)
    formatted = [[r['method'], '{:.3f}%'.format(100*r['coverage'])] +
                 ['{:.6f}'.format(r[k]) for k in columns[2:]] for r in macros]
    labels = ['Method', 'Coverage', 'Width', '90% interval\nscore', 'Gaussian\nCRPS*', 'Laplace\nCRPS*', 'AURC']
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.axis('off')
    table = ax.table(cellText=formatted, colLabels=labels, cellLoc='center', bbox=[0, .35, 1, .58])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#ccd4df')
        cell.set_facecolor('#213c56' if r == 0 else '#eef3f8' if r % 2 else 'white')
        if r == 0:
            cell.set_text_props(color='white', weight='bold')
    ax.text(0, .23, 'Same frozen point P; same memory_dynamic online spatiotemporal calibration for all three.\n'
        'Test windows 2-39; equal Re/channel weights. Target coverage 90%. Lower scores/width/AURC are better.\n'
        '* CRPS uses explicit Gaussian/Laplace completions of issued 5/95% bounds, not a native learned CDF.\n'
        'Previously used data: exploratory comparison.', transform=ax.transAxes, fontsize=10, va='top')
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(FIGURES / ('comparison.' + extension), dpi=220, bbox_inches='tight')
    plt.close(fig)
    report = ['# XPF16、8-noise、32-noise：统一在线时空校准', '',
        '三行均使用同一个冻结的增强State DiPred、相同零噪声点预测P、相同MemoryQuantileCalibrator(memory_dynamic)、相同物理风险特征与更新规则。20帧历史一次预测后续完整20帧，四步去噪99/66/33/0不变。没有重新训练网络。', '',
        '之前的匹配区间表中XPF16与32-noise已经采用此共同控制器；上次单列的原生32样本CRPS是未校准诊断，并非该区间表。本次将三行明确放入统一重放，不声称修改了原本不同的控制器。', '',
        '8-noise实际生成8条路径（500000–500007，是32路径种子的前缀），用样本std构造初始对称区间；32-noise用冻结实际32路径std。XPF16使用原始学习的05/95分位区间。三者分别在同一10条校准轨迹上拟合自己的初始时空形状和幅值，之后各自维护在线状态，不能将32-noise校准系数套给8-noise。', '',
        '共同P不代表区间中点相同：XPF16可输出残差偏移，而采样初始区间对称于P。三个完整区间全部发出后才读取该窗口真值；更新只影响下一窗口。', '',
        '## 测试窗口2–39，Re与u/v等权', '',
        '|方法|覆盖率|宽度↓|90%区间评分↓|高斯补全CRPS↓|Laplace补全CRPS↓|AURC↓|',
        '|---|---:|---:|---:|---:|---:|---:|']
    report += ['|' + '|'.join(r) + '|' for r in formatted]
    xpf = macros[0]
    for reference in macros[1:]:
        report += ['', 'XPF16相对{}：区间评分降低{:.3f}%，宽度降低{:.3f}%，AURC降低{:.3f}%。'.format(
            reference['method'], *[100 * (1 - xpf[k] / reference[k]) for k in
                ('standardized_interval_score', 'standardized_width', 'relative_aurc')])]
    report += ['', '结论：统一在线修正后，XPF16保持较窄区间和较好的区间评分/风险排序；8路径不足以达到32路径的表现。CRPS不能据此宣称稳定领先：XPF16的高斯补全CRPS略低于32-noise，而Laplace补全CRPS更高。覆盖差异和全部分组反例仍需一起看。']
    report += ['', '宽度、区间评分和CRPS按原训练通道std标准化。CRPS必须声明完整CDF：这里分别用高斯与Laplace补全同一05/95区间，保留两套结果，不能把它们称为原始两分位头的完整分布CRPS。覆盖是逐点边际覆盖，不是20帧整体联合覆盖。', '',
        '## 核验及反例', '',
        '全部5616条点预测误差不变；XPF16和32-noise的3744条旧区间、风险排序及两套CRPS逐值复现。8样本std和CRPS由NumPy/Scipy独立重算；实际前两窗口重放通过，半窗口反馈被拒绝。相同输入和初始化下三个控制器输出与更新状态完全一致。', '',
        '原八指标×开发/测试Re×通道共96项比较保留，未只报有利均值：']
    for count in counts:
        report.append('- XPF16 vs {}：{}改善、{}持平、{}退步。'.format(count['reference'], count['wins'], count['ties'], count['regressions']))
    report += ['', '完整逐窗口结果、开发集结果、96项比较与反例、两类CRPS的分组比较及审计见runs/xpf16_noise8_noise32_20261111/。图表和CSV见results/xpf16_noise8_noise32_20261111/。', '',
        '部署时XPF16为4次主干调用+轻量头；8-noise为32次采样调用+4次共同P调用；32-noise为128+4次。此处32路径矩复用缓存，不将缓存IO时间当作部署推理速度。', '',
        '所有开发/测试轨迹已用于历史研究，本轮为探索性公平对照，不是新鲜独立确认；不声称严格条件覆盖、联合覆盖或跨工况保证。未自动推送GitHub。']
    (ROOT / '研究结果_XPF16与8noise32noise统一在线校准_20261111.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(dict(test=macros, comparisons=counts), indent=2), flush=True)


if __name__ == '__main__':
    main()
