"""Only the three user-requested methods: selected16, original28,32-noise.

Reuse the already audited original28 frozen replay. Reproduce its intervals
under the same controller on actual windows; do not add learned baselines.
"""
import csv
import json
import shutil
import time
from statistics import mean
import torch

from clean_reliability_common import ROOT, configure, dump, digest
from evaluate_selected16_noise32 import OUT as PAIR, Provider as PairProvider, load
from experiment_selected16_quantile import OUT as TRAIN, ABLATION, new_head, MEMORY
from experiment_reliability_transport_e12 import normalization, PREDICTOR
from experiment_continue_rich_uncertainty import controller
from evaluate_reliability_transport_e12 import summarize
from conditional_online_risk import risk_features
from audit_fourcall_uq_extensions import audit, regression_tables, test_macro

OUT = ROOT / 'runs/requested16_28_noise32_20261110'
ORIGINAL = ROOT / 'runs/feature_ablation28_followup_20261110'
NAMES = ('selected16', 'original28', 'noise32')
MODES = ('memory_dynamic',)


def read(path):
    return json.loads(path.read_text())


class Provider(PairProvider):
    def __init__(self, norm):
        super().__init__(norm)
        scale = load(MEMORY / 'statistics.pt')['memory_scale']
        self.heads['original28'] = new_head('matched_full28', scale)


def main():
    configure()
    assert read(PAIR / 'complete.json')['frozen_artifacts_unchanged']
    assert read(TRAIN / 'independent_audit.json')['passed']
    assert read(ORIGINAL / 'analysis_verification.json')['passed']
    OUT.mkdir(exist_ok=False)
    source_files = [PAIR / 'per_case.json', ORIGINAL / 'per_case.json', TRAIN / 'compact16.pt', ABLATION / 'full.pt',
                    PAIR / 'calibration.pt', ABLATION / 'calibration.pt', PREDICTOR / 'augmented_diffusion/model.pt']
    dump(OUT / 'protocol.json', dict(time=time.time(), source_sha256=digest(ROOT / 'report_requested16_28_noise32.py'),
        predictor_sha256=digest(source_files[-1]), artifacts={str(p.relative_to(ROOT)): digest(p) for p in source_files},
        user_clarification='原来的28，说错了。直接对照原28以及32-noise，不需要另外对照。',
        names=NAMES, modes=MODES, no_other_baselines_evaluated_in_this_report=True,
        original28='Prior complete-source continuation full.pt, same publishedinitialstate and819pairs/2000updates/seed42 asselected16. Reuse frozenactualreplay without newtraining.',
        scope='Alloriginal8metrics/96development-testRe/channelcells/1e-6, onlyselected16vsoriginal28 andnoise32. Oldbroadcomparisonrequest superseded byuser; oldartifacts preserved butnot newcomparison targets.',
        common_center='Allmethods share frozen4call zero-noise P.32-noise uses actual32pathstd with independent recalibration on10IDs and samefixedmemory_dynamic controller. This is not32ensemblemean pointforecast or empirical5/95samplequantiles.',
        online_scope='Onlycomplete20reveals update calibrators; noneuralweight updates. Sharedcalibrator retains its originalphysicalfeatures.16channels describes neuralquantilehead, not allinputs ofentirepipeline.',
        computation='16and28require4DiPredcalls+onehead. Fresh32-noisestd with commonP needs128samplingcalls+4center calls. Cachedmoments reused; no speedup inferred from cachedIO.',
        caveat='Repeated historicaldata; subsetchosen afterpriorablation. Exploratory comparison, no independentstatistical or strictcoverage claim.'))
    rows = read(PAIR / 'per_case.json')
    assert len(rows) == 3744
    old = read(ORIGINAL / 'per_case.json')
    original = [dict(row, map='original28') for row in old if row['map'] == 'ablate_full' and row['variant'] == 'memory_dynamic']
    assert len(original) == 1872
    rows += original
    assert len(rows) == 5616 and {r['map'] for r in rows} == set(NAMES)
    dump(OUT / 'per_case.json', rows)
    summary = summarize(rows)
    dump(OUT / 'summary.json', summary)
    shutil.copy2(PAIR / 'events.jsonl', OUT / 'events.jsonl')
    shutil.copy2(ORIGINAL / 'events.jsonl', OUT / 'original28_events.jsonl')
    for path in (OUT / 'events.jsonl', OUT / 'original28_events.jsonl'):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(events) == 156
        for issued, revealed in zip(events[::2], events[1::2]):
            assert issued['event'] == 'issued' and revealed['event'] == 'revealed'
            assert issued['start'] == revealed['start'] and issued['split'] == revealed['split']
            assert issued['time'] <= revealed['time']
    calibration = load(PAIR / 'calibration.pt')
    calibration['original28'] = load(ABLATION / 'calibration.pt')['full']
    torch.save({name: {mode: {key: value.cpu() for key, value in state.items()} for mode, state in modes.items()}
                for name, modes in calibration.items()}, OUT / 'calibration.pt')
    initial = {name: {'memory_dynamic': modes['separable_integral']} for name, modes in calibration.items()}
    audit(OUT, Provider(normalization()), initial, controller, lambda row: None, feature_builder=risk_features)
    counts = regression_tables(OUT, summary, summary, ('selected16',), MODES,
                               (('original28', 'memory_dynamic'), ('noise32', 'memory_dynamic')))
    assert len(counts) == 2 and all(sum(r[k] for k in ('wins', 'ties', 'regressions')) == 96 for r in counts)
    values = []
    for name in NAMES:
        value = test_macro(summary, name, 'memory_dynamic')
        groups = [r for r in summary if r['map'] == name and r['variant'] == 'memory_dynamic'
                  and r['split'] == 'test' and r['period'] == 'warm' and r['group'] != 'macro']
        value['coverage_error'] = mean(abs(r['coverage'] - .9) for r in groups)
        values.append(dict(method=name, **value))
    dump(OUT / 'test_macro.json', values)
    selected, full, noise = values
    labels = dict(selected16='16通道', original28='原28通道', noise32='32-noise标准差')
    lines = ['# 仅三组：16通道、原28通道、32-noise', '',
        '按用户最新要求，只保留三组正式比较，不再加入屏蔽16、其他风险头或更多训练对照。16通道为梯度2＋坐标2＋历史时间波动2＋去噪特征8＋预测时间变化2，真实16输入卷积；去掉历史/误差投影及位移模块。', '',
        '原28通道复用上一轮已审计full.pt：与16通道有相同源权重起点、819训练对、2000步、seed42、优化器与批次。两者点预测器冻结。32-noise采用同一DiPred实际32条采样的标准差，以共同四步零噪声P为中心，独立重校准；不是32路径均值中心，也没有从标准差伪造经验分位数。三组均用相同固定memory_dynamic在线校准和10条独立校准身份。', '',
        '## 测试窗口2–39，Re/u/v等权', '',
        '|方法|神经风险头参数|Score↓|Width↓|90%逐点覆盖%|AURC↓|最差区域覆盖%|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in values:
        params = {'selected16': '34004', 'original28': '58770', 'noise32': '无额外神经风险头'}[r['method']]
        lines.append(f"|{labels[r['method']]}|{params}|{r['standardized_interval_score']:.9f}|{r['standardized_width']:.9f}|{100*r['coverage']:.4f}|{r['relative_aurc']:.9f}|{100*r['worst_patch_coverage']:.4f}|")
    delta_full = 100 * (selected['standardized_interval_score'] / full['standardized_interval_score'] - 1)
    delta_noise = 100 * (selected['standardized_interval_score'] / noise['standardized_interval_score'] - 1)
    lines += ['', f'16通道相对原28的Score变化为{delta_full:+.5f}%；相对32-noise为{delta_noise:+.5f}%（正值更差）。风险头参数减少{100*(1-34004/58770):.4f}%，不是整个DiPred系统参数减少同样比例，也不直接代表端到端提速。', '',
        '|16通道参考|96项改善|持平|退步|', '|---|---:|---:|---:|']
    for r in counts:
        lines.append(f"|{labels[r['reference']]}|{r['wins']}|{r['ties']}|{r['regressions']}|")
    lines += ['', '三组共5616条开发/测试窗口/通道记录，完整8指标×12工况/通道、1e-6容差、两组96项对照及全部反例保留。旧广泛参考与辅助训练仅作为既有归档，不在本次正式对照中。原28因果事件另存original28_events.jsonl；16与32-noise事件存events.jsonl，各模型都在真值揭示前发出区间，随后独立更新。', '',
        '16和28均只需4次DiPred调用；若新鲜计算32路径标准差并另外保留共同零噪声中心，需128＋4次调用。本次复用真实采样统计缓存，没有把缓存读取时间当部署速度。', '',
        'Score/Width为训练通道标准化无量纲量，点MSE相同。覆盖是逐点边际量，不是完整20帧联合覆盖保证。在线校准器仍有固定的重叠物理特征，16通道仅指神经头输入。历史数据已反复使用，且子集是在前一轮消融后指定，全部为探索性，不宣称新鲜独立确认、统计等价或严格条件覆盖。', '',
        '网络：selected16_quantile_head.py；推理：infer_selected16_quantile.py，Selected16Predictor(geometry, arm="compact16")。权重与初始校准：runs/selected16_quantile_20261110/；本次三组结果：runs/requested16_28_noise32_20261110/。']
    (ROOT / '研究结果_16对照原28与32noise_20261110.md').write_text('\n'.join(lines) + '\n')
    dump(OUT / 'complete.json', dict(time=time.time(), scope_exactly_requested_three_methods=True,
        references_only_original28_and_noise32=True, new_neural_training=False,
        original28_never_refitted=True, score_change_vs_full_percent=delta_full, score_change_vs_noise_percent=delta_noise))
    print(json.dumps(dict(test_macro=values, original96counts=counts), indent=2), flush=True)
    print('REQUESTED16_28_NOISE32_REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
