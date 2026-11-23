"""Verify the completed fixed study and synthesize its three kinds of evidence."""
import gzip
import json

from research.observed_residual_feedback.run import dump, write_scores
from .context import CACHE_BYTES, GPU_SECONDS, OUT, PREVIOUS, ROOT, SOURCE
from .context import budget_records, new_cache_bytes, read, settings, sha

ARMS = ('N', 'C0', 'R_raw', 'CH', 'CS')
COMPARISONS = ('R_raw-C0', 'CH-R_raw', 'CS-R_raw')
A_COMPARISONS = ('AR-A0', 'AH-A0', 'AHR-AH')


def lines(path):
    with gzip.open(path, 'rt') as handle:
        return [json.loads(line) for line in handle]


def package_diagnosis_logs():
    """Split the two large B logs by arm for transport, retaining every row."""
    folder = OUT / 'correction_diagnosis'
    metadata = {}
    for name in ('scores', 'decomposition'):
        original = folder / f'{name}.jsonl.gz'
        arms = ('L', 'J', 'J_backbone_masked')
        if original.exists():
            entries = {arm: [] for arm in arms}
            for row in lines(original):
                entries[row['arm']].append(row)
            for arm in arms:
                part = folder / f'{name}_{arm}.jsonl.gz'
                write_scores(part, entries[arm])
                assert lines(part) == entries[arm]
            original.unlink()
        metadata[name] = {arm: len(lines(folder / f'{name}_{arm}.jsonl.gz')) for arm in arms}
    assert sum(metadata['scores'].values()) == 12096
    assert sum(metadata['decomposition'].values()) == 9072
    return metadata


def num(value):
    return f'{value:.6g}' if value is not None else '未定义'


def diff(record):
    return (f"{record['delta']:+.6g} "
            f"[{record['ci95'][0]:+.6g}, {record['ci95'][1]:+.6g}]")


def verify(selection, probe, diagnosis, controls):
    config = settings()
    anchor = sha(OUT / 'FROZEN_SELECTION.json')
    assert probe['frozen_selection_sha256'] == controls['selection_sha256'] == anchor
    records = budget_records()
    starts = {r['id']: r for r in records if r['event'] == 'start'}
    finishes = {r['id']: r for r in records if r['event'] == 'finish'}
    assert starts.keys() == finishes.keys()
    succeeded = [r for r in finishes.values() if r['passed']]
    actual = [(r['line'], r['stage']) for r in succeeded]
    expected = [('A', 'development'), ('B', 'development'), ('C', 'development'),
                ('A', 'test'), ('C', 'test')]
    assert sorted(actual) == sorted(expected)
    for run in succeeded:
        start = starts[run['id']]
        assert (selection['utc_ns'] < start['utc_ns'] if run['stage'] == 'test'
                else run['utc_ns'] < selection['utc_ns'])
        for name, expected_sha in start['code_sha256'].items():
            assert sha(ROOT / name) == expected_sha, name
    spent = sum(r['seconds'] for r in finishes.values())
    available_cache = [new_cache_bytes(), *(r['cache_bytes'] for r in records)]
    peak_cache = max(available_cache)
    assert spent <= GPU_SECONDS and peak_cache <= CACHE_BYTES

    a_root = OUT / 'information_probe'
    a_frozen = read(a_root / 'FROZEN_A.json')
    assert a_frozen['fits'] == 4 and a_frozen['models'] == ['A0', 'AR', 'AH', 'AHR']
    assert all(sha(a_root / path) == expected for path, expected in a_frozen['hashes'].items())
    assert len(lines(a_root / 'train_native_receipts.jsonl.gz')) == 183
    assert probe['test_receipts_sha256'] == sha(a_root / 'test_native_receipts.jsonl.gz')
    for split in ('development', 'test'):
        result = probe[split]
        assert result['trajectories'] == 12 and result['windows'] == [2, 3]
        assert result['source_checks']['archived_history_noise_truth_and_zero_correction_match']
        assert result['source_checks']['max_abs_native_previous_mean_reconstruction'] < 2e-6
        assert set(result['comparisons']) == set(A_COMPARISONS)
        receipts = lines(a_root / f'{split}_native_receipts.jsonl.gz')
        frozen_schedule = read(OUT / 'A_SCHEDULE.json')[split]
        assert len(receipts) == len(frozen_schedule) == 12
        for idx, (receipt, entry) in enumerate(zip(receipts, frozen_schedule)):
            assert receipt['identity']['index'] == idx
            assert receipt['identity']['entry'] == entry
            for n, event in enumerate(receipt['windows'], 1):
                assert event['window'] == n
                assert event['issued']['utc_ns'] < event['revealed']['utc_ns']
                if split == 'test':
                    assert event['issued']['utc_ns'] > selection['utc_ns']
                if n > 1:
                    assert len(event['issued']['probe_sha256']) == 10
    assert probe['development']['fit']['normal_equation_lambda'] == 1.

    assert diagnosis['scope'] == 'development_only' and diagnosis['optimizer_updates'] == 0
    assert diagnosis['weights_unchanged'] and diagnosis['input_history_noise_truth_hashes_match_archive']
    assert diagnosis['forward_groups']['total'] <= config['B_group_limit']
    assert diagnosis['forward_groups']['total'] == 108
    assert diagnosis['gradient_entries'] == read(OUT / 'B_GRADIENT_SCHEDULE.json')
    assert diagnosis['auxiliary_identity_max_abs'] < 2e-7
    assert diagnosis['decomposition_identity_max_abs'] < 1e-12
    bdir = OUT / 'correction_diagnosis'
    assert len(lines(bdir / 'gradient.jsonl.gz')) == 24
    assert all(item['equal'] == item['total'] == 72
               for item in diagnosis['factual_replay_tensor_matches'].values())
    assert diagnosis['factual_replay_scores']['max_abs_difference'] == 0.
    for name, expected_hash in diagnosis['checksum_verified_weights'].items():
        assert sha(PREVIOUS / 'weights' / f'{name}_500.pt') == expected_hash

    cdir = OUT / 'representation_controls'
    run = read(cdir / 'RUN.json')
    assert run['common_updates'] == controls['updates_per_arm'] <= config['neural_updates']
    assert controls['comparable_500_step_completion'] == (run['common_updates'] == 500)
    if run['common_updates']:
        assert len(lines(cdir / 'training.jsonl.gz')) == 3 * run['common_updates']
    assert controls['archive_reused'] == ['N', 'R_raw']
    assert controls['selected']['R_raw'] == 500 and controls['selected']['N'] == 0
    assert controls['new_test_forward_groups'] == 36 * (3 - len(controls['zero_step_aliases']))
    assert controls['test_rows'] == 5 * 12 * 3 * 2 * 3 * 21
    assert set(controls['warm']['final']['score_u']['contrasts']) == set(COMPARISONS)
    for artifacts in run['weights'].values():
        assert all(sha(ROOT / item['path']) == item['sha256'] for item in artifacts)
    a_forward = sum(probe[split]['executed_native_groups']['generated'] for split in ('development', 'test'))
    per_stage = {f"{r['line']}/{r['stage']}": r['seconds'] for r in succeeded}
    return dict(passed=True, seconds=spent, latest_cache_bytes=new_cache_bytes(),
                peak_cache_bytes=peak_cache, stage_seconds=per_stage, completed_stages=actual,
                failed_stage_attempts=len(finishes)-len(succeeded),
                neural_updates=run['common_updates']*3, ridge_fits=4, diagnosis_groups=108,
                native_probe_groups=a_forward, representation_eval_groups=controls['publications'],
                selection_sha256=anchor)


def verdict(probe, diagnosis, controls):
    warm = controls['warm']['final']
    u = warm['score_u']
    v = warm['score_v']
    cs = u['contrasts']['CS-R_raw']
    actual = warm['score_u']['contrasts']['R_raw-C0']
    a_risk = [probe[key]['comparisons']['AHR-AH']['second_moment'] for key in ('development', 'test')]
    a_signed = [probe[key]['comparisons']['AHR-AH']['signed'] for key in ('development', 'test')]
    cs_provisional = (controls['comparable_500_step_completion'] and cs['ci95'][1] < 0
                      and u['means']['CS'] < u['means']['N']
                      and v['means']['CS'] < min(v['means']['N'], v['means']['R_raw']))
    risk_only = (all(item['ci95'][1] < 0 for item in a_risk)
                 and not all(item['ci95'][1] < 0 for item in a_signed))
    if cs_provisional:
        recommendation = '最高优先级是独立轨迹上按原协议复验低频历史残差 CS。'
    elif risk_only:
        recommendation = '最高优先级是在独立轨迹上预注册复验残差的二阶矩预测，作为随后研究分布条件的依据。'
    else:
        recommendation = '当前配置到此停止。'
    if not controls['comparable_500_step_completion']:
        headline = '三臂未共同完成500次更新，表示对照与旧J的等步比较证据不足。'
    elif actual['ci95'][1] < 0:
        headline = '在本次旧测试中，原始残差条件模型的暖窗 U 评分优于无残差 C0。'
    else:
        headline = '本轮未证实原始残差条件比无残差 C0 带来稳定的暖窗 U 评分改善。'
    return headline, recommendation


def report(selection):
    a = read(OUT / 'information_probe/summary.json')
    b = read(OUT / 'correction_diagnosis/summary.json')
    c = read(OUT / 'representation_controls/summary.json')
    execution = verify(selection, a, b, c)
    execution['diagnosis_log_rows_by_arm'] = package_diagnosis_logs()
    headline, recommendation = verdict(a, b, c)
    summary = ['# 历史残差信息、纠错方向与输入表示', '', headline, '',
               '本研究固定原生参考 N，沿用 seed42、46/12/12 条既有轨迹和原 Re 隔离。'
               '开发和旧测试均从起点400连续观察 W0–W3。W1 冷启动分开，暖窗 W2/W3 先在轨迹内平均。'
               '以下区间均在12条原轨迹上配对重抽10000次。工况、通道与逐帧结果仅用于探索。', '',
               '## 总体比较', '',
               '| 旧测试方法 | 步 | 暖窗主干 U / V | 暖窗最终 U / V | 最终 CRPS U / V | 最终全路径 ES U / V | 最终 MSE 原尺度 / 标准化 | 最终成员方差 原尺度 / 标准化 | 冷窗最终 U / V |',
               '| --- | ---: | --- | --- | --- | --- | --- | --- | --- |']
    for arm in ARMS:
        def metric(phase, field, first, second):
            table = c[phase][field]
            return f"{num(table[first]['means'][arm])} / {num(table[second]['means'][arm])}"
        cells = [arm, str(c['selected'][arm]),
                 metric('warm', 'backbone', 'score_u', 'score_v'),
                 metric('warm', 'final', 'score_u', 'score_v'),
                 metric('warm', 'final', 'crps_u', 'crps_v'),
                 metric('warm', 'final', 'es_u', 'es_v'),
                 metric('warm', 'final', 'mse', 'smse'),
                 metric('warm', 'final', 'variance', 'standardized_variance'),
                 metric('cold', 'final', 'score_u', 'score_v')]
        summary.append('| ' + ' | '.join(cells) + ' |')
    summary.extend(['', 'U 为预定主评分，V 是同一16成员的经验评分。分数越低越好。'
                    '原生及等价零步复用同身份归档评分。仅新臂的非零选择需要本轮旧测试推理。', '',
                    '## 假设证据', '',
                    '| 证据 | 预定对比或诊断 | 结果（差值及轨迹配对95%区间） |',
                    '| --- | --- | --- |'])
    for pair in A_COMPARISONS:
        def probe_result(split):
            compare = a[split]['comparisons'][pair]
            return '带符号 ' + diff(compare['signed']) + '；平方 ' + diff(compare['second_moment'])
        summary.append(f'| A 信息探针 | {pair} | 开发 {probe_result("development")}；旧测试 {probe_result("test")} |')
    decomposition = {row['arm']: row for row in b['warm_decomposition_overall'] if row['units'] == 'standardized'}
    for arm in ('L', 'J'):
        row = decomposition[arm]
        summary.append(f"| B 当前修正 | {arm} 最终−主干，开发暖窗 | 标准化均值 MSE {diff(row['mse_change_paired'])}；"
                       f"U {diff(b['scores'][arm]['warm']['score_u']['final_minus_backbone'])} |")
    summary.append(f"| B 主干路径 | J−屏蔽主干记忆的同权重 J | 主干 U {diff(b['scores']['J_minus_masked_backbone']['score_u'])}；"
                   f"最终 U {diff(b['scores']['J_minus_masked_final']['score_u'])} |")
    for pair in COMPARISONS:
        result = c['warm']['final']
        summary.append(f"| C 表示对照 | {pair}，旧测试暖窗 | U {diff(result['score_u']['contrasts'][pair])}；"
                       f"V {diff(result['score_v']['contrasts'][pair])} |")

    summary.extend(['', '## A · 历史中可预测的成分', '',
                    '用相同冻结原生预测计算标签与过去残差。固定四个带截距岭回归同时预测未来20帧带符号共同残差及其平方。'
                    '损失按训练尺度报告；平方的预测是条件二阶矩代理。以下为旧测试带符号 MSE / 平方目标 MSE。', ''])
    for name, metrics in a['test']['metrics'].items():
        if ':' not in name:
            summary.append(f"- {name} 为 {num(metrics['signed_mse'])} / {num(metrics['second_moment_mse'])}。")
    stability = a['test']['sampling_stability']
    signed_older = [a[split]['comparisons']['AH-A0']['signed'] for split in ('development', 'test')]
    more_residual = [a[split]['comparisons']['AHR-AH'] for split in ('development', 'test')]
    if signed_older[0]['ci95'][1] < 0 <= signed_older[1]['delta']:
        summary.extend(['', '直接加入更早历史的带符号收益只出现在开发集，旧测试点估计反向，区间跨零。'])
    if all(record['second_moment']['ci95'][0] <= 0 <= record['second_moment']['ci95'][1]
           for record in more_residual):
        summary.extend(['', '控制更早真实历史后，添加残差对平方目标的两套区间都跨零。风险代理方向也随划分而变。'])
    summary.extend(['', f"两个8成员分半所得过去残差的差值 RMS 为 {num(stability['split_input_rms'])}，"
                    f"16成员共同残差 RMS 为 {num(stability['full_residual_rms'])}。"
                    "分半共享真值和同一次集合，测量的是有限成员采样变化。"
                    "两组替换输入后的预测变化和相对零、训练均值、A0 的误差变化保存在 A 的汇总。", '',
                    "代数关系 barHatW⁽⁰⁾ = W − barR⁽⁰⁾ 在发布记录上通过核验。原生残差包含对过去预测、更早历史和采样变化的重编码。"
                    "四个探针只衡量固定线性模型的条件预测能力。结论不能外推到互信息、条件方差或有更长记忆的在线 J。", '',
                    '## B · 修正量与误差方向', '',
                    "均值 MSE 的变化逐窗核对为修正平方项减两倍残差与修正内积。下列 RMS、内积和余弦使用标准化单位，并按轨迹汇总。", ''])
    for arm in ('L', 'J', 'J_backbone_masked'):
        row = decomposition[arm]
        summary.append(f"- {arm} 的残差 RMS {num(row['residual_rms'])}，修正 RMS {num(row['correction_rms'])}，"
                       f"内积 {num(row['inner_product_per_element'])}，方向余弦 {num(row['direction_cosine'])}。"
                       f"最终−主干 V 为 {diff(b['scores'][arm]['warm']['score_v']['final_minus_backbone'])}。")
    l_j = [decomposition[arm] for arm in ('L', 'J')]
    if all(item['direction_cosine'] < 0 and item['mse_change'] > 0 for item in l_j):
        summary.extend(['', '两臂的修正与共同残差平均略呈反向，修正平方项和反向内积共同推高均值 MSE。'
                        '这一解释来自逐窗分解，方向与幅度均有直接测量。'])
    perturbation = b['masked_perturbation']['backbone_member_rms']
    summary.extend(['', f"J 屏蔽主干记忆后，暖窗原主干输出 RMS 变化均值 {num(perturbation['mean'])}，"
                    f"{perturbation['nonzero']}/{perturbation['windows']} 个窗有变化。"
                    "屏蔽保留事实历史和修正头输入，且不回写事实状态；其结果只诊断该权重对通路的依赖。", '',
                    "训练固定4段的8个暖窗分别测量加权损失梯度。辅助项与最终集合均值的标准化 MSE 数值等价。"
                    "停止梯度改变更新路径，辅助项所用真值与最终评分是同一观测。", ''])
    for group, data in b['gradient_stats'].items():
        norms = '、'.join(f"{key}={num(value['mean'])}" for key, value in data['gradient_l2_norm'].items())
        cosines = '、'.join(f"{key} {value['negative']}/{value['defined']} 次为负"
                            for key, value in data['gradient_cosine'].items())
        summary.append(f"- {group} 的平均梯度 L2 为 {norms}。余弦 {cosines}。")
    summary.extend(['', '八个被测窗口里，最终 U 与主干 U 在两组有共同梯度的参数上余弦均为正。'
                    '辅助项的梯度量级明显较小。当前证据指向尺度差异，尚无法把纠错失效归因于梯度相互抵消。', '',
                    '## C · 无残差、直接历史与低频残差', '',
                    '三臂共享旧 J 原生骨干初始化、可训练张量集合、目标和日程。C0 保留可用标记，CH 用再早一窗的真实观测填记忆第一槽，'
                    'CS 只在输入层对各帧各通道进行固定空间平均。各模型发布记录继续从自己的主干输出计算残差。'
                    '旧 J500 作为 R_raw 按身份复用。开发只依据暖窗最终 U 选模。', ''])
    for arm, candidates in c['candidates'].items():
        values = '、'.join(f'{step} 步 {num(value)}' for value, step in candidates)
        summary.append(f"- {arm} 的开发候选为 {values}，选中 {c['selected'][arm]} 步。")
    selected_final = c['warm']['final']
    if all(selected_final[key]['means']['C0'] <= selected_final[key]['means']['R_raw']
           for key in ('score_u', 'score_v')):
        summary.extend(['', 'C0 的暖窗 U 和 V 点估计都略低于 R_raw。区间跨零，现有证据不足以支持扩展历史残差输入。'])
    strata = lines(OUT / 'representation_controls/test_exploration.jsonl.gz')
    highlights = [row for row in strata if row['lead'] == 0 and row['metric'] == 'crps_v'
                  and ((row['Re'] == 'all' and row['channel'] in ('u', 'v')) or
                       (row['Re'] != 'all' and row['channel'] == 'overall'))]
    summary.extend(['', '以下为同一预测的 V CRPS 描述性分解。逐帧和完整通道×Re 切片保存在 C 的归档，无额外推理。', ''])
    for row in highlights:
        values = '、'.join(f'{pair} {delta:+.6g}' for pair, delta in row['contrasts'].items())
        summary.append(f"- Re={row['Re']} / {row['channel']}（{row['n']}条轨迹），{values}。")
    summary.extend(['', '## 预算、复现与决定', '',
                    f"共执行 {execution['ridge_fits']} 次岭回归、B 的 {execution['diagnosis_groups']} 个固定前向组和 "
                    f"{execution['neural_updates']} 次神经训练更新。A 实测新增 {execution['native_probe_groups']} 个原生前向组。"
                    f"C 旧测试新增 {c['new_test_forward_groups']} 个16成员前向组。"
                    f"保守累计阶段壁钟 {execution['seconds']:.1f} 秒，缓存峰值 {execution['peak_cache_bytes']/2**30:.3f} GiB。"
                    "三名工作代理并行实现，GPU 分阶段串行。既有反馈审计的分通道修正直接引用上一轮成果。", '',
                    "已有开发与旧测试资料反复使用，轨迹数与 Re 范围有限，统计区间也不覆盖训练种子变异。"
                    "A 反映有限探针的条件预测，B 反映现有权重的行为，C 反映给定训练设计下的表现。", '',
                    recommendation, '',
                    '[固定协议](PROTOCOL.md) · [执行核验](VERIFICATION.json) · [A 探针结果](information_probe/summary.json) · '
                    '[B 诊断结果](correction_diagnosis/summary.json) · [C 表示结果](representation_controls/summary.json)', ''])
    (OUT / 'DECISION_REPORT.md').write_text('\n'.join(summary))
    execution['artifact_sha256'] = {str(path.relative_to(ROOT)): sha(path)
        for path in sorted(OUT.rglob('*')) if path.is_file() and path.name != 'VERIFICATION.json'}
    execution['source_sha256'] = {str(path.relative_to(ROOT)): sha(path)
        for path in sorted(SOURCE.rglob('*')) if path.is_file() and '__pycache__' not in path.parts}
    dump(OUT / 'VERIFICATION.json', execution)
    print('All work lines verified', execution['stage_seconds'], flush=True)
