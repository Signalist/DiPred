"""Verify the fixed execution and summarize trajectory-level contrasts."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from research.observed_residual_feedback.run import (
    ARMS, MEMBERS, OUT, ROOT, dump, read, sha,
)


def json_lines(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return [json.loads(line) for line in handle]


def load_log(name):
    path = OUT / f'{name}.jsonl'
    return json_lines(path if path.exists() else path.with_suffix('.jsonl.gz'))


def trajectory_values(rows, arms, windows, field, metric, channel='overall', lead=0, group=None):
    values = {}
    for arm in arms:
        for row in rows:
            if (row['arm'] == arm and row['window'] in windows and row['field'] == field
                    and row['channel'] == channel and row['lead'] == lead
                    and (group is None or row['Re'] == group)):
                values.setdefault(row['trajectory'], {}).setdefault(arm, []).append(row[metric])
    assert values and all(set(items) == set(arms) for items in values.values())
    assert all(len(v) == len(windows) for items in values.values() for v in items.values())
    return {key: {arm: float(np.mean(items[arm])) for arm in arms} for key, items in values.items()}


def avg(values, arm):
    return float(np.mean([row[arm] for row in values.values()]))


def contrast(values, first, second):
    delta = np.array([values[key][first] - values[key][second] for key in sorted(values)])
    rng = np.random.default_rng(123)
    samples = rng.integers(len(delta), size=(10000, len(delta)))
    ci = np.quantile(delta[samples].mean(axis=1), [.025, .975])
    return dict(delta=float(delta.mean()), ci95=ci.tolist(), lower_count=int((delta < 0).sum()),
                n=len(delta), by_trajectory={key: float(d) for key, d in zip(sorted(values), delta)})


def verify(schedule, selection, run, stages):
    from research.observed_residual_feedback.run import REFERENCE
    frozen = read(OUT / 'FROZEN.json')
    assert frozen['reference'] == frozen['actual_head'] == REFERENCE
    assert sha(OUT / 'FROZEN.json') == selection['config_sha256']
    assert all(sha(ROOT / name) == expected for name, expected in frozen['sha256'].items())
    preflight = read(OUT / 'PREFLIGHT.json')
    assert preflight['passed'] and list(preflight['native_zero_init_max_abs'].values()) == [0., 0.]
    assert all(preflight[key] for key in ('L_masks_backbone_slots', 'J_backbone_uses_residual',
        'both_heads_use_residual', 'head_backprop_to_current_backbone', 'frozen_tensors_unchanged',
        'publication_order_and_future_label_pollution', 'scale_and_U_V_checks'))
    assert preflight['config_sha256'] == selection['config_sha256']
    assert run['frozen_backbone_unchanged'] and run['native_calls_per_forecast'] == 4
    assert run['budget_met'] and 0 <= run['common_updates'] <= 500
    checkpoints = sorted(stages['development'])
    assert checkpoints == [step for step in (0, 250, 500) if step <= run['common_updates']]
    logs = load_log('training') if run['common_updates'] else []
    formal_updates = len(logs)
    assert formal_updates == run['common_updates'] * len(ARMS)
    for index in range(run['common_updates']):
        for arm, row in zip(ARMS, logs[index*2:index*2+2]):
            assert (row['arm'], row['step']) == (arm, index+1)
            assert all(row[key] == value for key, value in schedule['train'][index].items())
            assert len(row['warm_losses']) == 2 and np.isfinite(row['warm_losses']).all()
    for arm in ARMS:
        for step, artifact in zip(checkpoints, run['weights'][arm]):
            assert artifact['path'].endswith(f'/{arm}_{step}.pt')
            assert sha(ROOT / artifact['path']) == artifact['sha256']
        assert len(run['weights'][arm]) == len(checkpoints)
        candidates = []
        for step in checkpoints:
            table = stages['development'][step]
            values = trajectory_values(table, (arm,), (2, 3), 'final', 'score_u')
            assert len(values) == 12
            candidates.append((avg(values, arm), step))
        assert all(s == step and abs(value - v) < 1e-14
                   for (value, step), (v, s) in zip(candidates, selection['candidates'][arm]))
        assert min(candidates)[1] == selection['selected'][arm]

    logs = load_log('publication')
    cursor = 0
    n_committed = 0
    score_rows_count = 0
    shared_inputs = {}
    for split, checkpoints in stages.items():
        entries = schedule[split]
        for stage, rows in sorted(checkpoints.items()):
            if split == 'development':
                methods = ('N',) if stage == 0 else ARMS
                scored = ('N', *ARMS) if stage == 0 else ARMS
            else:
                methods = ('N', *[arm for arm in ARMS if selection['selected'][arm]])
                scored = ('N', *ARMS)
            identities = {(r['trajectory'], r['arm'], r['window'], r['field'], r['channel'], r['lead'])
                          for r in rows}
            expected = {(e['trajectory'], arm, window, field, channel, lead)
                        for e in entries for arm in scored for window in (1, 2, 3)
                        for field in ('backbone', 'final') for channel in ('overall', 'u', 'v')
                        for lead in range(21)}
            assert len(rows) == len(identities) and identities == expected
            score_rows_count += len(rows)
            index = {(r['trajectory'], r['arm'], r['window'], r['field'], r['channel'], r['lead']): r
                     for r in rows}
            identities_by_trajectory = {e['trajectory']: e for e in entries}
            for row in rows:
                entry = identities_by_trajectory[row['trajectory']]
                assert (row['split'], row['stage'], row['start'], row['Re'], row['seed']) == (
                    split, stage, entry['start'], entry['Re'], entry['noise_seeds'][row['window']-1])
                if row['lead'] == 0:
                    for estimator in ('u', 'v'):
                        assert (row['score_' + estimator] == row['crps_' + estimator] +
                                .1 * row['es_' + estimator])
                if row['arm'] == 'N' and row['field'] == 'final':
                    base = index[(row['trajectory'], 'N', row['window'], 'backbone',
                                  row['channel'], row['lead'])]
                    assert dict(row, field='backbone') == base
                if row['arm'] in ARMS and (split == 'development' and stage == 0
                        or split == 'test' and selection['selected'][row['arm']] == 0):
                    base = index[(row['trajectory'], 'N', row['window'], row['field'],
                                  row['channel'], row['lead'])]
                    assert dict(row, arm='N') == base
            for entry in entries:
                previous = {}
                for n, seed in enumerate(entry['noise_seeds'], 1):
                    count = len(methods)
                    issued = logs[cursor:cursor+count]
                    revealed = logs[cursor+count:cursor+2*count]
                    cursor += 2*count
                    assert len(revealed) == count
                    assert len({r['noise_sha256'] for r in issued}) == 1
                    assert len({r['history_sha256'] for r in issued}) == 1
                    assert len({r['truth_sha256'] for r in revealed}) == 1
                    key = (split, entry['trajectory'], n)
                    input_hashes = (issued[0]['noise_sha256'], issued[0]['history_sha256'],
                                    revealed[0]['truth_sha256'])
                    assert shared_inputs.setdefault(key, input_hashes) == input_hashes
                    assert max(r['utc_ns'] for r in issued) < min(r['utc_ns'] for r in revealed)
                    for arm, issue, reveal in zip(methods, issued, revealed):
                        for event, row in [('issued', issue), ('revealed', reveal)]:
                            assert (row['event'], row['arm'], row['split'], row['stage']) == (event, arm, split, stage)
                            assert (row['trajectory'], row['window'], row['seed']) == (entry['trajectory'], n, seed)
                        assert issue['availability'] == (n > 1)
                        if n > 1:
                            old = previous[arm]
                            assert issue['history_sha256'] == old['truth_sha256']
                            assert issue['previous_residual_sha256'] == old['residual_sha256']
                            assert issue['previous_correction_sha256'] == old['mean_correction_sha256']
                        if split == 'test':
                            assert selection['utc_ns'] < issue['utc_ns']
                        previous[arm] = reveal
                        n_committed += 1
    assert cursor == len(logs)
    return dict(passed=True, publications=n_committed, formal_updates=formal_updates,
                metric_rows=score_rows_count, weights_verified=True, own_revealed_state_chain_verified=True,
                test_selection_precedes_issue=True, code_sha256=sha(Path(__file__)))


def old_repair_summary():
    path = OUT / 'old_audit_repaired.csv'
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    results = {}
    for step in ('0', '500', '1000'):
        results[step] = {}
        for channel in ('overall', 'u', 'v'):
            results[step][channel] = {}
            for metric in ('score_u', 'score_v', 'mse', 'variance', 'width90', 'coverage90'):
                by_trajectory = {}
                for row in rows:
                    if row['checkpoint'] == step and row['channel'] == channel:
                        by_trajectory.setdefault(row['trajectory'], []).append(float(row[metric]))
                assert len(by_trajectory) == 12 and all(len(v) == 2 for v in by_trajectory.values())
                values = {key: dict(on=float(np.mean(v)), off=0.) for key, v in by_trajectory.items()}
                results[step][channel][metric] = contrast(values, 'on', 'off')
    dump(OUT / 'old_audit_repaired_paired.json', results)
    return results


def report():
    run = read(OUT / 'RUN.json')
    selection = read(OUT / 'SELECTION.json')
    schedule = read(OUT / 'schedule.json')
    stages = dict(development={}, test={'selected': json_lines(OUT / 'test_selected.jsonl.gz')})
    for step in (0, 250, 500):
        path = OUT / f'development_{step}.jsonl.gz'
        if path.exists():
            stages['development'][step] = json_lines(path)
    verification = verify(schedule, selection, run, stages)
    old = old_repair_summary()
    results = {}
    rows = stages['test']['selected']
    metrics = ('score_u', 'score_v', 'crps_u', 'crps_v', 'es_u', 'es_v', 'mse', 'smse',
               'variance', 'standardized_variance', 'width90', 'residual_mae',
               'residual_signed_mean', 'residual_physical_mae', 'residual_physical_signed_mean')
    for phase, windows in [('warm', (2, 3)), ('cold', (1,))]:
        results[phase] = {}
        for field in ('backbone', 'final'):
            results[phase][field] = {}
            for metric in metrics:
                values = trajectory_values(rows, ('N', 'L', 'J'), windows, field, metric)
                results[phase][field][metric] = dict(
                    means={arm: avg(values, arm) for arm in ('N', 'L', 'J')},
                    J_minus_L=contrast(values, 'J', 'L'), J_minus_N=contrast(values, 'J', 'N'))
    strata = {}
    groups = sorted({r['Re'] for r in rows})
    for group in [None, *groups]:
        strata[str(group or 'all')] = {}
        for channel in ('overall', 'u', 'v'):
            block = {}
            for lead in range(21):
                selected_metrics = ('crps_u', 'crps_v', 'es_u', 'es_v', 'mse', 'variance') if not lead else (
                    'crps_u', 'crps_v', 'mse', 'variance')
                block[str(lead)] = {metric: contrast(trajectory_values(
                    rows, ('J', 'L'), (2, 3), 'final', metric, channel, lead, group), 'J', 'L')
                    for metric in selected_metrics}
            strata[str(group or 'all')][channel] = block
    dump(OUT / 'test_contrasts.json', results)
    dump(OUT / 'test_strata.json', strata)

    warm = results['warm']['final']
    delta_u = warm['score_u']['J_minus_L']
    delta_v = warm['score_v']['J_minus_L']
    delta_n = warm['score_u']['J_minus_N']
    if not any(selection['selected'].values()):
        headline = '两臂都回到零步，当前验证没有支持残差进入主干的增量。'
    elif delta_u['delta'] >= 0:
        headline = '按预定暖启动 U 综合分数，将已观测残差送入主干未优于只在末端修正使用残差。'
    elif delta_u['ci95'][1] >= 0:
        headline = 'J 的暖启动 U 综合分数点估计优于 L，配对区间跨零，增量证据不足。'
    else:
        headline = 'J 相对 L 在暖启动 U 综合分数上出现探索性增量。'
    if delta_u['delta'] < 0 <= delta_n['delta']:
        headline += ' J 未优于原生 N，证据限于这一训练对照。'
    if delta_u['delta'] < 0 <= delta_v['delta']:
        headline += ' 固定16成员的 V 点估计未改善，部署口径无收益证据。'
    lines = ['# 已观测残差进入下一窗口主干的受控实验', '', headline, '',
             f"seed42，实际每臂 {run['common_updates']} 次更新。冻结的 L/J 检查点分别为 "
             f"{selection['selected']['L']} 和 {selection['selected']['J']}。"
             f"原开发/旧测试各12条轨迹，起点400，20帧初始历史加三个20帧预测窗。"
             "表中暖窗结果先按轨迹平均 W2/W3，再对轨迹等权。各分数越低越好。", '',
             '| 旧测试 | 输出 | 方法 | U 综合 | V 综合 | CRPS U / V | 全路径 ES U / V | MSE 原尺度 / 标准化 | 成员方差 原尺度 / 标准化 | 平均绝对 R 或 E（标准化） |',
             '| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: |']
    for phase in ('warm', 'cold'):
        for field in ('backbone', 'final'):
            table = results[phase][field]
            for arm in ('N', 'L', 'J'):
                def v(metric):
                    return f"{table[metric]['means'][arm]:.7g}"
                lines.append(f"| {phase} | {field} | {arm} | {v('score_u')} | {v('score_v')} | "
                             f"{v('crps_u')} / {v('crps_v')} | {v('es_u')} / {v('es_v')} | "
                             f"{v('mse')} / {v('smse')} | {v('variance')} / {v('standardized_variance')} | "
                             f"{v('residual_mae')} |")
    lines.extend(['', '暖启动最终预测的轨迹级配对差，按12条原始轨迹重抽10000次（seed123）。', ''])
    for name, d in [('J−L, U', delta_u), ('J−L, V', delta_v), ('J−N, U', delta_n),
                    ('J−N, V', warm['score_v']['J_minus_N'])]:
        lo, hi = d['ci95']
        lines.append(f"- {name} 为 {d['delta']:+.7g}，95%区间 [{lo:+.7g}, {hi:+.7g}]，{d['lower_count']}/{d['n']} 条轨迹差值为负。")
    backbone_u = results['warm']['backbone']['score_u']
    final_u = warm['score_u']['means']
    base_u = backbone_u['means']
    lines.extend(['', f"暖窗 J−L 的主干 U 差为 {backbone_u['J_minus_L']['delta']:+.7g}，"
                  f"加入修正后为 {delta_u['delta']:+.7g}。"
                  f"L、J 的修正使各自 U 综合分数变化 {final_u['L']-base_u['L']:+.7g}、"
                  f"{final_u['J']-base_u['J']:+.7g}。"
                  "两臂最终暖窗的原尺度 MSE 与经验 V 综合分数点估计均高于原生。", '',
                  '开发选择只使用 W2/W3 最终 U 综合分数，已在测试首次发布前冻结。', ''])
    for arm in ARMS:
        values = '、'.join(f'{step}步 {value:.8g}' for value, step in selection['candidates'][arm])
        lines.append(f'- {arm} 的候选开发分数为 {values}。')
    params = read(OUT / 'TRAIN_STARTED.json')['parameters']
    assert params['L'] == params['J']
    lines.extend(['', '误差方向与集合离散度', '',
        f"两臂各有 {params['L']['total']:,} 个参数，其中 {params['L']['trainable']:,} 个参与训练。"
        '骨干仅 up2/out 开放更新，编码器、投影和修正头通过两个暖窗的目标联合更新。', '',
        'R 是真值减该方法当次发布的未修正主干输出，E 是 R 减该成员修正。表中误差是集合均值预测的绝对误差，有符号均值和原尺度误差见 test_contrasts.json。'
        '向下一窗传递的是上一窗 R 和修正的成员均值。过去与未来同编号帧处于不同物理时刻，成员索引也不跨窗匹配。', '',
        'U 按非自配对分母 M(M−1)，V 按 M²。同一16成员的 U/V 差反映集合内部距离项权重，不能用 U 改善代替 V 改善。'
        '检查主干和最终输出可区分修正前后的净变化。L/J 的条件编码器、投影、残差头、总参数量相同，共享编码器在 L 中以屏蔽槽位的输入另外执行一次主干分支。'
        '反馈既改变当前主干也可能经训练改变下一窗状态，仅凭组间差值无法归因到单一动力学机制。', '',
        '同一测试预测的分解摘要。负值表示 J 的 CRPS 低于 L。', ''])
    for group, channels in strata.items():
        for channel, item in channels.items():
            d = item['0']['crps_v']
            leads = [i for i in range(1, 21) if item[str(i)]['crps_v']['delta'] < 0]
            lines.append(f"- Re {group} / {channel} 的 V CRPS 差为 {d['delta']:+.6g}，"
                         f"{d['lower_count']}/{d['n']} 条轨迹改善，改善未来帧 {', '.join(map(str, leads)) or '无'}。")
    old_u = old['1000']['overall']['score_u']
    old_v = old['1000']['overall']['score_v']
    lines.extend(['', '旧审计修复', '',
        '只使用旧审计已保存的同前向 V 行及其成对 U 字段。单通道 MSE/方差乘对应标准差平方与两通道方差尺度均值的比，宽度乘对应标准差与两通道标准差均值的比。'
        '这恢复旧实现错误广播前的单通道口径。分数基于标准化预测，不受该尺度错误影响。旧报告原样保留。'
        f"旧 C 的1000步同前向 C_on−C_off 综合差为 U {old_u['delta']:+.7g}、V {old_v['delta']:+.7g}。"
        'F/O 的同前向 V、宽度及覆盖率未保存，无法按该口径补算。', '',
        '执行与边界', '',
        f"预检通过。两臂初始按原生输出逐值复现，J 主干对过去残差干预有响应，L 主干无响应，两臂末端头均有响应。冻结参数及缓冲区哈希不变。"
        f"预检、训练与评价保守计时 {run['seconds']:.2f} 秒，训练显存峰值 {run['peak_cuda_allocated_bytes']/2**30:.2f} GiB。"
        "每成员每窗口四次原生骨干调用。除训练反向外，新增编码、投影、末端修正及已观测状态计算。"
        f"发布与揭示日志核验 {verification['publications']} 条方法窗口链，当前真值读取在所有该窗预测写入发布日志之后。", '',
        '旧开发和旧测试为反复使用的数据，单种子、12条轨迹与有限 Re 只能提供探索证据。'
        + ('若追查该信号，建议独立复验。' if delta_u['ci95'][1] < 0 and any(selection['selected'].values())
           else '本配置到此结束。'), ''])
    (OUT / 'DECISION_REPORT.md').write_text('\n'.join(lines))
    verification['artifact_sha256'] = {str(path.relative_to(ROOT)): sha(path)
        for path in sorted(OUT.rglob('*')) if path.is_file() and path.name != 'VERIFICATION.json'}
    dump(OUT / 'VERIFICATION.json', verification)
    print('Verified', verification['publications'], 'publications and', verification['metric_rows'], 'metric rows')
