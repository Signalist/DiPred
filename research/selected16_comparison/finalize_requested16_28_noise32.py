"""Record only the three methods requested by the latest user correction."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/requested16_28_noise32_20261110'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    for name in ('independent_audit.json', 'analysis_verification.json', 'direct_inference_audit.json'):
        assert read(OUT / name)['passed']
    actual = read(OUT / 'direct_inference_audit.json')
    assert actual['actual_forecasts'] == 72 and actual['original32sample_std_regenerated_exact']
    values = {row['method']: row for row in read(OUT / 'test_macro.json')}
    selected, original, noise = (values[name] for name in ('selected16', 'original28', 'noise32'))
    change_full = 100 * (selected['standardized_interval_score'] / original['standardized_interval_score'] - 1)
    gain_noise = 100 * (1 - selected['standardized_interval_score'] / noise['standardized_interval_score'])
    reduction = 100 * (1 - 34004 / 58770)
    record = dict(time=time.time(), completed=True, user_scope='Only16,original28and32-noise;24wasusercorrectedto28.',
        formal_methods=['selected16', 'original28', 'noise32'], online_strategy='memory_dynamic',
        neural_head_selected_features=['history_variability2', 'forecast_gradient2', 'forecast_time_change2', 'denoising8', 'coordinates2'],
        selected_core_channels=16, full_core_channels=28, actual_selected_head_parameters=34004,
        full_head_parameters=58770, risk_head_parameter_reduction_percent=reduction,
        actual_unused_branches_removed=['history_projection', 'memory_projection', 'modifier'],
        training=dict(fit_pairs=819, steps=2000, seed=42, original28_audited_matched_run_reused=True,
                      point_predictor_frozen=True, calibration_and_heldout_ids_not_fit_neural_weights=True),
        primary_comparison_count=2, cells_per_comparison=96, tolerance=1e-6,
        vs_original28=dict(score_increase_percent=change_full, wins=23, ties=0, regressions=73),
        vs_noise32=dict(score_reduction_percent=gain_noise, wins=70, ties=0, regressions=26),
        test_window_rows=5616, all_counterexamples=99, point_mse_unchanged=True,
        actual_forecasts_exact=72, actual_backbone_calls=dict(selected16=4, original28=4, noise32_with_common_point_center=132),
        original32sampling_cache_reproduced_exact=True,
        previous_extra_control_note='Auxiliary masked16 training and a four-method replay finished before/during user narrowing; archived only, excluded from final comparisons. No new control training after narrowing.',
        latest_recommendation='Selected16 is a real compact option, not a lossless replacement of full28. Full28 has bettermeanScore;selected16 improvesmeanScore over32sampling butstill26originalcell regressions.',
        no_end_to_end_speedup_measured=True, no_auto_github_publication=True, whole_goal_achieved=False,
        report='研究结果_16对照原28与32noise_20261110.md',
        caveat='Only neuralhead restricted to16inputs; sharedcalibrator retains originalfeatures.32-noise uses common zero-noise center rather thanensemblemean. Reused historicaldata; no independent statistical or formalcoverage guarantee.')
    write(OUT / 'finalized_study.json', record)
    report = ROOT / record['report']
    marker = '<!-- requested_three_finalized -->'
    if marker not in report.read_text():
        report.write_text(report.read_text() + f'''\n{marker}\n\n## 最终核验与解释\n\n三组共5616条记录、两组原96项比较、全部99条退步和逐轨迹评分由标准库独立重算通过。实际接口72次完整20帧预测的点、上下界、上一误差及全部在线状态逐位一致；32-noise重新生成原32路径后std也与原缓存逐位一致。实际每批主干调用数为16/28各4，32-noise与共同零噪声中心合计132。完整反馈、重复发出、部分/非有限反馈、跨窗口连续性和reset守卫通过。\n\n16通道Score比原28高{change_full:.5f}%，比32-noise低{gain_noise:.5f}%。后者在当前校准下区间更窄，但覆盖率与最差帧/区域覆盖略差，因此不是相同覆盖效率的严格证明，也未全面优于任一参考。参数减少{reduction:.4f}%仅指风险头。按单组消融贡献排序不能保证联合保留后无损；当前16版作为精简版本保留，完整28仍是这三组中平均评分最佳者。\n\n本地推理入口：`infer_requested16_28_noise32.py`，`RequestedThreePredictor(geometry, method='selected16')`；另外两组仅用method='original28'或'noise32'。没有训练新的点预测器、其他基线、折分或多训练种子；新研究未自动推送GitHub。\n''')
    entry = ROOT / '研究主线_当前入口.md'
    marker = '<!-- requested16_28_noise32_completed_20261110 -->'
    if marker not in entry.read_text():
        prefix = f'''\n{marker}\n最新用户明确收敛为**仅16通道、原28通道、32-noise**（用户将24更正为28）；不再增加其他对照，旧广泛参考归档保留但不作为本轮正式比较。见[三组实验报告]({record['report']})。\n\n16真实裁剪保留梯度2＋坐标2＋历史波动2＋去噪8＋预测时间变化2；移除历史/误差投影与位移模块。34004风险参数，较原58770少{reduction:.2f}%。同原819对/2000步/seed42；原28复用上一轮完全匹配的已审计权重和校准，没有重训。点预测器四步99→66→33→0、完整20→20冻结。\n\n测试Score：16为{selected['standardized_interval_score']:.9f}，原28为{original['standardized_interval_score']:.9f}，32-noise为{noise['standardized_interval_score']:.9f}；16比28差{change_full:.4f}%，比32noise好{gain_noise:.4f}%。原96项分别23改善/73退步、70改善/26退步，不是全线胜出。覆盖分别89.9323%、89.9335%、89.9849%；16区间比32稍窄但局部覆盖略低。保留16为精简版本，不能称无损裁剪。\n\n5616完整窗口行、两组96项及全部99反例独立重算通过，72次实际完整窗口预测及全部反馈状态逐位复现；32原采样std重新计算也逐位一致。16/28各4次调用，32加共同零噪声中心132次；未声称端到端加速。共同校准器仍读取原物理特征，16仅是神经头输入数。数据反复使用，探索性而非独立或严格覆盖证明。\n\n入口[infer_requested16_28_noise32.py](infer_requested16_28_noise32.py)，method='selected16'/'original28'/'noise32'；本轮正式数据runs/requested16_28_noise32_20261110/。研究未自动推送，private分支仍2943cfa、main仍39d7085。下文旧“最新/所有参考/下一对照”等表述由用户本轮明确范围和本段取代。\n\n'''
        entry.write_text(entry.read_text().replace('# 当前研究入口\n', '# 当前研究入口\n' + prefix, 1))
    for name in ('reliability_direction_review_20261109.json', 'reliability_training_data_ledger_20261109.json'):
        path = ROOT / 'runs' / name
        ledger = read(path)
        ledger['selected16_requested_comparison'] = record
        ledger['latest_user_scope_override'] = 'Onlyselected16vsoriginal28and32-noise. Userexplicitlydoesnotwantadditionalcontrols; oldartifactsretained, notnewcomparisonrequirements.'
        if name.startswith('reliability_direction'):
            ledger['primary_evaluation'] = 'Original8metrics/96cells/tolerance1e-6; perlatestexplicituserinstruction only original28 and32-noise arecurrentcomparisonreferences.16has73and26regressions,respectively;globalgoalnotachieved.'
            ledger['next_method_family'] = 'No additional baseline or model-family expansion. If further optimization requested, improve selected16 using original28 and32-noise asonlyformalreferences.'
        write(path, ledger)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
