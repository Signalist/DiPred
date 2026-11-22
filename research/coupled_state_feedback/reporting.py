"""CPU-only reporting and completion audit, never another inference."""
from collections import Counter
import csv
import json

import numpy as np
import torch
from .run import OUT, ROOT, cfg, code_hashes, digest, dump, read, rows, macro, check_dependencies


def summary(rr):
    return {k:float(np.mean(list(macro(rr,k).values()))) for k in
        ('score','crps','es','smse','mse','variance','standardized_variance','time_mean_crps')}


def bootstrap(v):
    v=np.array(v,dtype=float)
    rng=np.random.default_rng(123)
    samples=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
    return float(v.mean()),*np.quantile(samples,[.025,.975]).tolist()


def report():
    c=cfg(); complete=read(OUT/'COMPLETE.json'); selection=read(OUT/'selection.json')
    train=read(OUT/'training_summary.json'); pre=read(OUT/'PREFLIGHT.json')
    rr=[r for r in rows() if r['split']=='test' and r['lead']==0]
    overall=[r for r in rr if r['channel']=='overall']
    aggregates={arm:summary([r for r in overall if r['arm']==arm]) for arm in ('N','F','O','C')}
    comparisons=[]
    for reference in ('O','F','N'):
        a=macro([r for r in overall if r['arm']=='C'],'score')
        b=macro([r for r in overall if r['arm']==reference],'score')
        assert a.keys()==b.keys()
        delta,lo,hi=bootstrap([a[k]-b[k] for k in sorted(a)])
        comparisons.append(dict(comparison='C-'+reference,C_step=selection['selected']['C'],
            reference_step=selection['selected'][reference],C_score=aggregates['C']['score'],
            reference_score=aggregates[reference]['score'],score_delta=delta,ci95_low=lo,ci95_high=hi,
            crps_delta=aggregates['C']['crps']-aggregates[reference]['crps'],
            es_delta=aggregates['C']['es']-aggregates[reference]['es'],
            mse_delta=aggregates['C']['mse']-aggregates[reference]['mse'],
            variance_delta=aggregates['C']['variance']-aggregates[reference]['variance'],
            time_mean_crps_delta=aggregates['C']['time_mean_crps']-aggregates[reference]['time_mean_crps']))
    with (OUT/'main_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    channels={arm:{ch:summary([r for r in rr if r['arm']==arm and r['channel']==ch])
        for ch in ('u','v')} for arm in aggregates}
    re={arm:{re:summary([r for r in overall if r['arm']==arm and r['Re']==re])
        for re in sorted({r['Re'] for r in overall})} for arm in aggregates}
    lead={arm:{ch:{str(k):float(np.mean([r['crps'] for r in rows() if r['split']=='test'
        and r['arm']==arm and r['channel']==ch and r['lead']==k])) for k in range(1,21)}
        for ch in ('u','v')} for arm in aggregates}
    updates=[json.loads(s) for s in (OUT/'events.jsonl').read_text().splitlines() if json.loads(s)['event']=='update']
    timing={arm:dict(updates=sum(r['arm']==arm for r in updates),
        seconds=sum(r['seconds'] for r in updates if r['arm']==arm),
        mean_update_seconds=float(np.mean([r['seconds'] for r in updates if r['arm']==arm])),
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in updates if r['arm']==arm)) for arm in aggregates if arm!='N'}
    dump(OUT/'analysis.json',dict(overall=aggregates,comparisons=comparisons,channel=channels,Re=re,
        lead_crps=lead,training_cost=timing,estimator='test empirical 32; dev/training U 16/8',
        bootstrap_unit='trajectory after window mean',bootstrap_repeats=10000,bootstrap_seed=123))
    primary=comparisons[0]
    if primary['score_delta']<0:
        conclusion='C对O的完整路径综合分数有点估计增量。'
        if primary['ci95_high']>=0:
            conclusion+='但配对轨迹95%区间跨零，当前证据不足以确认增量。'
        else:
            conclusion+='配对轨迹95%区间低于零；这仍只是单种子、旧探索性测试上的可行性信号。'
    else:
        conclusion='C没有超过O，本轮未获得当前预测双向反馈带来完整路径预测增量的证据。'
    if selection['selected']['C']==0:
        conclusion+='C开发选择回退到原生零步，不能把潜在状态变化或smoke梯度当作预测成功。'
    lines=['# 预测状态—潜在纠错状态双向耦合：决策报告','',conclusion,'',
        f"已共同完成F/O/C各{complete['common_updates']}次正式更新，停止原因：{complete['stop']}。"
        f"选择N/F/O/C分别为{[selection['selected'][a] for a in ('N','F','O','C')]}步。"
        '这次新假设独立接受检验，旧负结果没有被用作本结构有效或无效的证据。','',
        '## 唯一主要比较表','',
        '负差值表示C更好。测试分数=标准化经验CRPS+0.1×完整20帧RMS经验ES；与开发U估计不同。'
        '每轨迹先平均400/420两个窗口，12轨迹配对bootstrap10000次、seed123，只有下列三个预定比较。','',
        '| 比较 | C步/对照步 | C分数 | 对照分数 | 分数差 | 95%区间 | 原尺度MSE差 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in comparisons:
        lines.append(f"| {r['comparison']} | {r['C_step']}/{r['reference_step']} | {r['C_score']:.8g} | {r['reference_score']:.8g} | {r['score_delta']:.6g} | [{r['ci95_low']:.6g}, {r['ci95_high']:.6g}] | {r['mse_delta']:.6g} |")
    lines+=['','## 完整路径指标与开发候选','',
        '以下CRPS、ES、时间平均场CRPS都用已有训练标准差单位；MSE和方差为原尺度。MSE是集合均值与真值的场误差。']
    for arm,a in aggregates.items():
        lines.append(f"- {arm}：CRPS {a['crps']:.8g}，ES {a['es']:.8g}，MSE {a['mse']:.8g}，标准化MSE {a['smse']:.8g}，方差 {a['variance']:.8g}，时间平均场CRPS {a['time_mean_crps']:.8g}。")
    lines+=['','开发仅0/500/1000步固定16成员，最小CRPS_U+0.1ES_U选取，平分更早。所有非零候选如下，不能用回退隐藏其表现；完整记录在selection.json和scores.jsonl。']
    for a in selection['candidates']:
        if a['step']:
            lines.append(f"- {a['arm']} step{a['step']}：U综合 {a['score']:.8g}，CRPS_U {a['crps']:.8g}，ES_U {a['es']:.8g}，标准化MSE {a['smse']:.8g}，原尺度MSE {a['mse']:.8g}。")
    lines+=['','## 通道、工况与lead代价','',
        '所有分解都来自同一次32成员测试输出，未重新推理；scores.jsonl保存逐窗口u/v/lead的CRPS、MSE和方差，analysis.json保存通道、Re及lead汇总。以下差值均为C−对照，未追加置信区间或新的主要比较。']
    for ref in ('O','F','N'):
        lines.append(f"- C−{ref}通道综合分数："+'；'.join(f"{ch} {channels['C'][ch]['score']-channels[ref][ch]['score']:+.6g}" for ch in ('u','v'))+'。')
        lines.append(f"- C−{ref}各Re综合分数："+'；'.join(f"Re{g} {re['C'][g]['score']-re[ref][g]['score']:+.6g}" for g in re['C'])+'。')
        for ch in ('u','v'):
            dd={int(k):lead['C'][ch][k]-lead[ref][ch][k] for k in lead['C'][ch]}
            worst=max(dd,key=dd.get)
            lines.append(f"- C−{ref}/{ch}：20个lead中{sum(v>0 for v in dd.values())}个CRPS退步；最大差在lead{worst}，为{dd[worst]:+.6g}。")
    lines+=['','## 机制、参数与信息边界','',
        '正式权重逐张量核验up2和out发生更新，其他原始主干参数hash保持不变。O/C初始化与参数量相同，只有预测信息槽位不同；F参数较少，不能称各臂完全同容量。']
    for arm,count in train['counts'].items():
        lines.append(f"- {arm}：总参数{count['total']}，训练参数{count['trainable']}。")
    lines+=[f"预检标准化四步最大误差{max(pre['zero_feedback_max_abs_standardized'].values()):.6g}，容差2e−6。"
        f"初始反馈投影梯度L1={pre['gradients']['initial_feedback']:.6g}、初始G梯度为0；投影更新后G梯度L1={pre['gradients']['after_projection_G']:.6g}。"
        f"最终loss经R返回p3的梯度L1={pre['gradients']['final_loss_via_R_to_p3']:.6g}。",
        f"当前预测干预导致C状态最大变化{pre['interventions']['C_state_delta']:.6g}；O预测槽位屏蔽响应为0；"
        f"固定下一步x/H后输出变化{pre['interventions']['fixed_x_h_output_delta']:.6g}。单未来帧扰动影响相邻物理帧、真值改变不影响预测、H逐值不变、R每病例/成员重置均通过。",
        'G的3×3×3卷积只是一项局部物理帧耦合假设，不能解释成恢复真实PDE输运。R没有真实残差监督，不能称物理误差或校准不确定性。']
    if (OUT/'feedback_intervention.json').exists():
        for r in read(OUT/'feedback_intervention.json'):
            lines.append(f"固定开发干预{r['trajectory']}/400：反馈开/关U综合{r['on_score']:.8g}/{r['off_score']:.8g}，预测RMS变化{r['prediction_rms_delta']:.6g}。这只证明被使用，不能替代C−O。")
    else:
        lines.append('最终C选择零步，未追加关闭反馈干预。')
    lines+=['','## 成本、可复核性与限制','',
        f"GPU为{read(OUT/'environment.json')['gpu']}。预检M8峰值allocated {pre['peak_allocated_bytes']/1024**3:.3f}GiB，"
        f"reserved {pre['peak_reserved_bytes']/1024**3:.3f}GiB，未需容量降级。同步墙钟保守累计GPU占用{complete['conservative_gpu_seconds']:.3f}秒，"
        '低于7200秒；该计时包含进程中的I/O，不能视为纯kernel耗时。']
    for arm,v in timing.items():
        lines.append(f"- {arm}：正式更新累计{v['seconds']:.3f}秒，平均{v['mean_update_seconds']:.4f}秒/更新，allocated峰值{v['peak_allocated_bytes']/1024**3:.3f}GiB。")
    lines+=['每成员生成仍为4次主干调用，O/C另有3次G；完整四步反向及G产生真实计算/显存开销，不能称无成本。'
        '训练反向不增加教师；梯度检查点等价性已测，但本机容量足够，正式训练未启用重算。',
        f"新增预测缓存0bytes；本轮完成时artifact共{complete['total_artifact_bytes']}bytes（含训练权重），小于4GiB。"
        f"旧测试只有一个固定采样重复，实际生成{complete['test_inference_cases']}个病例×模型组；零步模型复用N。",
        '已有原始manifest含47条train，但本研究沿用既有46条；24150_20未重新纳入。固定训练窗口只按帧数筛选，排除项在config。'
        '系统h5py空包与pip权限故障通过独立.cache中校验SHA256的官方wheel修复，见environment_repairs.json；未改数据、尺度或增加正式训练。',
        'config/PROTOCOL、执行源码、每个0/500/1000权重均有SHA256；事件日志包含共同日程、loss、梯度范数、显存及生成身份。'
        '权重只保存可训练张量，与hash固定的原生checkpoint组合可重建，未覆盖旧权重。',
        '只有一个训练种子；12条旧测试轨迹、有限Re和bootstrap不覆盖训练种子不确定性。开发与旧测试已有历史使用，结果是可行性信号，不是稳定泛化或ICML级创新证明。'
        '没有自动部署、extension、oracle/JVP大矩阵、下一轮模块或超参数搜索。']
    if primary['score_delta']<0:
        lines+=['','建议仅在下一轮独立复验及机制消融中核查该信号；本轮不自动执行。']
    else:
        lines+=['','本结构和当前预算未证明双向反馈必要，停止本轮，不自动更换模块续试。']
    (OUT/'DECISION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('REPORT',primary,flush=True)


def verify():
    check_dependencies()
    c=cfg(); complete=read(OUT/'COMPLETE.json'); started=read(OUT/'TRAIN_STARTED.json')
    pre=read(OUT/'PREFLIGHT.json'); training=read(OUT/'training_summary.json'); sel=read(OUT/'selection.json')
    assert complete['code_hashes']==started['code_hashes']==pre['code_hashes']==code_hashes()
    assert started['config_sha256']==sel['config_sha256']==digest(OUT/'config.json')
    assert complete['common_updates']==training['common_updates']
    assert complete['conservative_gpu_seconds']<=7200 and complete['new_prediction_cache_bytes']==0
    events=[json.loads(s) for s in (OUT/'events.jsonl').read_text().splitlines()]
    updates=[r for r in events if r['event']=='update']
    common=complete['common_updates']
    assert len(updates)==3*common
    assert [(r['step'],r['arm']) for r in updates]==[(i,a) for i in range(1,common+1) for a in ('F','O','C')]
    for r in updates:
        schedule=c['schedule'][r['step']-1]
        assert r['window']==schedule['window'] and r['noise_seed']==schedule['noise_seed']
    for item in complete['checkpoint_hashes']:
        assert digest(ROOT/item['path'])==item['sha256']
    for arm in ('F','O','C'):
        a=torch.load(OUT/'weights'/f'{arm}_0.pt',map_location='cpu',weights_only=False)['state']
        final=max(v['step'] for v in complete['checkpoint_hashes'] if v['arm']==arm)
        b=torch.load(OUT/'weights'/f'{arm}_{final}.pt',map_location='cpu',weights_only=False)['state']
        assert all(k.startswith(('base.net.up2.','base.net.out.','update.','feedback.')) for k in a)
        if final:
            assert any(not torch.equal(a[k],b[k]) for k in a if k.startswith('base.net.up2.'))
            assert any(not torch.equal(a[k],b[k]) for k in a if k.startswith('base.net.out.'))
        assert training['parameter_updates'][arm]['frozen_unchanged']
    o=torch.load(OUT/'weights/O_0.pt',map_location='cpu',weights_only=False)['state']
    cc=torch.load(OUT/'weights/C_0.pt',map_location='cpu',weights_only=False)['state']
    assert o.keys()==cc.keys() and all(torch.equal(o[k],cc[k]) for k in o)
    data=rows()
    keys=[(r['split'],r['trajectory'],r['start'],r['arm'],r['step'],r['channel'],r['lead']) for r in data]
    assert len(keys)==len(set(keys))
    for arm in ('N','F','O','C'):
        test=[r for r in data if r['split']=='test' and r['arm']==arm]
        assert len(test)==len(c['windows']['test'])*43
        assert all(r['members']==32 and r['estimator']=='empirical' and r['step']==sel['selected'][arm] for r in test)
        dev=[r for r in data if r['split']=='development' and r['arm']==arm]
        assert all(r['members']==16 and r['estimator']=='U' for r in dev)
        expected=[0] if arm=='N' else sorted(set([0]+[i for i in (500,1000) if i<=common]))
        assert sorted({r['step'] for r in dev})==expected
        assert len(dev)==len(c['windows']['development'])*3*len(expected)
        if arm!='N':
            candidates=[]
            for step in expected:
                rr=[r for r in dev if r['step']==step and r['channel']=='overall']
                candidates.append((float(np.mean(list(macro(rr,'score').values()))),step))
            assert min(candidates)[1]==sel['selected'][arm]
    generated=[r for r in events if r['event']=='generated' and r['split']=='test']
    assert len(generated)==len(c['windows']['test'])
    assert sum(len(r['outputs']) for r in generated)==complete['test_inference_cases']
    selection_index=next(i for i,r in enumerate(events) if r['event']=='selection_frozen')
    assert all(i>selection_index for i,r in enumerate(events) if r['event']=='generated' and r['split']=='test')
    main=list(csv.DictReader((OUT/'main_comparison.csv').open()))
    assert [r['comparison'] for r in main]==['C-O','C-F','C-N']
    # Hash every delivery artifact except the self-referential audit file and open logs.
    artifacts={str(p.relative_to(ROOT)):digest(p) for folder in (OUT,ROOT/'research/coupled_state_feedback')
        for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='VERIFICATION.json'
        and '__pycache__' not in p.parts and p.suffix!='.log'}
    size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert size<=4*1024**3
    dump(OUT/'VERIFICATION.json',dict(passed=True,common_updates=common,formal_updates=len(updates),
        score_rows=len(data),test_cases=len(generated),test_inference_cases=complete['test_inference_cases'],
        protocol_and_dependency_hashes=True,initial_O_C_identical=True,
        actual_up2_out_updates=True,frozen_parameters_unchanged=True,selection_recomputed=True,
        test_after_selection=True,one_test_repeat=True,only_three_primary_comparisons=True,
        cache_bytes=0,artifact_bytes=size,artifacts_sha256=artifacts))
    print('VERIFIED',common,len(data),size,flush=True)
