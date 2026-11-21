"""Small trajectory-paired tables and bounded audit report; no model forwards."""
from run import *
from collections import defaultdict


def read(p):
    with p.open() as f:return list(csv.DictReader(f))


def interval(values):
    a=np.asarray(values);rng=np.random.default_rng(123);v=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    return np.quantile(v,[.025,.975])


def table(rows,keys):
    output=['| '+' | '.join(keys)+' |','| '+' | '.join('---' for _ in keys)+' |']
    for r in rows:output.append('| '+' | '.join(f'{r[k]:.7g}' if isinstance(r[k],float) else str(r[k]) for k in keys)+' |')
    return '\n'.join(output)+'\n'


def main():
    assert (OUT/'COMPLETE.json').exists() and not (OUT/'SUMMARY.json').exists()
    complete=json.loads((OUT/'COMPLETE.json').read_text());assert complete['passed']
    rows=read(OUT/'trajectory_scores.csv');delta=read(OUT/'delta_by_trajectory.csv')
    ix={(r['split'],r['trajectory'],int(r['step']),r['segment'],r['channel']):r for r in rows}
    groups=defaultdict(list)
    for r in rows:groups[r['split'],int(r['step']),r['segment'],r['channel']].append(r)
    summaries=[];pairs=[]
    for (split,step,segment,channel),rr in groups.items():
        rr=sorted(rr,key=lambda r:r['trajectory']);assert len(rr)==12
        row=dict(split=split,step=step,segment=segment,channel=channel,trajectories=12)
        for m in METRICS:
            a=np.array([float(r[m]) for r in rr]);d=np.array([float(r[m])-float(ix[split,r['trajectory'],0,segment,channel][m]) for r in rr]);lo,hi=interval(d)
            row.update({m:float(a.mean()),m+'_delta':float(d.mean()),m+'_ci95_low':float(lo),m+'_ci95_high':float(hi),m+'_improved':int((d<0).sum())})
        summaries.append(row)
        if step:
            for r in rr:pairs.append(dict(split=split,trajectory=r['trajectory'],Re=r['Re'],step=step,segment=segment,channel=channel,**{m+'_delta':float(r[m])-float(ix[split,r['trajectory'],0,segment,channel][m]) for m in METRICS}))
    csvwrite(OUT/'score_summary.csv',summaries);csvwrite(OUT/'paired_trajectory.csv',pairs)
    groups=defaultdict(list)
    for r in delta:groups[r['split'],int(r['step']),int(r['denoise']),r['region'],r['channel']].append(r)
    ds=[]
    for (split,step,denoise,region,channel),rr in groups.items():
        assert len(rr)==12 and len({r['elements'] for r in rr})==1
        ds.append(dict(split=split,step=step,denoise=denoise,region=region,channel=channel,trajectories=12,elements_per_trajectory=int(rr[0]['elements']),rms_mean=float(np.mean([float(r['rms']) for r in rr])),rms_min=min(float(r['rms']) for r in rr),rms_max=max(float(r['rms']) for r in rr)))
    csvwrite(OUT/'delta_summary.csv',ds)
    overall=[r for r in summaries if r['channel']=='overall'];lookup={(r['split'],r['step'],r['segment']):r for r in overall}
    patterns=[]
    for step in [250,1000]:
        tf=lookup['train',step,'full'];to=lookup['train',step,'overlap'];do=lookup['development',step,'overlap']
        clues=[]
        if tf['crps_delta']>=0:clues.append('A:训练子集full均值未改善')
        if tf['crps_delta']<0 and to['crps_delta']>=0:clues.append('B:训练full改善但overlap未改善')
        if to['crps_delta']<0 and do['crps_delta']>0:clues.append('C:训练overlap改善而开发overlap退步')
        uncertain=any(r['crps_ci95_low']<=0<=r['crps_ci95_high'] for r in [tf,to,do])
        patterns.append(dict(step=step,mean_pattern=clues or ['D:混合模式'],uncertain_intervals=uncertain,train_full_delta=tf['crps_delta'],train_overlap_delta=to['crps_delta'],development_overlap_delta=do['crps_delta']))
    dump(OUT/'SUMMARY.json',dict(patterns=patterns,bootstrap_unit='original trajectory; train subset is purposive Re-balanced12, not full46 estimate',bootstrap_samples=10000,bootstrap_seed=123,score_summary_rows=len(summaries),delta_summary_rows=len(ds)))
    report='''# 冻结条件适配瓶颈核查

本轮固定清单完成，参数全部冻结、优化器更新数为0。只核查有效C_aug/seed42/step0、250、1000，k5/start400/repeat0/32成员。当前执行HEAD为1d118a1d3a433a9111d42d5fa904a84eeb40cf69；没有训练、搜索或新的测试/extension评分，也未推送。旧作废训练批次没有载入，只在旧成本证据中出现。

既有有效九模型的科学/保护18项选择全部step0，已由selection.json核验。本轮不更改选择，不把C_aug结果外推成C_mask/C_unc独立信息作用的否定。

训练12轨迹按Re编号数值升序、组内文件名字典序：每组首条后同序轮流补齐。原开发12轨迹全用。清单及缓存/权重hash在manifest.json、config.json，均先于运行冻结。训练子集是定位线索，不能代表全46训练轨迹。

## 核心表1：同一输出的三个评分时段

物理帧采用0基：子完整预测425–444，overlap425–439，extra440–444。每个checkpoint一次32成员完整前向，三个时段来自同一输出和同一次20帧真值读取。CRPS包含成员到真值项及成员间距离半项；所有标准化使用原冻结归一化。差值为当前步−step0，负值表示分数下降。95%区间按12原轨迹配对bootstrap10000次、seed123，仅定位性，不宣称多重比较确认。

'''
    report+=table(overall,['split','step','segment','crps','crps_delta','crps_ci95_low','crps_ci95_high','crps_improved','mse','smse'])
    report+='\nscore_summary.csv完整保留overall/u/v、CRPS两项与MSE/SMSE差及区间；trajectory_scores.csv和paired_trajectory.csv保留每条轨迹。\n\n## 核心表2：实际历史条件修改的位置\n\n以下为四次去噪的delta在标准化历史坐标中的区域RMS，先按各区域元素数归一化，再等权平均原轨迹。观测位置15帧、生成位置5帧，不比较未经归一化范数。delta通过原适配器forward hook记录，没有修改sample数学行为、添加硬mask或新修正；幅度仅是结构诊断，不是因果损害。\n\n'
    report+=table([r for r in ds if r['channel']=='overall'],['split','step','denoise','region','elements_per_trajectory','rms_mean','rms_min','rms_max'])
    report+='\ndelta_by_trajectory.csv保留每条轨迹、去噪99/66/33/0、u/v/overall身份。step0仅读取零适配器delta，不重复主干。\n\n## 解释线索\n\n'
    for p in patterns:
        report+=f"- step{p['step']}：{'；'.join(p['mean_pattern'])}。训练full差{p['train_full_delta']:.8g}，训练overlap差{p['train_overlap_delta']:.8g}，开发overlap差{p['development_overlap_delta']:.8g}。关键区间是否跨零：{p['uncertain_intervals']}。\n"
    report+='''
A只说明这些固定训练样本未获改善，不能据随机在线loss宣布欠拟合。B才是“full目标改善但overlap未改善”的时段权衡线索；C只支持本子集上训练/开发差异的泛化线索，不能单凭该模式证明过拟合。若不同step/轨迹/通道方向不同或区间跨零，归为D混合或证据不足；上述线索可并存，不强制唯一原因。已有在线日志抽样不同，本轮没有将其当固定训练性能曲线。

完整CRPS由两项组成。额外5帧改善/退步对full的权重为25%，不能直接用extra绝对分数与overlap绝对分数比较训练成败；必须看同一时段相对step0的差。没有重新选checkpoint、拟合门控或修改lr/loss/容量。

## 数值、信息边界与成本

'''
    report+=f"CRPS、其两项、原/标准化MSE均验证full=.75overlap+.25extra，最大绝对误差{complete['segment_identity_maxabs']:.4g}，固定容差1e−12；原开发相同身份评分及step0原生评分重算最大误差{complete['existing_score_maxabs']:.4g}，容差1e−12。任何不一致原实现会写STOP并停止，本轮无失败。旧step0物理复现阈值2e−6由已有预检及本轮零层权重/缓存身份核验，无新增零步主干重放。\n\n"
    report+=f"新增完整前向{complete['new_forward_groups']}组={complete['new_paths']}条路径、{complete['new_network_batch_calls']}次批网络调用；复用step0 {complete['reused_step0_groups']}组={complete['reused_paths']}路径，共72条件/checkpoint组。没有新父路径。GPU同步生成计时{complete['generation_seconds']:.3f}秒，执行墙钟{complete['wall_seconds']:.3f}秒；预测缓存新增0bytes，只保存小型统计与hash。最终交付文件体积见VERIFICATION.json，代码/文档/评分I/O成本不等于GPU推理时延，GPU耗时不是Codex费用。\n\n"
    report+='''全部参数eval/requires_gradFalse，无优化器；输入与旧S5历史、父A、32成员子噪声实际hash对应。每条三个预测和delta都先完成并写发布事件，才读取425–444未来标签。旧协议/代码/权重/结果逐文件hash保持不变。不重算旧时空矩阵，不重放全缓存，仅648行评分和1728行delta小表。

复现入口见research/conditioning_failure_audit/README.md。FROZEN、COMPLETE、identities、events、score_replay_checks及最终VERIFICATION提供完成证据；原数据/权重不复制，大缓存不新增。

**唯一下一动作：建议将本次固定参数化的训练支线归档，保留这两张定位表；本轮不启动任何新训练或测试集实验。**
'''
    (OUT/'DECISION_REPORT.md').write_text(report)
    print('SUMMARIZED',patterns)

if __name__=='__main__':main()
