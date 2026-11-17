"""Independent mask/score accounting, clustered uncertainty and full report."""
from allocation import *
from itertools import combinations
import hashlib

def markdown(df):
    if len(df)==0:return '(empty)'
    def val(x):return f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(val(x) for x in r)+' |' for r in df.itertuples(index=False,name=None))

def audit():
    checks={};maxerror=0.;quota_tests=0
    selection=json.loads((OUT/'selection.json').read_text());assert selection['protocol_sha256']==s.digest(W/'compute_allocation_20261117/PROTOCOL.md')
    sets={split:set(frame(s.load(OUT/f'{split}_observables.pt')).file) for split in ('train','calibration','development','late','future')}
    assert not (sets['train']&sets['calibration'] or sets['train']&sets['development'] or sets['calibration']&sets['development'])
    assert sets['late']==sets['development']==sets['future']
    for split in ('calibration','development','late','future'):
        d=s.load(OUT/f'{split}_observables.pt');issued=pd.read_csv(OUT/f'{split}_issued_masks.csv');records=pd.read_csv(OUT/f'{split}_windows.csv');ia=json.loads((OUT/f'{split}_issue_audit.json').read_text())
        assert ia['models_sha256']==s.digest(OUT/'models.pt') and ia['masks_sha256']==s.digest(OUT/f'{split}_issued_masks.csv')
        assert not ia['targets_or_scores_read']
        for (_,q,start),g in issued.groupby(['policy','quota','start']):
            k=int(np.floor(q*len(g)+.5));assert int(g.expensive.sum())==k
            # Independent pandas top-k ordering, not allocation.choose.
            expected=set(g.sort_values(['priority','file'],ascending=[False,True]).head(k).file)
            assert expected==set(g[g.expensive==1].file);quota_tests+=1
        for (mode,action),r in records.groupby(['mode','action']):
            low,high=paired_scores(split,d,mode,action)
            for (policy,q),part in r.groupby(['policy','quota']):
                part=align(part,d);prob=part.expensive.to_numpy()
                for metric in METRICS:
                    check=low[metric].to_numpy()*(1-prob)+high[metric].to_numpy()*prob
                    maxerror=max(maxerror,float(np.max(np.abs(check-part[metric].to_numpy()))))
        # Test the analytic random expectation by enumerating all quota subsets
        # for the first batch (up to12 requests); no Monte Carlo assumption.
        low,high=paired_scores(split,d);ix=frame(d).groupby('start').indices[min(frame(d).start)]
        k=len(ix)//2;lo=low.functional_crps.to_numpy()[ix];hi=high.functional_crps.to_numpy()[ix]
        exact=np.mean([np.where(np.isin(np.arange(len(ix)),c),hi,lo).mean() for c in combinations(range(len(ix)),k)])
        formula=((1-k/len(ix))*lo+(k/len(ix))*hi).mean();assert abs(exact-formula)<1e-12
        checks[split]=dict(windows=len(d['meta']),trajectories=len(frame(d).file.unique()),random_expectation_enumeration_error=abs(exact-formula))
    assert maxerror<1e-12
    s.dump(OUT/'accounting_audit.json',dict(passed=True,quota_tests=quota_tests,score_composition_max_error=maxerror,splits=checks,frozen_protocol_hash_matches=True,train_calibration_development_trajectory_disjoint=True,late_future_same_development_trajectories=True,notes='Independent pandas top-k and exhaustive first-batch random subsets; score source numeric audit inherited from lowcost_joint; bootstrap conditions on frozen policies and realized batches.'))

def main():
    audit();selection=json.loads((OUT/'selection.json').read_text());rows=[];pergroup=[];tasks=[];diagnostics=[];allsummary=[];feature=[]
    for split in ('development','late','future'):
        tr=pd.read_csv(OUT/f'{split}_trajectories.csv');su=pd.read_csv(OUT/f'{split}_summary.csv');su['split']=split;allsummary.append(su)
        for mode in ('raw','fixed','online'):
            for action in ('smooth','secc3'):
                d=tr[(tr['mode']==mode)&(tr.action==action)]
                for q in (.25,.5,.75):
                    benefit=d[(d.policy=='benefit')&(d.quota==q)].set_index('file').sort_index()
                    for baseline in ('random_expected','random','risk','width','oracle'):
                        base=d[(d.policy==baseline)&(d.quota==q)].set_index('file').loc[benefit.index]
                        delta=(base.functional_crps-benefit.functional_crps).to_numpy();rng=np.random.default_rng(42);boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
                        rows.append(dict(split=split,mode=mode,action=action,quota=q,baseline=baseline,gain_percent=100*delta.mean()/base.functional_crps.mean(),absolute_gain=delta.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),wins=int((delta>0).sum()),trajectories=len(delta)))
                    for metric in METRICS:
                        full=d[(d.policy=='benefit')&(d.quota==1.)].set_index('file').loc[benefit.index]
                        random=d[(d.policy=='random_expected')&(d.quota==q)].set_index('file').loc[benefit.index]
                        tasks.append(dict(split=split,mode=mode,action=action,quota=q,metric=metric,score=benefit[metric].mean(),gain_vs_full_percent=100*(1-benefit[metric].mean()/full[metric].mean()),gain_vs_equal_random_percent=100*(1-benefit[metric].mean()/random[metric].mean())))
                for (policy,q,re),g in d.groupby(['policy','quota','group']):pergroup.append(dict(split=split,mode=mode,action=action,policy=policy,quota=q,group=re,functional_crps=g.functional_crps.mean(),expensive=g.expensive.mean()))
        for kind in KINDS:
            f=su[(su['mode']=='raw')&(su.action=='smooth')&(su.policy=='benefit/'+kind)&(su.quota==.5)].iloc[0]
            feature.append(dict(split=split,representation=kind,dimensions=selection['models']['benefit/'+kind]['dimensions'],alpha=selection['models']['benefit/'+kind]['alpha'],functional_crps=f.functional_crps))
        obs=s.load(OUT/f'{split}_observables.pt');low,high=paired_scores(split,obs);gain=low.functional_crps-high.functional_crps;masks=pd.read_csv(OUT/f'{split}_issued_masks.csv')
        for policy in ('benefit','risk','width'):
            pred=align(masks[(masks.policy==policy)&(masks.quota==.5)],obs).priority
            diagnostics.append(dict(split=split,policy=policy,spearman_with_true_benefit=pred.corr(gain,method='spearman'),spearman_with_cheap_loss=pred.corr(low.functional_crps,method='spearman')))
    paired=pd.DataFrame(rows);paired.to_csv(OUT/'paired.csv',index=False);pd.DataFrame(pergroup).to_csv(OUT/'by_re.csv',index=False);task=pd.DataFrame(tasks);task.to_csv(OUT/'task_gains.csv',index=False);pd.DataFrame(diagnostics).to_csv(OUT/'priority_diagnostics.csv',index=False);pd.DataFrame(feature).to_csv(OUT/'representations.csv',index=False)
    summary=pd.concat(allsummary,ignore_index=True);summary.to_csv(OUT/'all_summary.csv',index=False)
    primary=paired[(paired['mode']=='raw')&(paired.action=='smooth')&(paired.quota==.5)&paired.baseline.isin(['random_expected','risk','width'])]
    calibration=paired[(paired.action=='smooth')&(paired.quota==.5)&(paired.baseline=='random_expected')]
    speed=task[(task['mode']=='raw')&(task.action=='smooth')&(task.quota==.5)&task.metric.isin(['functional_crps','crps_speed_max'])]
    secc=paired[(paired['mode']=='raw')&(paired.action=='secc3')&(paired.quota==.5)&(paired.baseline=='random_expected')]
    endpoints=[]
    for split in ('development','late','future'):
        d=summary[(summary.split==split)&(summary['mode']=='raw')&(summary.policy=='benefit')]
        routed=float(d[(d.action=='smooth')&(d.quota==.5)].functional_crps.iloc[0])
        for action in ('smooth','secc3'):
            base=float(d[(d.action==action)&(d.quota==1)].functional_crps.iloc[0]);endpoints.append(dict(split=split,baseline='always_'+action,main_half_budget_gain_percent=100*(1-routed/base)))
    timing=pd.read_csv(OUT/'timing.csv') if (OUT/'timing.csv').exists() else pd.DataFrame()
    all_tasks=task[(task['mode']=='raw')&(task.action=='smooth')&(task.quota==.5)]
    re_table=pd.DataFrame(pergroup);re_table=re_table[(re_table['mode']=='raw')&(re_table.action=='smooth')&(re_table.quota==.5)&re_table.policy.isin(['benefit','random_expected'])].pivot(index=['split','group'],columns='policy',values='functional_crps').reset_index();re_table['gain_percent']=100*(1-re_table['benefit']/re_table['random_expected'])
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,3.8))
    for ax,split in zip(axes,('development','late','future')):
        d=summary[(summary.split==split)&(summary['mode']=='raw')&(summary.action=='smooth')]
        for policy in ('benefit','risk','width','random_expected','oracle'):
            g=d[d.policy==policy].sort_values('quota');ax.plot(g.quota*100,g.functional_crps,marker='o',label=policy)
        se=float(summary[(summary.split==split)&(summary['mode']=='raw')&(summary.action=='secc3')&(summary.policy=='benefit')&(summary.quota==1)].functional_crps.iloc[0]);ax.axhline(se,color='gray',ls=':',label='always S-ECC3')
        ax.set_title(split);ax.set_xlabel('Expensive request fraction (%)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Physical functional mean CRPS');axes[-1].legend(fontsize=8);fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/allocation.png',dpi=170);fig.savefig(OUT/'figures/allocation.pdf');plt.close(fig)
    chosen=selection['models'][selection['chosen']['benefit']]
    report=f'''# 预期收益分配计算：固定预算下出现可复核的任务收益

主要动作是廉价AR+PCA与昂贵32路径smooth之间选择；主要指标预先固定为五个训练尺度标准化物理泛函的平均经验CRPS。**50%昂贵调用预算下，预计收益策略在三个时段的均值都优于同预算的随机期望、预测风险和区间宽度分配；相对随机期望，development和late的轨迹级95%区间仍跨零。** 这是一项同步多请求服务实验；不等价于单轨迹在线调度，也不是新的独立数据确认。

|说明|结果|
|---|---|
|训练|92窗口/46轨迹；原RealPDE、seed42；标签=廉价损失−昂贵损失|
|模型|{chosen['kind']}，训练PCA {chosen['dimensions']}维，带截距17个监督读出权重，ridge alpha={chosen['alpha']}；PCA/标准化参数另存，不冒充总共17参数|
|选择|90校准窗口/10轨迹，在25/50/75%预算的平均策略损失上选择；随后冻结全部动作掩码才读取评估评分|
|评估|development108、late120、future72窗口，同12开发轨迹；逐轨迹平均和配对区间|
|信息|共同四步P/F与49边际头；路由只读已观测X和当前P/F/宽度；新增路径仅在选中请求执行|

## 同预算比较

正数表示预计收益路由降低主要评分。CI是基线−本策略的绝对评分差，按轨迹重采样10000次；它描述固定策略/实际请求批次，未包含拟合/选模不确定性，未将窗口或像素当独立样本。

{markdown(primary[['split','baseline','gain_percent','ci_low','ci_high','wins','trajectories']])}

完整25/50/75%结果见`paired.csv`，全部预算曲线如下。0/100%为共同端点。50%不是查看开发结果后选出的最佳预算；预先指定的全部预算均保留。随机期望由固定配额下每请求被选概率精确计算，不是多训练种子结果；另保留单seed42随机实现。

![预算—任务损失](figures/allocation.png)

## 代价与强对照

与始终使用昂贵动作比较：

{markdown(pd.DataFrame(endpoints))}

最大速度仍有代价，不能把宏平均收益说成所有任务都改善：

{markdown(speed[['split','metric','gain_vs_full_percent','gain_vs_equal_random_percent']])}

把同一冻结掩码的昂贵动作换成S-ECC3的敏感性分析（未重训该收益模型）：

{markdown(secc[['split','gain_percent','ci_low','ci_high','wins']])}

逐个任务、Re组、全部表示和全部失败配置分别保留在`task_gains.csv`、`by_re.csv`、`representations.csv`、`selection_grid.csv`。S-ECC3完整端点也在图中，不能只与较弱的smooth比较。

相同预算下的收益并非每项任务共同改善。平均收益包含涡量和MVPE方面的改善；TKE、动能或VS在部分时段变差：

{markdown(all_tasks[['split','metric','gain_vs_equal_random_percent']])}

逐Re均值（仍不是独立的新工况确认）：

{markdown(re_table)}

预测风险与真实动作收益的Spearman相关在三个时段为负，而收益读出为正；这是本数据中风险排名无法替代收益排名的直接诊断，不是一般定理：

{markdown(pd.DataFrame(diagnostics))}

## 校准与表示

同一动作掩码用于raw/fixed/online共同边际校准，避免把改变校准归功于路由。50%预算、相对同预算随机期望：

{markdown(calibration[['split','mode','gain_percent','ci_low','ci_high']])}

等监督维度网格的X/P/XP/F读出全部保留：

{markdown(pd.DataFrame(feature))}

本实验支持冻结预测器表示可用于简单收益读出，不证明扩散F增加了新观测信息或有独有优势。同四调用确定性表征与小标注对照已有`../error_structure_20261117/REPORT.md`中的限制；这里没有重新证明不同训练目标的因果优势。

## 实际分发成本

{markdown(timing)}

预载模型，A100，12个同时到达请求串行GPU执行，包含共同预测、特征提取、路由、所选联合生成和保存GPU输出；不含磁盘与评分。七轮按seed42随机顺序交错测量，完整原始耗时与测量次序见timing_raw.json；初始顺序计时保留作诊断。记录总批次延迟及其除以12的平均服务成本，后者不能称为单用户响应延迟。昂贵分支复用共同P/F/边际，不重新运行49头；不先生成随后丢弃的廉价样本。比较相同配额策略实测成本，非只计主干调用数。

## 审计与边界

`accounting_audit.json`独立用pandas重算top-k配额、逐指标组合评分，并枚举首批次所有半预算子集核对随机期望。`dispatch_audit.json`直接从观测历史分发预测，检查与冻结掩码及缓存评分一致。上游场评分、边际相同、物理坐标和合法延迟校准沿用已完成数值审计。

廉价生成器和收益监督都用了同92训练窗口，因此训练动作标签存在拟合内乐观风险；真正比较使用轨迹分离的cal/development，后者有历史复用。分配是在同相对发行时刻的多轨迹批次中进行，不声称这些实验在现实日历同时发生，不排名未来窗口；跨轨迹资源竞争也意味着bootstrap仅作描述性证据。单seed符合当前小数据探索范围。

本路线已有具体用途收益，仍需要新轨迹/新工况独立确认。整体六路线目标仍active；在线有符号状态、主动观测与经验物理任务误差估计继续推进。原主干、默认推理和外部仓库未替换。
'''
    (OUT/'REPORT.md').write_text(report);s.dump(OUT/'provenance.json',dict(source_files={str(p.relative_to(W)):s.digest(p) for p in (W/'compute_allocation_20261117').glob('*.py')},protocol_sha256=s.digest(W/'compute_allocation_20261117/PROTOCOL.md'),model_sha256=s.digest(OUT/'models.pt'),selection_sha256=s.digest(OUT/'selection.json')))
    print(primary[['split','baseline','gain_percent','ci_low','ci_high']].to_string(index=False),flush=True)

if __name__=='__main__':main()
