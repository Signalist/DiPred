"""Observation-budget evidence, including VOI failures against risk selection."""
from observations import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compare(a,b):
    rng=np.random.default_rng(42);ix=rng.integers(0,len(a),(10000,len(a)));g=(1-b.mean()/a.mean())*100;boot=(1-b[ix].mean(1)/a[ix].mean(1))*100
    return dict(gain_percent=float(g),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),trajectory_wins=int((b<a).sum()),trajectories=len(a))

def report(root):
    summaries=[];rows=[];by_re=[];efficiency=[];predicted=[]
    for split in ('calibration','development','late','future'):
        d=pd.read_csv(root/split/'trajectories.csv');summaries.append(d.assign(split=split));d=d.set_index(['method','budget','file']);base=d.loc[('uniform',0)]
        for method in d.index.get_level_values('method').unique():
            for budget in BUDGETS:
                current=d.loc[(method,budget)].reindex(base.index)
                if budget:
                    for ref in ('raw','uniform','variance','risk','random_expectation16'):
                        a=base if ref=='raw' else d.loc[(ref,budget)].reindex(base.index);rows.append(dict(split=split,method=method,budget=budget,reference=ref,**compare(a.mse.values,current.mse.values)))
                for group,g in current.groupby('group'):by_re.append(dict(split=split,method=method,budget=budget,group=group,trajectories=len(g),mse=float(g.mse.mean()),raw_gain_percent=float((1-g.mse.mean()/base.loc[g.index,'mse'].mean())*100)))
            if method in ('voi','variance','risk','random_expectation16'):
                target=d.loc[('uniform',16)].reindex(base.index)
                for b in BUDGETS:
                    actual=d.loc[(method,b)].reindex(base.index);efficiency.append(dict(split=split,method=method,budget=b,reference='uniform16',mean_loss_matches=bool(actual.mse.mean()<=target.mse.mean()),**compare(target.mse.values,actual.mse.values)))
        w=pd.read_csv(root/split/'windows.csv');w=w[(w.method=='voi')&(w.budget==8)];predicted.append(dict(split=split,predicted=float(w.predicted_reduction.mean()),actual=float(w.realized_reduction.mean()),correlation=float(w[['predicted_reduction','realized_reduction']].corr().iloc[0,1])))
    gains=pd.DataFrame(rows);gains.to_csv(root/'paired_gains.csv',index=False);pd.DataFrame(efficiency).to_csv(root/'same_loss_observation_budget.csv',index=False);pd.DataFrame(by_re).to_csv(root/'by_re.csv',index=False);pd.DataFrame(predicted).to_csv(root/'predicted_vs_realized.csv',index=False);alltr=pd.concat(summaries);alltr.to_csv(root/'all_trajectories.csv',index=False)
    title='可观测未来误差低秩回归' if root.name=='predictive' else '全局低秩残差条件更新';lines=[f'# 主动观测：{title}','',
    '本任务新增了未来第5帧的稀疏观测：每个位置同时观测u/v，随后只评价第6—20帧。这不是原始无新增观测20→20预测的提升。所有策略使用同一已冻结同化器；主预算为8点/16个标量，位置在任何观测值到达前确定。均匀、空间覆盖、方差、风险、VOI和平均尺度VOI均保留。',
    '', '训练819窗口/46轨迹，校准90/10。超参仅由均匀策略4/8/16/32预算的校准损失选择。dev/late/future沿用同12条开发轨迹，后两组不能作独立确认。随机策略含一个可执行固定流，以及同一RNG的16次反事实抽样均值；后者估计随机策略期望，并非同时获得16倍观测。',
    '', '## 主预算8点：相对不增加观测的MSE改善','', '| 方法 | dev% | late% | future% |','|---|---:|---:|---:|']
    for method in ['uniform','spacefill','random','random_expectation16','variance','risk','voi','voi_average_scale']:
        values=[]
        for split in ['development','late','future']:values.append(f"{gains[(gains.split==split)&(gains.method==method)&(gains.budget==8)&(gains.reference=='raw')].iloc[0].gain_percent:.3f}")
        lines.append('| '+' | '.join([method]+values)+' |')
    lines+=['','## VOI在同预算下相对强对照','', '| 时间段 | 对照 | MSE改善% | 95%轨迹CI | 胜出轨迹 |','|---|---|---:|---|---:|']
    for r in gains[(gains.split!='calibration')&(gains.method=='voi')&(gains.budget==8)&gains.reference.isin(['uniform','variance','risk','random_expectation16'])].to_dict('records'):lines.append(f"| {r['split']} | {r['reference']} | {r['gain_percent']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['trajectory_wins']}/12 |")
    lines+=['','## 观测效率与解释边界','',
    '完整0/1/2/4/8/16/32预算曲线保留。same_loss_observation_budget.csv逐一比较各预算与同时间段uniform16的真实后续损失，同时给出轨迹区间；均值达到目标不代表统计确认。不能仅凭VOI的模型内预计下降宣称减少观测。',
    '', '模型是由训练残差估计的线性二阶矩代理，并非真实PDE的保证或可识别物理参数。所有方法的未观测坐标不进入同化器；零预算精确返回原P。主要损失是未被直接观测的未来15帧物理MSE，没有把观测帧的重构误差混入收益。',
    '', '逐窗/逐轨迹结果位于各split目录；完整轨迹和逐工况表为all_trajectories.csv和by_re.csv；原始选点/掩码为requests.pt。模型内预期和实际收益的差异见predicted_vs_realized.csv。独立代数、数值重建、真实推理、掩码隔离和时刻审计见audit.json。同seed42，不将像素当独立样本；轨迹CI为探索性，未作多重比较确认。']
    (root/('REPORT.md' if root.name=='predictive' else 'global_REPORT.md')).write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True);styles={'uniform':'#666666','random_expectation16':'#b19c76','variance':'#6d89b7','risk':'#ba7148','voi':'#217c5f','voi_average_scale':'#a368a3'}
    for ax,split in zip(axes,['development','late','future']):
        d=alltr[alltr.split==split]
        for method,color in styles.items():
            v=d[d.method==method].groupby('budget')[['mse','base_mse']].mean();ax.plot(v.index,100*(1-v.mse/v.base_mse),'o-',label=method,color=color,ms=3)
        ax.axhline(0,color='black',lw=.7);ax.set_title(split);ax.set_xlabel('Observed sites (2 velocity scalars/site)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Future15 MSE reduction vs raw P (%)');axes[0].legend(fontsize=8);fig.suptitle(title if False else 'New observations at lead5; same assimilation across policies');fig.tight_layout();fig.savefig(root/'budget_curve.png',dpi=180);fig.savefig(root/'budget_curve.pdf');plt.close(fig)
    print(gains[(gains.method=='voi')&(gains.budget==8)&(gains.split!='calibration')].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictive',action='store_true');args=p.parse_args();report(OUT/'predictive' if args.predictive else OUT)
