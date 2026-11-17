"""Three retained mechanisms; do not convert a development win into generality."""
from critic import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def paired(a,b):
    rng=np.random.default_rng(42);ix=rng.integers(0,len(a),(10000,len(a)));v=(1-b[ix].mean(1)/a[ix].mean(1))*100
    return dict(gain_percent=float(100*(1-b.mean()/a.mean())),ci_low=float(np.quantile(v,.025)),ci_high=float(np.quantile(v,.975)),wins=int((b<a).sum()),n=len(a))

def main():
    rows=[];efficiency=[];alltr=[];grouped=[]
    for split in ('calibration','development','late','future'):
        parent=pd.read_csv(ROOT/split/'trajectories.csv');critic=pd.read_csv(CRITIC/split/'trajectories.csv');d=pd.concat([parent,critic],ignore_index=True);alltr.append(d.assign(split=split));indexed=d.set_index(['method','budget','file']).sort_index();baseline=indexed.loc[('uniform',0)]
        for budget in BUDGETS[1:]:
            b=indexed.loc[('critic',budget)].reindex(baseline.index)
            for reference in ('raw','uniform','variance','risk','random_expectation16','voi','voi_average_scale'):
                a=baseline if reference=='raw' else indexed.loc[(reference,budget)].reindex(baseline.index);rows.append(dict(split=split,budget=budget,method='critic',reference=reference,**paired(a.mse.values,b.mse.values)))
        for method in ('critic','voi','risk','variance'):
            for reference,target_budget in [('uniform',16),('random_expectation16',32)]:
                a=indexed.loc[(reference,target_budget)].reindex(baseline.index)
                for budget in BUDGETS:
                    b=indexed.loc[(method,budget)].reindex(baseline.index);efficiency.append(dict(split=split,method=method,budget=budget,reference=reference,reference_budget=target_budget,mean_matches=bool(b.mse.mean()<=a.mse.mean()),**paired(a.mse.values,b.mse.values)))
        for (method,budget,group),g in d.groupby(['method','budget','group']):grouped.append(dict(split=split,method=method,budget=budget,group=group,mse=float(g.mse.mean()),gain_percent=float(100*(1-g.mse.mean()/g.base_mse.mean())),trajectories=len(g)))
    gains=pd.DataFrame(rows);gains.to_csv(CRITIC/'paired_gains.csv',index=False);pd.DataFrame(efficiency).to_csv(BASE/'observation_efficiency.csv',index=False);pd.DataFrame(grouped).to_csv(BASE/'predictive_and_critic_by_re.csv',index=False);alltr=pd.concat(alltr);alltr.to_csv(BASE/'predictive_and_critic_trajectories.csv',index=False);summary=alltr.groupby(['split','method','budget'])[['mse','base_mse']].mean();summary['gain_percent']=100*(1-summary.mse/summary.base_mse);summary.to_csv(BASE/'predictive_and_critic_summary.csv')
    selected=json.loads((CRITIC/'selection.json').read_text());lines=['# 新增稀疏观测：有小幅用途收益，VOI尚未稳定超过风险选点','',
    '**已完成三种简单机制的同预算探索、实际带掩码推理和审计。** 观察驱动的原20→20预测保持原定义；这里在未来第5帧新增k个位置的u/v观测，再发出后续15帧预测，只评分这15帧。单个位置成本为2个标量。所有位置在观测值到达前选定，各策略共用完全相同的同化器。主预算8点，完整曲线0/1/2/4/8/16/32点。',
    '', '## 三种机制与结论','',
    '1. **全局低秩残差二阶矩**：标准化真实训练残差的32因子+对角未建模项，均匀策略校准选rank32、remainder1、gain.5。8点VOI相对原P改善0.27/0.88/0.44%，但不及风险选点。完整负结果见[第一阶段](global_REPORT.md)。',
    '2. **可观测未来误差回归**：直接拟合第5帧候选观测与未来15帧残差的交叉关系，保留完整256×256观测二阶矩及rank32未来因子。共同同化器仍只由均匀校准选择，选shrink.5、gain1。8点VOI改善1.09/0.97/1.18%，仍未超过风险策略1.38/1.98/1.43%。见[第二阶段](predictive/REPORT.md)。',
    f'3. **直接学习观测收益**：在训练集模拟单点观测的实际未来MSE下降，使用复用的16维X/P/F表示和两个局部量，拟合128×19=2432个线性系数。共同同化器不变；校准选ridge={selected["alpha"]}、冗余惩罚={selected["penalty"]}。8点收益为2.07/1.02/1.18%。开发集超过风险选点，但later时间段反向，说明收益没有稳定延续。',
    '', '三者都是经过明确记录的顺序探索；后两者在看过前一阶段开发诊断后提出，不能冒充预注册独立验证。保持同seed42、同原RealPDE，不训练新主干、不增加联合样本数。',
    '', '## 相同同化器、相同8点预算','', '| 策略 | dev MSE改善% | late% | future% |','|---|---:|---:|---:|']
    for method in ('uniform','spacefill','random_expectation16','variance','risk','voi','voi_average_scale','critic'):
        vals=[summary.loc[(s,method,8),'gain_percent'] for s in ('development','late','future')];lines.append('| '+method+' | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
    lines+=['','直接收益评分器相对同预算风险策略的轨迹bootstrap比较：','', '| 时间段 | MSE改善% | 95%CI | 胜出轨迹 |','|---|---:|---|---:|']
    for r in gains[(gains.budget==8)&(gains.reference=='risk')&(gains.split!='calibration')].to_dict('records'):lines.append(f"| {r['split']} | {r['gain_percent']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['wins']}/12 |")
    lines+=['','## 是否真的减少观测','',
    '风险策略8点在三组时间窗的MSE均值都低于随机策略32点，但三个配对轨迹区间均跨零。因此这里只能报告“均值上少用75%位置达到该随机基准”，不能宣称已确认75%的观测节约。相对uniform16，优势还随时间段变化。observation_efficiency.csv的完整离散比较没有做单调化或插值，也没有隐去增加观测后反而退步的点。',
    '', '数据表为`observation_efficiency.csv`；随机期望用同一seed42随机流的16次反事实选点估计，每次仍只使用k点，不是向某个策略提供16k点。训练819窗口/46轨迹、校准90/10；dev108/12、late120与future72沿用同12轨迹。轨迹CI为描述性，后两段不是独立确认，未作多重比较确认性校正。',
    '', '## 数值与因果审计','',
    '两种二阶矩模型均独立用NumPy观测空间求解和完整未来场重建校验；MSE差约1e-11。第二种VOI的8步选点还逐步用完整已选子集Schur公式重算。真实API从CPU完整历史发报，只接受选中的k×2物理观测；提前反馈、传入整块真值均拒绝，未观测坐标和未来15帧真值干预不改变更新结果。零预算精确返回原P。直接收益评分器的请求精确重放，训练单点收益标签亦经场重建核对。',
    '', '审计文件分别为`audit.json`、`predictive/audit.json`、`critic/audit.json`；模型、校准网格、逐窗误差、选点掩码与发报时刻均保存。未出现为达到结论而放宽误差阈值。实测计算耗时见timing.csv；获取观测所需的5帧等待单列为任务时刻，不计入算法毫秒数。',
    '', '## 当前判断','',
    '新增观测确实能在这一小数据设置降低未来误差，简单风险选点是必须保留的强对照。但两种相关结构VOI和一个直接收益评分器都没有稳定胜过它；当前证据不足以支持“更准确的VOI已带来可靠观测效率优势”。这是对这些小模型和本协议的有界负证据，不是证明所有主动观测或VOI方法无效。下一阶段优先推进尚未覆盖的经验物理任务误差估计与直接修正强对照，避免只在同一开发轨迹上继续堆选点变体。']
    timing=BASE/'timing.csv'
    if timing.exists():
        t=pd.read_csv(timing);lines+=['','## 实测计算开销（batch1中位数）','', '| 同化器 | 策略 | 主预测+选点ms | 收到观测后更新ms | 合计ms |','|---|---|---:|---:|---:|']
        for r in t[t.method.isin(['uniform','risk','voi','critic'])].to_dict('records'):lines.append(f"| {r['family']} | {r['method']} | {r['forecast_and_planning_ms']:.2f} | {r['update_ms']:.2f} | {r['total_compute_ms']:.2f} |")
        lines+=['','CPU物理历史输入与CPU观测输入，6窗口×5次交错重复；排除加载模型、离线训练和传感器等待。内存为当前进程torch GPU分配峰值，详见timing_samples.csv。']
    (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n');(CRITIC/'REPORT.md').write_text('# 直接学习观测收益\n\n完整相同同化器比较见[总报告](../REPORT.md)。\n\n本模型在开发集胜过风险选点，但在late反向、future区间跨零；没有稳定转移证据。协议见源码CRITIC_PROTOCOL.md。\n\n'+gains[(gains.budget==8)&(gains.reference=='risk')].to_csv(index=False))
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True);colors={'uniform':'#777777','random_expectation16':'#b49c77','risk':'#b86d4a','voi':'#427aa6','critic':'#318562'}
    for ax,split in zip(axes,['development','late','future']):
        for method,color in colors.items():
            v=summary.loc[(split,method)];ax.plot(v.index,v.gain_percent,'o-',label=method,color=color,ms=3)
        ax.axvline(8,color='black',ls=':',lw=.7);ax.axhline(0,color='black',lw=.6);ax.set_title(split);ax.set_xlabel('Observed sites (two scalars/site)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Future15 MSE reduction vs raw forecast (%)');axes[0].legend(fontsize=8);fig.suptitle('Same predictive-mode assimilation: observation benefits vary across time');fig.tight_layout();fig.savefig(BASE/'all_policy_budget_curve.png',dpi=180);fig.savefig(BASE/'all_policy_budget_curve.pdf');plt.close(fig)
    dump(BASE/'completion_manifest.json',dict(full_six_route_goal_complete=False,active_observation_pilot='three mechanisms evaluated; limited gains, no stable VOI superiority',source_files={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},protocols={p.name:digest(p) for p in Path(__file__).parent.glob('*PROTOCOL.md')},audits={name:json.loads((root/'audit.json').read_text())['passed'] for name,root in [('global',BASE),('predictive',ROOT),('critic',CRITIC)]}));print('ACTIVE_OBSERVATION_REPORT_DONE',flush=True)

if __name__=='__main__':main()
