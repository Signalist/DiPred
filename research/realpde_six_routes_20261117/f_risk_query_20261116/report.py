"""Build tables, figures, and an evidence-linked Chinese development report."""
from common import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize import main as summarize

def table(df):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return '—' if not np.isfinite(v) else f'{v:.6f}'
        return str(v).replace('|','\\|').replace('\n',' ')
    rows=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    rows.extend('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
    return '\n'.join(rows)
def main():
    summarize();std=norm('cpu')['std'].flatten().numpy();figs=OUT/'figures';figs.mkdir(exist_ok=True)
    r=pd.read_csv(OUT/'risk/main/summary.csv');a=r[r.region=='all'].groupby(['method','mode']).mean(numeric_only=True)
    point=a.loc[(list(('F','P','XPF','X','Fdet','Pprojection','Frandom','Fshuffle')),'raw'),:][['normalized_crps','coverage','normalized_width','normalized_interval']].reset_index().drop(columns='mode')
    point.to_csv(OUT/'fixed_center_summary.csv',index=False)
    rd=pd.read_csv(OUT/'risk/main/per_group.csv');rd=rd[(rd.region=='all')&(rd['mode']=='raw')&(~rd.method.str.startswith('old_'))].copy();rd['score']=rd.crps/rd.channel.map(dict(enumerate(std)))
    pivot=rd.groupby(['method','file']).score.mean().unstack(0);diff=(pivot.P-pivot.F).values;rng=np.random.default_rng(42);boots=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(1)
    f_gain=100*(a.loc[('P','raw'),'normalized_crps']-a.loc[('F','raw'),'normalized_crps'])/a.loc[('P','raw'),'normalized_crps']
    center=[]
    for arm in ('X','P','F','XPF'):
        raw=a.loc[('old_'+arm,'raw')];aligned=a.loc[('old_'+arm,'center_aligned')]
        center.append(dict(method=arm,raw_crps=raw.normalized_crps,aligned_crps=aligned.normalized_crps,physical_mse=raw.mse))
    center=pd.DataFrame(center);center.to_csv(OUT/'center_decomposition.csv',index=False)
    curves=[]
    for tag in ('width8','main','width32','labels25','labels50','symmetric'):
        d=pd.read_csv(OUT/'risk'/tag/'summary.csv');s=d[(d.region=='all')&(d['mode']=='raw')].groupby('method').mean(numeric_only=True)
        for arm in ('F','P'):
            ck=load(OUT/'risk'/tag/(arm+'.pt'));curves.append(dict(experiment=tag,method=arm,width=ck['width'],parameters=ck['parameters'],labeled_trajectories=len(ck['training_files']),crps=s.loc[arm,'normalized_crps'],coverage=s.loc[arm,'coverage']))
    curves=pd.DataFrame(curves);curves.to_csv(OUT/'capacity_data_curves.csv',index=False)
    local=pd.read_csv(OUT/'local/per_group.csv');ls=local.groupby(['shape','strategy','target','region','channel'])[['coverage','width','interval']].mean();localrows=[]
    for key,b in ls.groupby(level=[0,1,2]):
        allb=b.xs('all',level='region');detail=local[(local['shape']==key[0])&(local.strategy==key[1])&(local.target==key[2])].groupby(['region','channel','group']).coverage.mean()
        localrows.append(dict(shape=key[0],strategy=key[1],calibration_target=key[2],coverage=allb.coverage.mean(),width=allb.width.mean(),interval=allb.interval.mean(),worst_region_channel_coverage=b.coverage.min(),worst_region_channel_Re_coverage=detail.min()))
    local_summary=pd.DataFrame(localrows);local_summary.to_csv(OUT/'local/summary.csv',index=False)
    q=pd.read_csv(OUT/'query/summary.csv');qs=q.groupby(['method','category','statistic'])[['normalized_crps','coverage']].mean().reset_index()
    both=qs[qs.category=='new_both'].pivot(index='method',columns='statistic',values='normalized_crps').reset_index();both.to_csv(OUT/'query/new_queries_summary.csv',index=False)
    events=pd.read_csv(OUT/'query/event_summary.csv');ev=events.groupby(['method','category'])[['brier','prevalence']].mean().reset_index();eb=ev[ev.category=='new_both'][['method','brier','prevalence']]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    fig,axes=plt.subplots(1,3,figsize=(14,4));x=np.arange(4)
    axes[0].bar(x-.18,center.raw_crps,.36,label='Learned median',color='#8aa6c1');axes[0].bar(x+.18,center.aligned_crps,.36,label='Common P median',color='#294e73');axes[0].set_xticks(x);axes[0].set_xticklabels(center.method);axes[0].set_ylabel('Standardized CRPS');axes[0].set_title('Published heads: center decomposition');axes[0].legend(fontsize=8)
    for arm,col in [('F','#176e9c'),('P','#c2773e')]:
        v=curves[(curves.method==arm)&curves.experiment.isin(['width8','main','width32'])].sort_values('parameters');axes[1].plot(v.parameters,v.crps,'o-',label=arm,color=col)
        v=curves[(curves.method==arm)&curves.experiment.isin(['labels25','labels50','main'])].sort_values('labeled_trajectories');axes[2].plot(v.labeled_trajectories,v.crps,'o-',label=arm,color=col)
    axes[1].set_xlabel('Risk-head parameters');axes[1].set_ylabel('Standardized CRPS');axes[1].set_title('Matched capacity, fixed median');axes[1].legend();axes[2].set_xlabel('Labeled training trajectories');axes[2].set_title('Matched label budget, fixed median');axes[2].legend();fig.tight_layout();fig.savefig(figs/'mechanism.png');fig.savefig(figs/'mechanism.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for arm,col in [('constant','#999999'),('P','#c2773e'),('F','#176e9c'),('XPF','#446f51')]:
        for strategy,ls_ in [('global','--'),('groups','-')]:
            v=local_summary[(local_summary['shape']==arm)&(local_summary.strategy==strategy)].sort_values('calibration_target');ax.plot(v.width,1-v.worst_region_channel_coverage,'o'+ls_,label=f'{arm} / {strategy}',color=col,markersize=4)
    ax.set_xlabel('Mean standardized interval width');ax.set_ylabel('Worst region/channel miscoverage');ax.set_title('Development: width vs local risk (lower is better)');ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(figs/'local_tradeoff.png');fig.savefig(figs/'local_tradeoff.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4));methods=['X','P','F','XPF','noise8','noise32','F_ECC8','F_ECC32']
    for i,stat in enumerate(('mean','max')):
        v=both.set_index('method').loc[methods,stat];axes[i].bar(np.arange(len(v)),v.values,color=['#aaaaaa','#c2773e','#176e9c','#446f51','#a2a3c1','#646787','#9ac3d2','#518296']);axes[i].set_xticks(np.arange(len(v)));axes[i].set_xticklabels(methods,rotation=35,ha='right');axes[i].set_title('New region + time: '+stat);axes[i].set_ylabel('Standardized query CRPS')
    fig.tight_layout();fig.savefig(figs/'query_generalization.png');fig.savefig(figs/'query_generalization.pdf');plt.close(fig)
    if (OUT/'query/latency.csv').exists():
        lat=pd.read_csv(OUT/'query/latency.csv');fig,ax=plt.subplots(figsize=(6,4))
        for name,b in lat.groupby('method'):ax.plot(b.queries,b.median_ms,'o-',label=name)
        ax.set_xlabel('Queries sharing one forecast');ax.set_ylabel('Median latency (ms), batch=1');ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(figs/'query_latency.png');fig.savefig(figs/'query_latency.pdf');plt.close(fig)
    else:lat=pd.DataFrame()
    verification=json.loads((OUT/'verification.json').read_text());sym=curves[curves.experiment=='symmetric'][['method','crps','coverage']]
    baselines=pd.read_csv(OUT/'query/event_baselines.csv');bb=baselines[baselines.category=='new_both'].groupby('channel')[['prevalence','zero_probability_brier','calibration_climatology_brier']].mean().reset_index()
    text=f'''# RealPDE：F 风险读出、局部可靠性与物理查询实验

本轮完成，训练随机种子统一 **42**。只使用原 `train_real`，没有训练五场景模型，没有打开新的测试数据。这里全部性能数字来自历史开发集；它已被多次用于研究，不是独立最终确认。

## 研究结论

1. **F 的优势有一部分来自分布中心，但不完全来自中心。** 旧头对齐中位数后，F 仍优于 X/P；从头训练并固定共同中位数 P，F 的 CRPS={a.loc[('F','raw'),'normalized_crps']:.8f}，P={a.loc[('P','raw'),'normalized_crps']:.8f}，相对改善 **{f_gain:.3f}%**。12 条开发轨迹配对的 P−F 平均差={diff.mean():.8f}，轨迹重采样95%区间 [{np.quantile(boots,.025):.8f}, {np.quantile(boots,.975):.8f}]，改善轨迹 {int((diff>0).sum())}/12。此区间不等于独立Re组的置信保证。
2. **当前 F 的平均评分优势没有自动带来局部可靠性优势。** 在相同90%校准目标下，F 分组区间比 P 分组更宽、最差区域/通道覆盖更低。简单分组已经明显优于统一常宽；本轮不支持宣称 F 局部校准优于强 P 对照。
3. **普通 F 查询头未通过筛选。** 对新的空间区域和时间组合，F 的区域最大值 CRPS 弱于 X/P。使用 F 边际与真实样本秩耦合的简单 ECC 改善了部分原生样本查询：ECC32在新查询的区域均值上优于本轮直接头，但最大值任务仍弱于X/P，而且仍需要32条真实扩散路径。保留各任务的不同结论，暂不增加复杂一致性模块。
4. 固定中位数允许偏度改变分布均值，因此另外完成了**对称分布控制**，其均值和中位数都固定为 P：F的CRPS为0.082814、P为0.083450，F仍低约0.762%。容量与标注量对照也显示F在更小头、较少标注时的相对优势更明显；这些是单种子开发证据。

## 数据、预算与隔离

- 输入与输出都是完整20帧、u/v、32×64。原冻结四步 DiPred 不重训。
- 风险训练：原46条可用训练轨迹，200:400 和满足长度条件的400:600，819窗口；归一化来自原训练数据，特征统计只来自这些训练窗口。
- 校准：10条独立身份、90窗口；开发：12条独立身份、108窗口、3个Re组。分位头不在校准/开发数据上更新权重。
- 主风险比较8组；每组2000更新、batch4、完全相同初始化与抽样次序。额外容量/标注量曲线均只训练F/P；查询4组各4000更新、batch256。确定性F来自已有同结构6000次MSE训练的一调用模型，四步扩散与它的调用/训练计算不同，不能据此单独证明训练目标的因果作用。
- 所有风险模型针对同一个 P 的误差。Pprojection 用固定40→48投影控制维度；Frandom是固定随机初始化主干；Fshuffle训练时打乱窗口对应，开发时用训练特征作固定不对应的供体。因此shuffle是无当前对应信息的负控制，并非完全匹配分布的因果干预。
- 容量/标注曲线的子集按seed42嵌套抽取，特征归一化仍使用全训练集的无标签特征，故只称“标注轨迹效率”。

## 中心与形状

所有CRPS与宽度按原训练u/v标准差分别归一化，再等权平均。点式分布使用99分位数和端点常值尾部做2×pinball梯形积分；原生ensemble与ECC用真实经验CRPS，有限样本评分与连续分位函数积分存在估计差异。

旧共享参考头的中心分解：

{table(center)}

固定共同中位数、重新训练风险形状：

{table(point)}

双中心对称控制（均值与中位数都固定为P）：

{table(sym)}

容量与标注量曲线：

{table(curves)}

![中心分解与数据效率](figures/mechanism.png)

## 局部覆盖与宽度

10组在看未来真值之前定义：全域、三个时距、四象限、预测梯度高/低区域。梯度阈值由训练预测场隔点抽样的80%分位确定。空组不计入。

先审计原始区间及独立校准集上的全局加性CQR。再固定形状，比较全局乘性尺度与重叠组最大尺度，校准目标预设为80/85/90/95%。下表是90%校准目标；“最差区域/通道”按开发轨迹平均后取最差，也另报进一步按Re拆分的最差值。指标均为经验量，没有像素独立性或严格条件覆盖保证。

{table(local_summary[local_summary.calibration_target==.9])}

![局部风险宽度曲线](figures/local_tradeoff.png)

## 跨查询泛化与事件风险

训练48个矩形×时段查询；开发分别考察48个已见、32个新空间、48个新时间、32个空间时间均新的查询。每个共享头同时预测区域时段平均值与单速度分量最大值的分布。输入以区域均值/标准差/最大值汇聚X、P或48通道F，再附查询坐标与时段；这次负结果只否定该普通汇聚读出器，不是否定所有F查询方法。

新的区域和时间组合，标准化CRPS：

{table(both)}

![查询泛化](figures/query_generalization.png)

最大值超阈阈值只用训练最大值的50/75/90分位确定；以下Brier在两通道、三阈值等权平均：

{table(eb)}

事件发生率与朴素参考（两通道分别列出）：

{table(bb)}

部分u高阈值事件极少，低Brier可能由稀有事件造成，不能据此认定事件风险可用。`query/decisions_calibration_fpr10.csv`用已见查询的校准负例确定10%误报目标阈值，报告开发实际误报/漏报；没有正例时漏报率为空，不能宣称零漏报。概率可靠性见 `query/reliability_bins.csv`。

查询阈值单调性由分位函数保证；普通头的区域/时段包含关系仍有违例，逐方法见 `query/inclusion_violations.json`。同一组完整场样本复用到全部查询，原生ensemble与ECC均通过包含关系核验。ECC只是经典秩重排控制，不作为新方法。

## 成本

{table(lat[lat.queries==160]) if len(lat) else '独立耗时测量待完成。'}

耗时为A100、batch1、模型已载入，包含实际预测器调用、全部查询汇聚与三分位输出；每组ensemble只生成一次并供所有查询复用。排除模型/数据加载与CPU传输。比较仅说明这个实现的查询开销，不能在质量未达标时宣称等可靠性加速。

![查询数量与耗时](figures/query_latency.png)

## 验证与产物

- 独立NumPy重算风险CRPS，最大误差 {verification['numpy_risk_crps_max_error']:.3g}；查询CRPS {verification['numpy_query_crps_max_error']:.3g}；Brier {verification['numpy_query_brier_max_error']:.3g}；按O(K²)定义重算ensemble CRPS {verification['numpy_pairwise_ensemble_crps_max_error']:.3g}。
- 实际四步预测回放最大误差 {verification['actual_four_call_point_replay_max_error']:.3g}；固定中位数、训练初始化/抽样匹配、分位单调、目标污染不影响输入/预测、查询标签切片和样本事件包含关系核验通过。
- 每个模型保存seed/预算/参数与训练日志；来源哈希核对未变。完整检查见 `verification.json` 与最终 `completion_audit.json`。
- `risk/*/per_group.csv`：逐轨迹、窗口、通道、区域结果；`query/per_query.csv` 与 `query/ensemble_per_query.csv`：完整逐查询结果；`query/paired_trajectory.csv`：以轨迹为单位的配对差；`local/per_group.csv`：完整局部校准曲线；权重与实际预测快照同目录保存。
- 源码与复现实验命令：`../../f_risk_query_20261116/README.md`。原历史实验文件未修改。

## 下一研究判断

优先保留①“误差形状可读性”为目前有正证据的方向；现有结果不足以主张扩散独有机制。②的下一方法必须超过这里的强P分组对照；⑥先需要保留时空结构的查询读出器和有足够正例的预注册事件协议，本轮普通F查询头不应直接作为论文主结果。建议在新的独立轨迹上确认之前，继续把这些结果称为开发集筛选。
'''
    (OUT/'REPORT.md').write_text(text)
    print('REPORT',OUT/'REPORT.md',flush=True)
if __name__=='__main__':main()
