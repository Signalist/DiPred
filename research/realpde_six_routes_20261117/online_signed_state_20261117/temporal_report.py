"""Do not average away infinite intervals or trade coverage for apparent width."""
from temporal_intervals import *
from probability_report import paired
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    d=pd.read_csv(ROOT/'windows.csv');metrics=['coverage','width','interval_score','infinite','reversed'];mean=lambda x:float(np.mean(x.to_numpy()));tr=d.groupby(['method','interval','task','file','group'])[metrics].agg(mean).reset_index();tr.to_csv(ROOT/'trajectories.csv',index=False);tasks=tr.groupby(['method','interval','task'])[metrics].agg(mean);tasks.to_csv(ROOT/'task_summary.csv');macro=tr.groupby(['method','interval','file'])[metrics].agg(mean).reset_index();summary=macro.groupby(['method','interval'])[metrics].agg(mean);summary.to_csv(ROOT/'summary.csv');rows=[]
    for method in ('base','pod_rw','geo_shift','lin_shift','memory_gated','local_shift'):
        a=macro[(macro.method==method)&(macro.interval=='field_online')].set_index('file')
        for label,b in macro[macro.method==method].groupby('interval'):
            b=b.set_index('file').reindex(a.index)
            if np.isfinite(b.interval_score).all():rows.append(dict(method=method,interval=label,reference='field_online',metric='interval_score',**paired(a.interval_score.values,b.interval_score.values)))
            else:rows.append(dict(method=method,interval=label,reference='field_online',metric='interval_score',gain_percent=-np.inf,ci_low=-np.inf,ci_high=-np.inf,trajectory_wins=int((b.interval_score<a.interval_score).sum()),trajectories=12))
    pd.DataFrame(rows).to_csv(ROOT/'paired_interval_gains.csv',index=False);tr.groupby(['method','interval','task','group'])[metrics].agg(mean).to_csv(ROOT/'by_re.csv');state_rows=[]
    for label in ('field_online','task_fixed','rolling10','spci_qrf_lag3','acmcp_block_h1','pid_no_scorecaster'):
        a=macro[(macro.method=='base')&(macro.interval==label)].set_index('file')
        for method in ('geo_shift','lin_shift','local_shift','memory_gated','pod_rw'):
            b=macro[(macro.method==method)&(macro.interval==label)].set_index('file').reindex(a.index);state_rows.append(dict(method=method,interval=label,reference='base_same_interval_algorithm',**paired(a.interval_score.values,b.interval_score.values)))
    pd.DataFrame(state_rows).to_csv(ROOT/'same_algorithm_state_gains.csv',index=False)
    lines=['# 同一完整块反馈下的时序任务区间对照','',
    '12条已用开发轨迹，每条按0:20:800连续发报41块；所有方法先收到10块完整20帧反馈，再评价31块，共372个窗口。预热120块明确计入协议。它不同于此前dev/late/future逐split冷启动，不能把两者的数值直接混用。全部任务、区间算法、状态模型在评价前冻结。',
    '', 'SPCI行是固定共同点预测器上的条件残差QRF适配：10棵树、深度2、5个beta候选，lag3为预定主设置，lag1/5仅敏感性；使用标准叶权经验分位数代替作者包。AcMCP行是按官方R代码翻译的块h=1递推，含P/I与均值scorecaster；未验证逐帧h=20的MA/跨horizon模块。其KI只来自90个独立校准窗口，残差按冻结训练任务尺度标准化。没有在开发轨迹之间拼接伪历史。',
    '', '这些方法只产生五个标量任务区间，不生成联合场，也不改变场点MSE。比较覆盖率、训练尺度归一化宽度和90% interval score；低分更好。覆盖率不足不能仅凭区间窄宣称优势。无限/反向区间如实保留，汇总显式传播无穷值。',
    '', '## 主要发现','',
    '同一连续协议下，直接从field_online提取的五任务90%区间宏覆盖只有69.14%，其中最大速度45.97%、网格涡量峰值59.68%、MVPE59.68%。逐场分位校准没有自动保证非线性任务覆盖，这是一条明确失败边界。',
    '', 'AcMCP块h=1宏覆盖90.27%，标准化宽度约为field_online的2.49倍；interval score为.6661对.6737。去掉均值scorecaster的PID对照覆盖91.02%、score .6004；本短块序列没有显示scorecaster的额外优势。SPCI适配主设置覆盖71.08%、score .4087；简单rolling10覆盖74.03%、score .3534。它们的评分虽更低，仍未达到90%目标，不能据此宣称已校准；有限同轨迹历史和适配实现也限制了对原方法的推断。',
    '', '在相同AcMCP块控制器下，geo_shift/lin_shift的score约.6624/.6626，接近base的.6661；局部静态修正为.6519。标量时序校准的收益不应归因于潜状态机制。相同算法下修正增益的轨迹区间另存same_algorithm_state_gains.csv。',
    '', '## 原始P为共同中心时的结果','', '| 区间方法 | 覆盖率% | 标准化宽度 | 标准化interval score | 无限区间% |','|---|---:|---:|---:|---:|']
    for label,r in summary.loc['base'].iterrows():lines.append(f'| {label} | {100*r.coverage:.2f} | {r.width:.4f} | {r.interval_score:.4f} | {100*r.infinite:.2f} |')
    lines+=['','## 同一算法作用于不同点修正','', '| 点方法 | 区间算法 | 覆盖率% | interval score |','|---|---|---:|---:|']
    for method in ('base','geo_shift','lin_shift','local_shift','memory_gated','pod_rw'):
        for label in ('field_online','task_fixed','rolling10','spci_qrf_lag3','acmcp_block_h1'):
            r=summary.loc[(method,label)];lines.append(f'| {method} | {label} | {100*r.coverage:.2f} | {r.interval_score:.4f} |')
    lines+=['','## 解释边界','',
    '五任务平均值仅是同一尺度下的宏平均；逐任务`task_summary.csv`和逐工况`by_re.csv`同时保留。`paired_interval_gains.csv`按轨迹bootstrap，参考是相同点方法的field_online。所有结果探索性，未作独立外部确认或多重比较校正。标量区间对照的作用是揭示时序校准本身的收益，不能归因于新的物理潜状态，也不能替代整场ES/VS和自由滚动测试。',
    '', '实际发报/完整反馈日志见`events.jsonl`；保存端点见`issued_intervals.pt`；独立索引AcMCP递推、原端点重放和未来残差干预审计见`audit.json`。源码、协议和下载的官方源码hash见`provenance.json`。这不是原作者依赖栈的逐字节复现，报告不会把适配结果冒充完整SPCI/AcMCP多步基准。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    labels=['field_raw','field_fixed','field_online','task_fixed','rolling10','spci_qrf_lag3','acmcp_block_h1','pid_no_scorecaster'];fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    finite_scores=[summary.loc[('base',label),'interval_score'] for label in labels if np.isfinite(summary.loc[('base',label),'interval_score'])];cap=max(finite_scores)*1.2
    for i,label in enumerate(labels):
        r=summary.loc[('base',label)];axes[0].bar(i,r.coverage*100);axes[1].bar(i,r.interval_score if np.isfinite(r.interval_score) else cap)
        if not np.isfinite(r.interval_score):axes[1].text(i,cap,'infinite',ha='center',va='bottom',rotation=90)
    axes[0].axhline(90,color='black',ls='--');axes[0].set_ylabel('Task coverage (%)');axes[1].set_ylabel('Normalized interval score (lower is better)')
    for ax in axes:ax.set_xticks(range(len(labels)));ax.set_xticklabels(labels,rotation=40,ha='right');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same 10-block warm-up; 372 scored blocks on 12 reused trajectories');fig.tight_layout();fig.savefig(ROOT/'interval_evidence.png',dpi=180);fig.savefig(ROOT/'interval_evidence.pdf');plt.close(fig);print(summary.loc['base'].to_string(),flush=True)

if __name__=='__main__':main()
