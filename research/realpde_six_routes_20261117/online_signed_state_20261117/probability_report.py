"""Trajectory-weighted probability evidence and explicit scope limits."""
from probability import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SPLITS=['development','late','future'];METRICS=['es','vs','marginal_crps','functional_crps','crps_tke_peak','crps_kinetic_mean','crps_enstrophy_grid_peak','crps_speed_max','crps_mvpe','point_mse','sample_mean_mse']

def paired(a,b):
    rng=np.random.default_rng(42);ix=rng.integers(0,len(a),(10000,len(a)));gain=(1-b.mean()/a.mean())*100;boots=(1-b[ix].mean(1)/a[ix].mean(1))*100
    return dict(gain_percent=float(gain),ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),trajectory_wins=int((b<a).sum()),trajectories=len(a))

def main():
    rows=[];groups=[];alltr=[]
    for split in SPLITS:
        d=pd.read_csv(FOLDER/split/'trajectories.csv');alltr.append(d.assign(split=split))
        for mode in ('raw','fixed','online'):
            a=d[d['mode']==mode].set_index(['method','file']);base=a.loc['base']
            for method in NAMES[1:]:
                b=a.loc[method].reindex(base.index)
                for metric in METRICS:rows.append(dict(split=split,mode=mode,method=method,reference='base',metric=metric,**paired(base[metric].values,b[metric].values)))
                for group in sorted(base.group.unique()):
                    mask=base.group==group
                    for metric in METRICS:groups.append(dict(split=split,mode=mode,method=method,group=group,metric=metric,gain_percent=float((1-b.loc[mask,metric].mean()/base.loc[mask,metric].mean())*100),trajectories=int(mask.sum())))
            for method in ('geo_shift','lin_shift'):
                ref=a.loc['local_shift'];b=a.loc[method].reindex(ref.index)
                for metric in METRICS:rows.append(dict(split=split,mode=mode,method=method,reference='local_shift',metric=metric,**paired(ref[metric].values,b[metric].values)))
    gains=pd.DataFrame(rows);gains.to_csv(FOLDER/'paired_gains.csv',index=False);pd.DataFrame(groups).to_csv(FOLDER/'by_re.csv',index=False);pd.concat(alltr).to_csv(FOLDER/'all_trajectories.csv',index=False)
    def table(methods,metrics,modes=('raw','fixed','online')):
        lines=['| 方法 | 校准 | 指标 | dev收益% [95%CI] | late收益% [95%CI] | future收益% [95%CI] |','|---|---|---|---|---|---|']
        for method in methods:
            for mode in modes:
                for metric in metrics:
                    cells=[]
                    for split in SPLITS:
                        r=gains[(gains.method==method)&(gains['mode']==mode)&(gains.metric==metric)&(gains.split==split)&(gains.reference=='base')].iloc[0];cells.append(f'{r.gain_percent:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]')
                    lines.append('| '+' | '.join([method,mode,metric]+cells)+' |')
        return '\n'.join(lines)
    lines=['# 在线有符号状态：概率扩展结果','',
    '状态模型和概率扩展规则在评价前冻结。主要问题仍是原点MSE修正是否也改善概率预测；未根据本轮概率结果重新选择状态模型或改换主要用途。三组评估分别108/120/72窗口，均为同12条开发轨迹，不能作为三次独立确认。训练seed42；每次复用同一四步P/F、49头和32个AR+PCA联合样本。',
    '', '## 结论与取舍','',
    '历史几何状态的平移方案在三组时间窗、raw/fixed/online三种模式下均改善ES和VS；五任务平均CRPS的收益较小，固定校准下开发集反而略退步。它不能被表述成概率任务全面改进。固定POD随机游走KF和滚动整体偏置没有稳定改善ES。',
    '', '对每个成员做warp，能改善最大速度CRPS，但五任务平均CRPS显著退步；这说明单个样本的平滑/形变和条件分布质量不能混为一谈。静态local_shift的点MSE更好，几何历史平移的ES更好；这些指标支持的主张不同。上一轮热启动200帧自由滚动退步仍然成立。',
    '', table(['geo_shift','lin_shift','local_shift'],['es','functional_crps']),
    '', '## 成员形变的代价（次要消融，不用于择优）','',table(['geo_member_warp','lin_member_warp','local_member_warp'],['functional_crps','crps_speed_max'],('raw',)),
    '', '## 校准和审计','',
    '每个方法在相同90窗口/10轨迹拟合自己的32×2分位偏移；使用相同拟合公式、空间stride4、相同在线.5更新权重。在线状态和偏移均逐轨迹/逐split冷启动，只在完整20帧真值揭示后更新。平移方案保留原始样本离散度；member_warp同时改变边际和依赖。固定/在线校准后的点MSE始终是独立确定性修正场的MSE，另报sample_mean_mse避免混淆。',
    '', 'MVPE目标始终相对同一个原始P定义；涡量采用dx=+.003421664、dy=−.003421664。CRPS评价固定32节点经验分布，没有不适用的fair修正。字段覆盖率只作诊断，不宣称有限样本条件保证。所有区间以轨迹重采样，未作多重比较确认性校正。',
    '', '独立NumPy检查8个场快照×9种输出，ES/VS/五任务CRPS/覆盖/宽度/样本均值MSE最大误差见audit.json。点预测完整重放、既有低成本基线重放、校准偏移完全一致、未来目标干预的既发前缀不变、轨迹隔离和提前反馈拒绝均通过。',
    '', '审计文件：`audit.json`、`numerical_audit.csv`、`replay_audit.csv`；完整指标：各split的`windows.csv`/`trajectories.csv`/`summary.csv`、`paired_gains.csv`、`by_re.csv`。物理任务样本保存于各split/parts；不只保留有利指标。',
    '', 'SPCI/AcMCP的共同预热块任务比较另见`../temporal_intervals/REPORT.md`。本扩展不是完整六方向目标完成；新增观测VOI和经验物理任务误差估计等仍需推进。']
    (FOLDER/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex='col');methods=['pod_rw','memory_gated','local_shift','lin_shift','geo_shift','lin_member_warp'];colors=['#bc6460','#9a886e','#557b9b','#319282','#27694e','#9662a7']
    for col,split in enumerate(SPLITS):
        for row,metric in enumerate(['es','functional_crps']):
            ax=axes[row,col]
            for j,(method,color) in enumerate(zip(methods,colors)):
                r=gains[(gains.split==split)&(gains['mode']=='raw')&(gains.reference=='base')&(gains.metric==metric)&(gains.method==method)].iloc[0];ax.errorbar(r.gain_percent,j,xerr=[[r.gain_percent-r.ci_low],[r.ci_high-r.gain_percent]],fmt='o',color=color,capsize=3)
            ax.axvline(0,color='black',lw=.7);ax.set_yticks(range(len(methods)));ax.set_yticklabels(methods if col==0 else []);ax.set_title(f'{split}: {metric}');ax.set_xlabel('Loss reduction vs baseline (%)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Frozen online corrections: raw probabilities; trajectory 95% bootstrap intervals');fig.tight_layout();fig.savefig(FOLDER/'probability_evidence.png',dpi=180);fig.savefig(FOLDER/'probability_evidence.pdf');plt.close(fig)
    print(gains[(gains.reference=='base')&(gains['mode']=='raw')&(gains.method.isin(['geo_shift','lin_shift','local_shift']))&(gains.metric.isin(['es','functional_crps']))].to_string(index=False),flush=True)

if __name__=='__main__':main()
