"""Record all failed hypotheses, cost/quality tradeoffs, and scope limits."""
from study import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
STAGES={'raw':OUT,'conditional':OUT/'conditional_scores','moment_ar':OUT/'conditional_scores/ar','scorefit':OUT/'conditional_scores/scorefit'}
METRICS=['es','vs','functional_crps','crps_tke_peak','crps_kinetic_mean','crps_enstrophy_grid_peak','crps_speed_max','crps_mvpe']

def table(df):
    def f(v):return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(f(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
    stage_rows=[];paired=[];regroups=[];marginal_audit={};secc_identity=[];selection={}
    for stage,folder in STAGES.items():
        sel=json.loads((folder/'selection.json').read_text());method=sel['selected'];selection[stage]=sel
        for split in ('calibration','development','late','future'):
            s=pd.read_csv(folder/split/'summary.csv',index_col=0)
            row=dict(stage=stage,split=split,method=method,quality_gate=bool((s.loc[method,['es','vs','functional_crps']]/s.loc['smooth',['es','vs','functional_crps']]<=1.02).all()))
            row.update({k+'_gain_percent':100*(1-s.loc[method,k]/s.loc['smooth',k]) for k in METRICS});stage_rows.append(row)
            maximum=0.;rawdiff=0.
            for path in sorted((folder/split/'parts').glob('*.pt')):
                d=load(path)
                for name,err in d['marginal_error'].items():
                    if name in ('necc3','necc9'):rawdiff=max(rawdiff,err)
                    else:maximum=max(maximum,err)
            assert maximum==0
            marginal_audit[f'{stage}/{split}']=dict(fixed_marginal_max_error=maximum,unrestored_necc_max_error=rawdiff)
            if split!='calibration':
                tr=pd.read_csv(folder/split/'trajectories.csv');a=tr[tr.method==method].set_index('file')
                for ref in ('ecc32','smooth','secc3'):
                    b=tr[tr.method==ref].set_index('file').loc[a.index]
                    for metric in METRICS:
                        delta=(b[metric]-a[metric]).to_numpy();rng=np.random.default_rng(42);ci=np.percentile(delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1),[2.5,97.5])
                        paired.append(dict(stage=stage,split=split,reference=ref,metric=metric,gain_percent=100*delta.mean()/b[metric].mean(),ci_low=ci[0],ci_high=ci[1],wins=int((delta>0).sum()),n=len(delta)))
                if stage=='scorefit':
                    byre=tr.groupby(['method','group'])[METRICS].mean()
                    for group in a.group.unique():
                        regroups.append(dict(split=split,Re=group,**{k+'_gain_percent':100*(1-byre.loc[(method,group),k]/byre.loc[('smooth',group),k]) for k in METRICS}))
            for path in sorted((folder/split).glob('audit_*.pt')):
                d=load(path);q=d['samples']
                for k in (3,9):
                    if f'necc_rank{k}' in q and f'secc{k}' in q:secc_identity.append(dict(stage=stage,split=split,window=path.stem,width=k,max_error=float((q[f'necc_rank{k}']-q[f'secc{k}']).abs().max())))
    stages=pd.DataFrame(stage_rows);stages.to_csv(OUT/'stage_gains.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'paired.csv',index=False);pd.DataFrame(regroups).to_csv(OUT/'by_re.csv',index=False)
    # Same marginal calibration for each final joint method.
    calrows=[];replay_error=0.
    for split in ('development','late','future'):
        d=pd.read_csv(OUT/'common_calibration'/split/'summary.csv',index_col=[0,1]);reference=pd.read_csv(STAGES['scorefit']/split/'summary.csv',index_col=0)
        for mode in ('raw','fixed','online'):
            for method in ('lowcost','secc3'):
                a=d.loc[(mode,method)];b=d.loc[(mode,'smooth')]
                calrows.append(dict(split=split,mode=mode,method=method,**{k+'_gain_percent':100*(1-a[k]/b[k]) for k in ['marginal_crps']+METRICS}))
        for method in ('ecc32','smooth','secc3','necc3','lowcost'):
            old='ar__mixed' if method=='lowcost' else method
            replay_error=max(replay_error,float((d.loc[('raw',method),METRICS]-reference.loc[old,METRICS]).abs().max()))
    assert replay_error<1e-6
    pd.DataFrame(calrows).to_csv(OUT/'calibration_gains.csv',index=False)
    timing=pd.read_csv(STAGES['scorefit']/'timing.csv');refms=float(timing[timing.method=='smooth'].median_ms.iloc[0]);fastms=float(timing[timing.method=='ar__mixed'].median_ms.iloc[0]);speedup=refms/fastms
    focus=stages[stages.stage=='scorefit'][['split','quality_gate']+[k+'_gain_percent' for k in METRICS]]
    # Record all statistical claims as exploratory trajectory summaries.
    fig,axs=plt.subplots(1,3,figsize=(15,4));q=stages[stages.split=='development']
    axs[0].bar(q.stage,q.functional_crps_gain_percent);axs[0].axhline(0,color='k',lw=.6);axs[0].set_ylabel('Functional CRPS reduction vs smooth (%)');axs[0].tick_params(axis='x',rotation=20)
    for method in ['ecc32','smooth','secc3','ar__mixed']:
        tm=timing[timing.method==method].median_ms.iloc[0];s=pd.read_csv(STAGES['scorefit']/'development/summary.csv',index_col=0);ratio=s.loc[method,'functional_crps']/s.loc['smooth','functional_crps'];axs[1].scatter(tm,ratio);axs[1].annotate(method,(tm,ratio),xytext=(4,3),textcoords='offset points',fontsize=8)
    axs[1].set_xlabel('End-to-end median ms');axs[1].set_ylabel('Functional CRPS / smooth');axs[1].axhline(1,color='k',lw=.6)
    f=focus[focus.split!='calibration']
    for name in ['crps_speed_max','crps_mvpe','crps_enstrophy_grid_peak']:axs[2].plot(f.split,f[name+'_gain_percent'],'o-',label=name)
    axs[2].axhline(0,color='k',lw=.6);axs[2].set_ylabel('Task CRPS reduction vs smooth (%)');axs[2].legend(fontsize=7)
    fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/quality_cost.png',dpi=170);fig.savefig(OUT/'figures/quality_cost.pdf');plt.close(fig)
    arfit=json.loads((STAGES['scorefit']/'fit.json').read_text());rho=arfit['rho'];detailed=pd.DataFrame(paired);detailed=detailed[(detailed.stage=='scorefit')&(detailed.reference=='smooth')]
    text=f'''# RealPDE 低成本联合分布：资源取舍成立，尚非各任务等质量替代

本阶段固定当前四步DiPred、XPF逐点边际和32个场样本，只改变联合依赖。最终小型候选从训练真实残差学习四个时空/双分量相关系数，并混入25%方差的PCA32高斯分量。无需额外扩散路径，端到端中位时间从smooth的{refms:.2f}ms降至{fastms:.2f}ms（约{speedup:.2f}倍）。**这只证明实际加速，不能单独证明等质量**。

三个核心宏指标的预定2%门槛及全部任务结果如下。正数为降低评分，负数为退步。最大速度等任务有明显代价；不能用泛函平均掩盖，也不改选别的指标作为新的主要用途。更晚时段仍来自同12开发轨迹，不能视为新鲜独立确认。

{table(focus)}

## 方法与数据

只用原RealPDE：协方差训练819窗口/46轨迹，校准90窗口/10轨迹，开发108窗口/12轨迹，late120与future72为同12轨迹的更晚时间。沿用互斥轨迹/Re划分、旧训练归一化，单seed42。32个样本的噪声不是32个训练种子。

方法发展顺序全部保留，均在读取本目录开发结果之前完成：

1. 原始残差逐点经验高斯化，PCA32、X/P/F_last/F_mid0近邻条件因子、全局/4×8局部高斯权重、Schaake重排，共14个低成本候选。全部未通过校准宏质量门槛。
2. 各训练窗口先按其冻结XPF边际反解高斯分位坐标，去除条件尺度/形状差异，再估计相同因子。14候选仍未通过。
3. 换成全维可分离AR(1)创新：x/y/time/uv四个相关系数，由训练条件分数相邻相关估计；固定系数或F64维ridge读出，另试25%PCA混合，共4候选。仍未通过。
4. 保持同样简单的AR+PCA结构，在训练集每轨迹两个固定位置（共92窗口）按联合评分拟合空间/时间相关尺度。24个固定网格配置只用训练评分选择；校准只作门槛检查。选中rho={rho}，PCA方差比例固定.25。该候选不需要条件F读出，不能包装为“F条件生成的独有收益”。

{table(stages[['stage','split','method','quality_gate','es_gain_percent','vs_gain_percent','functional_crps_gain_percent']])}

线性高斯生成器按解析方差归一化，Phi映射的连续边际均匀；最终按其秩排列完全相同的32个边际节点。有限样本是固定离散分位节点，不称为独立连续均匀抽样；全部CRPS采用经验分布定义，未套fair修正。

## 强对照和物理检查

包括原生32路径ECC、既有smooth、S-ECC、N-ECC。依据[Trotta论文](https://journals.ametsoc.org/view/journals/hydr/25/8/JHM-D-23-0188.1.xml)与[作者实现](https://github.com/btrotta-bom/necc-paper-code/blob/master/ecc.py)，N-ECC会同时重排邻域和成员并平均，改变逐点边际。这里额外作逐点重排恢复原边际；在无平局情况下，它回到S-ECC的点内成员顺序，因此两者几乎相同。不是两个独立新方法。非周期边界使用完整覆盖的截断邻域，详见协议和独立NumPy审计。

N-ECC原始版本单独列出，不能将其边际改变混为纯联合收益。它在本有符号流场上产生明显误差，不构成对原降水方法的普遍否定。S-ECC3是本阶段更强的廉价后处理对照，但仍需32条扩散路径。

物理泛函使用dx=+.003421664、dy=-.003421664；重新计算最终场的TKE、动能、物理涡量平方峰值、最大速度、相对共同P的MVPE及散度功率。旧键名`crps_enstrophy_grid_peak`在本目录实为按上述物理坐标的涡量指标，不是图像梯度代理。降低散度不作为真实误差保证。

轨迹级配对区间（绝对评分差的95%描述性bootstrap区间；按整条轨迹重采样，不将像素当独立样本）：

{table(detailed[['split','metric','gain_percent','ci_low','ci_high','wins','n']])}

## 校准流程一致性

最终候选、ECC、smooth、S-ECC和原版N-ECC都分别评估raw、fixed、online。共同32节点的分量/分位偏移仅在校准集拟合，online在当前完整20帧真值到达后以增益.5更新；下一窗口才使用。每个split/轨迹从同初始偏移冷启动，遗漏时间窗口不产生虚构反馈。所有输出先形成后评分/更新，完整事件日志保留。

{table(pd.DataFrame(calrows))}

相同边际对照在每种校准模式下有序样本最大差为0。原版N-ECC为明确例外。此控制是经验校准，不声称条件共形覆盖保证。

## 实际成本与限制

{table(timing)}

A100、batch1、预载模型、15次计时，包括完整四步主干、49风险头、采样、排序与耦合，不含磁盘/评分。统计实际torch分配峰值；最终AR+PCA推理已导出紧凑依赖，只需PCA32与少量系数，不再常驻完整训练残差库。Schaake近邻控制仍需残差库，其内存代价另列。所有候选均实测，未以主干调用数冒充总延迟。

![质量与成本](figures/quality_cost.png)

## 审计和结论边界

全部固定边际方法的有序节点逐位一致，Gaussian方差、N-ECC NumPy重排、AR独立递推和CDF边际检查通过；数值ES/VS/泛函CRPS独立重算及直接预测重放见对应audit文件。共同校准raw重放最大评分差为{replay_error:.3g}。

当前证据支持：低成本联合模型可以接近宏平均评分并明显降低实际推理成本，但单项任务质量有明显退步，不应替换所有场景的ECC/smooth。前三类“直接协方差拟合”是明确负结果；直接联合评分拟合更有希望。这里都是已有统计机制的可复核探索，未建立新颖算法主张，也没有证明整条低成本联合分布路线错误。

完整六路线目标仍active：在线有符号状态、主动观测、预期计算收益分配、经验物理任务误差估计、结构修正强对照/独立验证仍需继续。旧主干和默认推理未替换，未向外部仓库发布。
'''
    (OUT/'REPORT.md').write_text(text)
    dump(OUT/'audit.json',dict(passed=True,marginals=marginal_audit,secc_necc_restoration=secc_identity,calibration_raw_replay_max_score_error=replay_error,bootstrap_unit='trajectory, reused-data exploratory only',source_hashes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}))
    dump(OUT/'progress.json',dict(status='active',classification='progress',stage='four simple joint-dependence hypotheses tested',selected='scorefit/ar__mixed',actual_speedup=speedup,scope='near-macro-score cost tradeoff, not uniform task-quality equivalence',remaining=['online signed state','active observation value','expected compute benefit allocation','empirical task error estimation','strong correction controls and independent confirmation']))
    print('REPORT_READY',speedup,OUT/'REPORT.md',flush=True)

if __name__=='__main__':main()
