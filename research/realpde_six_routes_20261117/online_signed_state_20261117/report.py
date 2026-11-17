"""Stage-one/two point-forecast evidence, including all negative results."""
from state import *

def table(df):
    def text(v):return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(text(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def main():
    allrows=[];base_errors={};geo=OUT/'geometric'
    for split in ('calibration','development','late','future'):
        lin=pd.read_csv(OUT/'linear_geometry'/f'{split}_direct_windows.csv');lin['method']=lin['method'].str.replace('geo_','lin_',regex=False)
        parts=[pd.read_csv(OUT/f'{split}_windows.csv'),pd.read_csv(OUT/f'{split}_controls.csv'),pd.read_csv(geo/f'{split}_direct_windows.csv'),lin];d=pd.concat(parts,ignore_index=True);assert np.isfinite(d.mse).all() and (d.mse>=0).all();d['split']=split;allrows.append(d)
        means=d[d.scenario=='natural'].groupby(['method','file','start']).mse.mean().unstack(0);base_errors[split]=float((means['geo_zero']-means['base']).abs().max());assert base_errors[split]<1e-8
    df=pd.concat(allrows,ignore_index=True);df.to_csv(OUT/'all_windows.csv',index=False);tr=df.groupby(['split','scenario','method','file','group']).mse.mean().reset_index();tr.to_csv(OUT/'all_trajectories.csv',index=False)
    natural=tr[tr.scenario=='natural'];avg=natural.groupby(['split','method']).mse.mean().unstack(0);gains=100*(1-avg/avg.loc['base']);gains.to_csv(OUT/'natural_gains.csv');paired=[]
    for split in ('development','late','future'):
        d=natural[natural.split==split].pivot(index='file',columns='method',values='mse')
        for method in ('rw_kalman','geo_rolling','lin_rolling','memory_gated'):
            for control in ('base','local_warp','memory_gated'):
                if method==control:continue
                diff=(d[control]-d[method]).to_numpy();rng=np.random.default_rng(42);boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1);paired.append(dict(split=split,method=method,control=control,gain_percent=100*diff.mean()/d[control].mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),wins=int((diff>0).sum()),trajectories=len(diff)))
    paired=pd.DataFrame(paired);paired.to_csv(OUT/'paired.csv',index=False)
    bygroup=natural.groupby(['split','group','method']).mse.mean().unstack('method');bygroup=(1-bygroup.div(bygroup['base'],axis=0))*100;bygroup.to_csv(OUT/'by_re.csv')
    drift=df[(df.scenario!='natural')&(df.age>=0)].groupby(['split','scenario','method','file']).mse.mean().groupby(['split','scenario','method']).mean().unstack('method');drift=(1-drift.div(drift['base'],axis=0))*100;drift.to_csv(OUT/'drift_postchange_gains.csv')
    curve=df[df.scenario!='natural'].groupby(['split','scenario','method','age']).mse.mean().reset_index();curve.to_csv(OUT/'drift_recovery_curves.csv',index=False)
    linear_roll=pd.read_csv(OUT/'linear_geometry/rollout_windows.csv');linear_roll['method']=linear_roll['method'].str.replace('geo_','lin_',regex=False)
    roll=pd.concat([pd.read_csv(OUT/'rollout_windows.csv'),pd.read_csv(geo/'rollout_windows.csv'),linear_roll],ignore_index=True);assert np.isfinite(roll.mse).all();roll.to_csv(OUT/'all_rollout_windows.csv',index=False);rm=roll.groupby(['mode','method','file']).mse.mean().groupby(['mode','method']).mean().unstack('mode');rg=100*(1-rm/rm.loc['base']);rg.to_csv(OUT/'rollout_gains.csv')
    # Mechanism diagnostic: signed coordinate persistence on genuinely adjacent
    # training blocks only; pooled correlations are descriptive, not causality.
    td=load(OUT/'train.pt');gd=load(SOURCE/'train_oracle.pt');pairs=[(a,b) for ix in group_ids(td['meta']) for a,b in zip(ix[:-1],ix[1:]) if td['meta'][b]['start']-td['meta'][a]['start']==20];first=[a for a,b in pairs];second=[b for a,b in pairs];corr=[]
    for family,c in [('fixed_pod',td['c']),('geometry',gd['coefficients'].numpy())]:
        for k in range(c.shape[1]):corr.append(dict(family=family,coordinate=k,lag1_correlation=float(np.corrcoef(c[first,k],c[second,k])[0,1])))
    pd.DataFrame(corr).to_csv(OUT/'coordinate_persistence.csv',index=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,4.5));shown=['rw_kalman','conditional_kalman','rnn','geo_rolling','lin_rolling','memory_gated','local_warp'];gains.loc[shown,['development','late','future']].plot.bar(ax=axes[0]);axes[0].axhline(0,color='black',lw=.8);axes[0].set_ylabel('Observed-history MSE reduction (%)');axes[0].set_xlabel('');axes[0].tick_params(axis='x',labelrotation=35)
    rg.loc[['rw_kalman','conditional_kalman','geo_rolling','lin_rolling']].plot.bar(ax=axes[1]);axes[1].axhline(0,color='black',lw=.8);axes[1].set_ylabel('200-frame free-roll MSE reduction (%)');axes[1].set_xlabel('');axes[1].tick_params(axis='x',labelrotation=35);fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/state_evidence.png',dpi=170);fig.savefig(OUT/'figures/state_evidence.pdf');plt.close(fig)
    fit=json.loads((OUT/'fit.json').read_text());selection=json.loads((OUT/'selection.json').read_text());gs=json.loads((geo/'selection.json').read_text());timing=pd.read_csv(geo/'timing.csv');audit=json.loads((OUT/'audit.json').read_text());gaudit=json.loads((geo/'inference_audit.json').read_text());assert audit['passed'] and gaudit['passed'];models=[]
    for name,m in selection['models'].items():models.append(dict(method=name,configuration=str(m['config']),calibration_mse=m['mse']))
    for name,m in gs['models'].items():models.append(dict(method='geo_'+name,configuration=str(m['config']),calibration_mse=m['mse']))
    ls=json.loads((OUT/'linear_geometry/selection.json').read_text())
    for name,m in ls['models'].items():models.append(dict(method='lin_'+name,configuration=str(m['config']),calibration_mse=m['mse']))
    selection_table=pd.DataFrame(models);selection_table.to_csv(OUT/'all_selected_models.csv',index=False)
    ls=json.loads((OUT/'linear_geometry/selection.json').read_text());laudit=json.loads((OUT/'linear_geometry/inference_audit.json').read_text());assert laudit['passed'];lt=pd.read_csv(OUT/'linear_geometry/timing.csv');lt['identifier']='linear';timing['identifier']='nonlinear80';timing=pd.concat([timing,lt],ignore_index=True)
    chosen=paired[paired.method.isin(['rw_kalman','geo_rolling','lin_rolling'])]
    report=f'''# 在线有符号状态：固定POD负结果，几何记忆有小幅收益

当前自然20→20任务证据：**8维固定残差模态上的卡尔曼状态未带来稳定改善；7维几何系数的简单滚动记忆降低MSE均值，但未超过已有局部修正器，多数轨迹级区间仍跨零。** 学到复杂状态转移和使用卡尔曼滤波本身尚无优势。后续已完成共同概率校准、块任务时序区间适配对照及几何故障恢复；完整六方向目标仍active，不能据在线路线阶段报告宣布全部完成。

## 后续概率、时序对照与几何故障结果

- [概率扩展](probability/REPORT.md)：冻结几何状态平移在raw模式下ES改善1.89/1.57/1.27%，轨迹CI均高于零；fixed/online同向。五任务宏CRPS收益较小且多数CI跨零。成员warp改善最大速度，却使宏CRPS退步，未据此换主指标。
- [时序区间对照](temporal_intervals/REPORT.md)：同12轨迹连续41块、共同预热10块后评价372窗口。SPCI是已说明的固定点预测器/叶权QRF适配；AcMCP是官方递推的块h=1 Python翻译，不是逐帧h=20或作者包逐字节复现。field_online五任务90%区间宏覆盖69.14%；AcMCP块h1为90.27%，但宽度约2.49倍；无scorecaster的PID对照更好。在线几何修正对同一标量控制器的额外收益较小。
- [受控几何故障](geometric_drift/REPORT.md)：原自然校准参数冻结，线性几何滚动的阶跃/渐变后MSE改善15.22/13.81%，12/12轨迹改善。只是预测端warp故障，真实X/Y及PDE参数未变。实际stream重放与未来反馈干预通过。

这些扩展都保留全部负结果、逐任务/轨迹/工况表及独立数值审计。后来时间窗和连续反馈实验仍属于已用轨迹，未获得新的独立确认。

## 已完成的机制检验

1. 固定8维物理残差坐标：两个全局u/v偏置，六个与其正交的训练POD模态。按物理场平均内积正交；训练819窗口/46轨迹，cal90/10，dev108/12，late120与future72沿用同12轨迹。POD使用未来残差仅用于训练/已到达反馈，当前真值不进入issue。受限oracle能解释约13–18%误差，但不代表可预测收益。
2. 随当前P变化的7维几何坐标：网格x/y位移、帧相位、u/v幅值与归一化偏置。完整旧目标到达后，执行有界80步配准识别历史系数；下一预测才使用。位移单位是网格/帧，不是辨识出的物理参数。滚动4块均值乘.5由自然cal选择。随机游走Kalman另列，不在查看dev后换主候选。
3. 便宜的线性切向识别：在同7个方向解正则7×7方程，仍应用同一有界非线性warp。独立使用自然cal的同一小网格，选择滚动4块均值乘.5。`lin_`方法的收益略小，但反馈识别成本大幅降低；全量直接stream、数值重放和自由滚动均保留。

全部单seed42。固定POD模型包含训练VAR、随机游走KF、VAR-KF、X/P/F16PC条件均值+KF、直接线性AR和{fit['rnn_parameters']}参数GRU16。RNN与VAR由校准选择零修正，未把这一结果隐藏。模型参数、全部网格、训练曲线及所有阶段协议保留。

## 自然观察驱动结果

正数表示相对共同原预测降低物理MSE；每轨迹先平均。几何方法使用真正逐窗口的主干预测和已揭示目标配准全量重放，不使用旧批量缓存的数值结果作为最终性能。

{table(gains.reset_index()[['method','calibration','development','late','future']])}

主要冻结候选对基线与强对照的轨迹级95%描述性bootstrap区间，CI是基线减本方法的绝对物理MSE差：

{table(chosen[['split','method','control','gain_percent','ci_low','ci_high','wins']])}

现有full-field residual-subspace memory按原实现纳入：unit/gated分别只在cal选择外部收缩。对于不相邻发行窗口，明确传入上一块已经揭示的目标，而不是把当前历史冒充上一预测的真值。local_warp是先前已冻结的2967参数F_last局部修正器。所有控制保留，不能用弱的滚动偏置替代这些控制。

![观察驱动与自由滚动](figures/state_evidence.png)

## 漂移与自由滚动是另外的诊断

stage one在每轨迹中点给P注入物理u/v偏置：幅度是训练分量残差RMS的一半，正负按轨迹编号固定交替；另有线性ramp。历史X不改，P相关输入按真实偏置更新，F仍来自原冻结主干。它测试预测器输出故障的恢复，不是未知PDE参数漂移。模型仍用自然cal固定的参数；报告整个变化后阶段和逐窗口曲线，没有事后挑恢复阈值。后续几何模型另按冻结协议完成step/ramp几何故障扩展，详见geometric_drift/REPORT.md；两种注入不是同一种实验。

{table(drift.reset_index()[['split','scenario','rolling_bias','rw_kalman','conditional_kalman','memory_gated','local_warp']])}

200帧自由滚动中只读取自己的预测历史。cold只观察初始20帧；warm先合法揭示3个完整预测块，再冻结真实反馈。每步显式skip_feedback，不从自产预测制造“误差标签”。同模式与同时间段的原预测作比较：

{table(rg.reset_index())}

观察驱动的小幅收益不能据此称为长期动力学改进。完整逐块误差见all_rollout_windows.csv。

## 实际在线识别成本和审计

{table(timing[['identifier','method','median_predict_ms','median_reveal_identification_ms','n','max_mse_replay_error']])}

时间来自预载模型/A100/单轨迹逐块调用；预测与真值到达后的80步配准或线性识别分别计时，识别并非免费。这里是接口诊断计时，样本少于专门性能基准，不用于总体加速主张。

固定POD显式物理场重构MSE与低维公式最大误差{audit['physical_mse_max_error']:.3g}；所有状态通过未来干预不改变已发前缀、轨迹隔离、过早反馈拒绝及skip不创建观测检查。其直接预测与缓存场最大差小于3e-5。几何接口全轨迹真实配准重放见geometric/inference_audit.json；其实际stream和缓存原主干MSE最大差为{max(base_errors.values()):.3g}。

数值审计曾发现几何批量缓存与单窗口识别的MSE误差略超1e-8阈值，旧失败审计保留在geometric/batch_cache_replay_diagnostic.json。因此完整cal/dev/late/future均改用实际单窗口主干与配准重放；没有放宽阈值，也没有据新结果重新选择模型。原始回放仍留存作对照。训练缓存360→400的间隔也已修复，旧权重/选择存于initial_gap_mismatch，详见IMPLEMENTATION_NOTES.md。

## 适用边界与后续要求

- 当前块反馈协议下已完成raw/fixed/online与SPCI/AcMCP适配对照；尚无作者包端到端或逐帧h=20验证，也不从块h1结果推断其MA/跨horizon机制。此处的任务定义边界已明确，不把它隐藏为完整多步复现。
- 新轨迹/新工况确认，以及物理参数监督下的可识别性仍未获得；当前不能把状态称为物理参数。
- 当前强对照是已有memory/localwarp及共同标量时序控制器，未探索静态局部修正与在线几何的组合；若以后组合，须另冻协议，不能把现有结果冒充组合模型胜出。

这份报告提供两个可复核的机制观察：本次小模型未能稳定利用固定全局POD状态；跟随预测几何的历史坐标带来小幅均值改善，但简单平均即可达到当前收益，尚未证明学习动力学转移/Kalman优于强基线。它不证明整条在线辨识路线错误。
'''
    (OUT/'REPORT.md').write_text(report);dump(OUT/'provenance.json',dict(source_files={str(p.relative_to(W)):digest(p) for p in (W/'online_signed_state_20261117').glob('*.py')},protocols={p.name:digest(p) for p in (W/'online_signed_state_20261117').glob('*PROTOCOL.md')},main_model_sha256=digest(OUT/'models.pt'),geometric_model_sha256=digest(geo/'model.pt'),linear_geometric_model_sha256=digest(OUT/'linear_geometry/model.pt'),shared_memory_sha256=digest(W.parent/'Shared/online_residual_subspace.py'),direct_vs_cache_base_mse_max_error=base_errors,route_complete=False,full_goal_complete=False))
    print(gains.loc[['rw_kalman','geo_rolling','geo_rw','memory_gated','local_warp']].to_string(),flush=True);print('FREE',rg.to_string(),flush=True)

if __name__=='__main__':main()
