"""Generate the partial-goal report from saved measurements, not hand-entered gains."""
from structure import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(df):
    def fmt(v):return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)
    rows=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    rows += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def main():
    local=OUT/'local';selection=json.loads((local/'selection.json').read_text());primary=selection['selected']['warp'];add=selection['selected']['additive']
    paired=[];re=[];summaries=[];learning=[]
    for split in ('development','late','future'):
        d=pd.read_csv(local/f'{split}_trajectories.csv');a=d.pivot(index=['file','group'],columns='method',values='mse')
        for method in a:
            delta=(a['base']-a[method]).to_numpy();rng=np.random.default_rng(42)
            boots=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1);ci=np.percentile(boots,[2.5,97.5])
            paired.append(dict(split=split,method=method,gain_percent=100*delta.mean()/a['base'].mean(),wins=int((delta>0).sum()),n=len(delta),ci_low=ci[0],ci_high=ci[1]))
            summaries.append(dict(split=split,method=method,mse=float(a[method].mean())))
        for group in a.index.get_level_values('group').unique():
            g=a.xs(group,level='group');re.append(dict(split=split,Re=group,gain_percent=100*(1-g[primary].mean()/g['base'].mean())))
        for kind,label in [('F_last','Fdiff'),('F_det4_6000','Fdet4'),('X_P','XP')]:
            for fraction,suffix in [(25,'_n25'),(50,'_n50'),(100,'')]:
                key=f'warp_{kind}{suffix}';learning.append(dict(split=split,features=label,train_percent=fraction,gain_percent=100*(1-a[key].mean()/a['base'].mean())))
    paired=pd.DataFrame(paired);paired.to_csv(OUT/'paired.csv',index=False);pd.DataFrame(re).to_csv(OUT/'by_re.csv',index=False);pd.DataFrame(learning).to_csv(OUT/'learning_curve.csv',index=False)
    # Report comparative deltas against XP and same-four-call deterministic F.
    comparisons=[]
    for split in ('development','late','future'):
        a=pd.read_csv(local/f'{split}_trajectories.csv').pivot(index='file',columns='method',values='mse')
        for ref in ['warp_X_P','warp_F_det4_1500','warp_F_det4_6000',add]:
            delta=(a[ref]-a[primary]).to_numpy();rng=np.random.default_rng(42);ci=np.percentile(delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1),[2.5,97.5])
            comparisons.append(dict(split=split,reference=ref,gain_percent=100*delta.mean()/a[ref].mean(),wins=int((delta>0).sum()),ci_low=ci[0],ci_high=ci[1]))
    pd.DataFrame(comparisons).to_csv(OUT/'paired_controls.csv',index=False)
    rollout=pd.read_csv(OUT/'rollout_windows.csv');rt=rollout.groupby(['method','mode']).mse.mean().unstack('method')
    rrows=[]
    for mode in rt.index:
        for method in (primary,add):rrows.append(dict(mode=mode,method=method,gain_percent=100*(1-rt.loc[mode,method]/rt.loc[mode,'base']),mse=rt.loc[mode,method]))
    rrows=pd.DataFrame(rrows);rrows.to_csv(OUT/'rollout_gains.csv',index=False)
    base=pd.read_csv(OUT/'calibration_oracle.csv');b=base[base.method=='base'].groupby('file').mse.mean().mean()
    oracle=[]
    for file in ('calibration_oracle.csv','calibration_oracle4.csv','calibration_local_oracle.csv'):
        d=pd.read_csv(OUT/file)
        for method,value in d.groupby(['method','file']).mse.mean().groupby('method').mean().items():oracle.append(dict(method=method,mse=value,gain_percent=100*(1-value/b)))
    oracle=pd.DataFrame(oracle)
    ablations=[]
    for split in ('development','late','future'):
        aa=pd.read_csv(OUT/f'{split}_ablation.csv').groupby(['method','file']).mse.mean().groupby('method').mean()
        for method,v in aa.items():ablations.append(dict(split=split,method=method,gain_percent=100*(1-v/aa['base'])))
    ablations=pd.DataFrame(ablations);ablations.to_csv(OUT/'ablation_summary.csv',index=False)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,3,figsize=(14,3.8))
    keys=[primary,'warp_X_P','warp_F_det4_1500','warp_F_det4_6000',add]
    for key in keys:
        q=paired[paired.method==key];axs[0].plot(q.split,q.gain_percent,'o-',label=key)
    axs[0].set_ylabel('Observed-history MSE reduction (%)');axs[0].legend(fontsize=7);axs[0].axhline(0,color='k',lw=.6)
    lc=pd.DataFrame(learning);lc=lc[lc.split=='development']
    for key,q in lc.groupby('features'):axs[1].plot(q.train_percent,q.gain_percent,'o-',label=key)
    axs[1].set_xlabel('Training trajectories (%)');axs[1].set_ylabel('Development MSE reduction (%)');axs[1].legend();axs[1].axhline(0,color='k',lw=.6)
    rr=rollout[rollout['mode']=='free'].groupby(['method','block']).mse.mean().unstack('method')
    for key in (primary,add):axs[2].plot(rr.index,100*(1-rr[key]/rr['base']),label=key)
    axs[2].set_xlabel('Free-running block (20 frames each)');axs[2].set_ylabel('MSE reduction vs base (%)');axs[2].axhline(0,color='k',lw=.6);axs[2].legend(fontsize=7)
    fig.tight_layout();fig.savefig(figures/'structure_results.png',dpi=170);fig.savefig(figures/'structure_results.pdf');plt.close(fig)
    focus=paired[paired.method.isin(keys)]
    timing=pd.read_csv(OUT/'timing.csv');audit=json.loads((OUT/'mechanism_audit.json').read_text())
    text=f'''# RealPDE 有符号结构修正：阶段结果，完整目标仍在推进

本阶段得到一个小型可复现候选：{primary}（2967参数），直接移动/修正预测未来场。它在校准集从预定候选中选定，之后冻结。此结果仅支持有完整真实历史的20→20预测；无反馈长滚动存在退步。尚未完成附件中的全部六条研究路线。

## 诊断与方法

所有候选使用同一当前发布的四步 DynamicsDiffusion20、FP32、seed42。修正训练819窗口/46轨迹，主干原训练47轨迹前200帧；修正使用这些训练身份的较晚窗口。校准90窗口/10轨迹，开发108窗口/12轨迹，轨迹和Re与训练分离。late120、future72仍是同12开发轨迹，只是时间延伸。所有这些数据有历史研究使用记录，不能称新鲜独立确认。

下面的oracle读取真实未来，属于优化找到的诊断参照，不是可部署预测，也不保证达到非凸参数族的全局最优。位移限制±1网格、相位±1帧、幅值±25%、偏置±0.25归一化单位。局部56参数在2×4网格平滑插值；正向x/y位移是数组索引坐标，不能直接解释为物理流速方向。

{table(oracle)}

局部训练头采用逐位置训练PCA32维、16隐藏通道、三层卷积。结构头2967参数，普通加性头2984参数，均2000步、batch8、同种子和数据顺序；仅用校准选修正强度。加性头输出2×4×2×4平滑残差，结构头输出7×2×4系数。同一P、同一信息时刻。global7另有54个岭回归/模态对照，完整结果见development_summary.csv和selection_grid.csv；不能将多个变体当成独立重复实验。

## 轨迹级结果

正百分比表示降低物理速度MSE。CI为基线MSE减候选MSE的绝对差，按完整轨迹重采样10000次；是单种子、重复使用数据上的描述性区间，同Re轨迹相关，不支持新Re总体置信保证。

{table(focus)}

相对强控制的配对比较：

{table(pd.DataFrame(comparisons))}

主候选逐Re：

{table(pd.DataFrame(re))}

![结果及失败边界](figures/structure_results.png)

## 收益是否只是固定平滑

补充控制也是在首次开发结果后提出：56个常量局部参数只在train上优化，统一3×3平滑及两控制强度只在calibration选择。固定原主候选，不以这些控制重选。主模型分别仅保留几何/幅值/偏置系数作机制消融。分量存在交互，单项消融收益不应相加。

{table(ablations)}

几何部分承担主要收益；固定控制未达到同等效果。这仍不等同于已辨识了真实物理输运速度。

## F控制与样本效率

补训了完全相同输入（含历史增量）、相同网络形状和99→66→33→0四调用图的确定性模型，目标仅为终点真实未来MSE。同seed42、相同主干训练身份及前200帧。1500步近似匹配原6000步去噪训练的网络调用数；6000步匹配更新次数，训练计算量更高，不能称严格等训练成本。Fdet用于同一冻结DiPred的误差修正；不替换P，所以实际使用Fdet还需额外主干计算。

最初已冻结的主候选不变。Fdet-only局部头及25/50%局部学习曲线是看到首次开发结果后补充的机制控制，校准选强度，不重新选择主候选。它们不能伪装为全部预注册。各比例使用嵌套完整训练轨迹；PCA也仅在该子集拟合。X/XP使用相同32维PCA和同小头；不宣称冻结F的1.6M表征计算免费，实际推理计时见下。

25%/50%训练量下三种局部头均由校准选为零修正，因此**没有证明小标注样本效率优势**。确定性1500步F在三个时段的平均MSE均略优于本候选，配对区间跨零；不能宣称扩散目标具有独有优势。复用当前DiPred的F仍可省去另算确定性主干的推理成本。

{table(pd.DataFrame(learning))}

## 长滚动与实际成本

独立任务：12条开发轨迹仅初始20帧是真实观测，连续自回归10个完整20帧块；同时间范围的observed控制每次读取真实历史。没有中途真值修正自由滚动输入。正收益不能从一个任务转移到另一个任务。

{table(rrows)}

{table(timing)}

计时为A100、batch1、预载模型、30次，包括完整主干、特征提取、PCA与场修正，不包括磁盘和评分。小差异受测量噪声影响。Peak是实际torch分配峰值，不是nvidia-smi显存预留。

## 审计与使用

合成已知位移/相位/振幅恢复的最大系数误差为{audit['synthetic_recovery']['max_coefficient_error']:.3g}，独立NumPy物理MSE差为{audit['independent_numpy_mse_max_error']:.3g}。实际观测直接推理与缓存重放通过，见inference_audit.json。真实流场位移—幅值分解未证明唯一，不把参数称为已辨识的物理状态。

从Workspace运行：

```python
import sys
sys.path.insert(0, 'error_structure_20261117')
from inference_rollout import Predictor
predictor = Predictor('{primary}')
future = predictor.predict(observed_standardized)  # [B,2,20,32,64], CUDA
```

归一化使用当前发布checkpoint中的mean/std；此入口不接受未来标签，不依赖评测缓存。

## 全目标剩余工作

1. 结构修正已有观测驱动收益及自由滚动反例；仍需审查同信息ResCrct/更直接残差控制、局部收益机制与扩展独立确认。
2. 低成本联合分布尚未完成；旧smooth仍依赖32条扩散路径。下一步优先从真实残差学习低秩/相关结构，与固定边际ECC、smooth、N-ECC比较，并实测成本。
3. 在线潜状态尚待有符号反馈、漂移与同信息控制实验。
4. 主动观测尚待固定新增观测时刻/预算的实际信息价值实验。
5. 计算分配尚待使用真实修正收益训练门控、相同预算对照和实际成本曲线。
6. 物理证书初步审查：calibration全部HDF5只含u/v/x/y/t/Re/aoa，无压力和出平面速度；本地inform.md亦说明压力零填充且不评分。仅凭这些观测不能构造有已核实前提的完整动量残差/伴随严格误差界。该结论不否定经验任务误差估计，后者仍待实验。

完整目标保持active。原始主线权重未替换，旧目录和外部仓库未发布改动。复现命令见源码目录README.md；所有负结果保留。
'''
    (OUT/'REPORT.md').write_text(text)
    dump(OUT/'progress.json',dict(status='active',classification='progress',completed=['global and local target-assisted diagnostics','same-four-call deterministic representation control','single-seed global readouts and tiny local correction heads','trajectory-level observed-history evaluation and learning curves','target-free inference and actual timing','free-running rollout counterexample'],remaining=['joint low-cost generator with strong controls','online signed-state correction','active observations','compute allocation','empirical physical-task error estimation and certificate conditions','ResCrct comparison and stronger independent confirmation'],primary=primary,source_manifest_sha256=digest(MANIFEST),report_sha256=digest(OUT/'REPORT.md')))
    print('REPORT',OUT/'REPORT.md',flush=True)

if __name__=='__main__':main()
