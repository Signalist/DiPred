"""Build evidence-based Chinese report and standalone scientific figures."""
from experiment import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(df):
    df=df.reset_index() if df.index.name else df
    cols=list(df.columns);lines=['| '+' | '.join(map(str,cols))+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in df.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)

def main():
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    rows=[];paired=[]
    for split in ('calibration','development','late','future'):
        s=pd.read_csv(OUT/'physics'/split/'summary.csv').set_index('method');b=s.loc['ecc32'];a=s.loc['smooth_ecc_b50_i1']
        rows.append(dict(split=split,ES_gain_percent=100*(b.es-a.es)/b.es,VS_gain_percent=100*(b.vs-a.vs)/b.vs,functional_CRPS_gain_percent=100*(b.functional_crps-a.functional_crps)/b.functional_crps,speed_max_CRPS_gain_percent=100*(b.crps_speed_max-a.crps_speed_max)/b.crps_speed_max,MVPE_CRPS_gain_percent=100*(b.crps_mvpe-a.crps_mvpe)/b.crps_mvpe))
        if split!='calibration':
            d=pd.read_csv(OUT/'physics'/split/'paired.csv');d=d[(d.method=='smooth_ecc_b50_i1')&(d.baseline=='ecc32')];d.insert(0,'split',split);paired.append(d)
    gains=pd.DataFrame(rows);gains.to_csv(OUT/'positive_gains.csv',index=False);pair=pd.concat(paired);pair.to_csv(OUT/'positive_paired.csv',index=False)
    fields=['ES_gain_percent','VS_gain_percent','functional_CRPS_gain_percent','speed_max_CRPS_gain_percent'];labels=['Energy score','Variogram score','Physical functional CRPS','Maximum-speed CRPS']
    fig,axes=plt.subplots(1,4,figsize=(13,3.4));g=gains[gains.split!='calibration'];colors=['#4477AA','#228833','#AA3377']
    for ax,key,label in zip(axes,fields,labels):
        ax.bar(g.split,g[key],color=colors);ax.axhline(0,color='black',lw=.7);ax.set_title(label);ax.set_ylabel('Improvement vs ECC32 (%)');ax.tick_params(axis='x',rotation=20)
    fig.suptitle('Fixed spatial correlation filter; identical XPF marginals; seed 42',fontsize=11);fig.tight_layout();fig.savefig(figdir/'gains.png',dpi=180);fig.savefig(figdir/'gains.pdf');plt.close(fig)
    metric_rows=[]
    for split in ('development','late','future'):
        s=pd.read_csv(OUT/'physics'/split/'summary.csv').set_index('method');b=s.loc['ecc32'];a=s.loc['smooth_ecc_b50_i1']
        for name in FUNCTIONS:metric_rows.append(dict(split=split,functional=name,gain=100*(b['crps_'+name]-a['crps_'+name])/b['crps_'+name]))
    fd=pd.DataFrame(metric_rows).pivot(index='functional',columns='split',values='gain').loc[FUNCTIONS,['development','late','future']]
    fig,ax=plt.subplots(figsize=(8,4));fd.plot.bar(ax=ax,color=colors);ax.axhline(0,color='black',lw=.7);ax.set_ylabel('CRPS improvement vs ECC32 (%)');ax.set_xlabel('');ax.tick_params(axis='x',rotation=20);ax.set_title('Benefits and regressions by physical functional');fig.tight_layout();fig.savefig(figdir/'tradeoffs.png',dpi=180);fig.savefig(figdir/'tradeoffs.pdf');plt.close(fig)
    phases=[]
    for stage in ('sensitivity','localized','physics','advection','helmholtz_filter'):
        selection=json.loads((OUT/stage/'selection.json').read_text());s=pd.read_csv(OUT/stage/'development/summary.csv').set_index('method');name=selection['selected'];b=s.loc['ecc32'];a=s.loc[name]
        phases.append(dict(stage=stage,selected=name,ES_gain_percent=100*(b.es-a.es)/b.es,VS_gain_percent=100*(b.vs-a.vs)/b.vs,functional_CRPS_gain_percent=100*(b.functional_crps-a.functional_crps)/b.functional_crps))
    phase=pd.DataFrame(phases);phase.to_csv(OUT/'route_outcomes.csv',index=False)
    aud=json.loads((OUT/'audit.json').read_text());lat=json.loads((OUT/'inference_audit.json').read_text())
    maxerr=max([aud['numerical_max_error']]+[json.loads((OUT/s/'audit.json').read_text())['numerical_max_error'] for s in ('sensitivity','localized','physics','advection','helmholtz_filter')])
    selectedpair=pair[pair.metric.isin(['es','functional_crps','crps_speed_max','crps_mvpe'])][['split','metric','relative_percent','ci_low','ci_high','wins','trajectories']]
    group=[]
    for split in ('development','late','future'):
        d=pd.read_csv(OUT/'physics'/split/'per_re.csv').set_index(['method','group']);gain=(d.loc['ecc32']-d.loc['smooth_ecc_b50_i1'])/d.loc['ecc32']*100
        for re,row in gain.iterrows():group.append(dict(split=split,Re_group=re,speed_gain_percent=row.crps_speed_max,functional_gain_percent=row.functional_crps))
    text=f'''# RealPDE 联合概率探索：已出现可复核收益，物理机制优势仍有限

## 结论

本轮找到了**保持逐点XPF分位分布完全不变、仅修正整场样本相关结构**的正向收益：固定空间滤波后再做秩重排，相对原版XPF49 + 真实32条扩散路径ECC，开发／后期／更晚窗口的物理泛函平均CRPS分别降低 **2.88%／3.43%／3.08%**；最大速度CRPS分别降低 **2.38%／2.97%／2.70%**，三个Re组均改善。ES平均略有改善，12条轨迹在三个时间段均改善；VS平均改善约0.28–0.37%，不是每条轨迹都改善。

这是联合分布结构上的收益，**边际CRPS没有变化**。它满足小规模探索中“出现正向收益”的经验标准，但目前最稳健的是简单空间相关修正，不能当作已完成的ICML级新物理算法。复杂投影/输运没有全面胜过这个控制；没有结论证明六条研究路线全部失败。

## 数据、选择与可比性

- 阅读了用户指定附件和项目已有RealPDE、XPF49、ECC、Warp/Modulate报告及代码。只用原RealPDE，不读5pde。冻结四步DiPred和49个独立双分位头，无新主干训练。本轮随机设置统一seed42；32条场样本的随机噪声编号不代表32个训练种子。
- 训练819窗口/46轨迹用于经验残差库与物理泛函归一化、75%/90%事件阈值；校准90窗口/10轨迹用于预定准则选方法。训练、校准、开发按原manifest的轨迹和Re组互斥。
- 开发108窗口/12轨迹，start=0..160步长20；后期120窗口，start=200..560步长40；更晚72窗口，start=600..800步长40。后两个时间段仍是同12条开发轨迹，不是新轨迹或新Re的独立测试。该开发集历史上已反复用于研究，区间仅用于探索。
- 所有方法固定32个样本。主比较使用相同XPF32个中点分位值、相同点预测、相同32条扩散路径；只改变样本成员排列。原生Diffusion另列，不能把边际校正收益当作新联合模块收益。
- 选择准则预先写入协议：校准集ES、VS和5个标准化物理泛函CRPS相对ECC32的比值平均。空间修正固定β=0.5、一次3×3滤波，之后不在开发/后期/更晚数据上调参。选择文件散列在future_protocol.json中冻结。

## 有收益的方法与数学机制

设Q_j是固定位置/变量/时刻j的32个XPF分位值。ECC初始场为Y，m是32个成员逐点均值，δ=Y−m。令S为独立作用于每个时刻和速度分量的3×3空间均值滤波（非周期边界延拓）：

`Z = m + 0.5 δ + 0.5 Sδ`，`Y'_{{k,j}} = Q_j[rank_j(Z_{{k,j}})]`。

因此每个位置的有序32个样本值保持逐位一致，有限样本边际CRPS、均值、区间宽度也保持不变；改善来自跨点/时间/变量排列的变化。此算子本身没有证明新的物理原创性，适合作为后续方法必须击败的便宜强对照。

正向百分比表示降低评分，负值表示退步：

{table(gains)}

![三个时间段的收益](figures/gains.png)

5个泛函是：空间速度涨落TKE峰值、平均动能、物理涡量平方峰值、最大速度、相对共同P的最大向量误差MVPE。各自CRPS除以训练集该泛函标准差，再等权平均。平均增益较受涡量泛函和高Re轨迹影响，不能只报告平均数。

![各泛函的收益与退步](figures/tradeoffs.png)

## 逐轨迹证据与反例

下表CI是“基线评分−方法评分”的**绝对差**，相对百分比另列。每次重采样整条轨迹，不按像素/窗口伪造独立样本；同Re轨迹相关，不能外推新Re置信度。物理泛函平均CRPS的开发和更晚CI仍跨0；最大速度CRPS的三个时间段CI均为正。

{table(selectedpair)}

三个Re组的最大速度CRPS均改善；泛函平均并非如此：

{table(pd.DataFrame(group))}

退步必须保留：后期TKE CRPS约退步1.25%；MVPE在开发/后期/更晚平均退步约0.06%/0.17%/0.17%。Brier平均变化很小，部分训练事件阈值在开发几乎无正例，不能以其微小变化声称极端风险预测已解决。

## 已检验的路线与失败边界

原始A：训练残差按观测历史匹配并随预测流场搬运，校准选中时变搬运；开发相对经验残差VS改善约0.30%，泛函CRPS退步约0.53%，未胜过ECC。原始A及B前两阶段使用行列正向的curl代理，不能解释为真正物理涡量；下表跨阶段的泛函数值因此不直接横比，只解释各自相同口径的ECC对照。

{table(phase)}

- B：以h=.1中心差分近似Jv，4/8方向，32个高斯系数样本；比较相同计算预算的8/16非线性路径基。半步差分稳定（相对RMSE约千分之一以内），对单位初始噪声的线性化误差约10–20%；主要联合评分仍弱于ECC，不能把采样次数减少当成同质量加速。
- B局部化：空间核长度4/8/16限制低秩虚假长程相关，并对非线性路径用相同修正。部分指标改善，仍未整体超越强对照。
- A物理投影：非周期中央差分Hodge投影后秩重排，强度/迭代数在校准选定；更低散度不等于更好的概率预测。实测PIV平面存在散度，强行压低会使分布错误。
- A+4：按真实dx/dy/dt和冻结P速度搬运相邻时刻扰动，比较纯时间平滑与反向速度。顺流优于反向，但仍不优于固定空间平滑。
- Helmholtz分量：`δ' = δ + β H(S−I)Hδ`只修正旋转分量，对照势流分量与普通平滑。选中β=.5、一次。物理泛函CRPS略优于普通平滑，但VS更差，未形成全面优势；相关结果见helmholtz_filter各split表。其后期数据已用于形成假设，更晚600+窗口才是本轮未参与构思的时间确认。
- 相位—幅值的完整条件分布、学习型随机残差演化和Sim-to-Real尚未被这轮实验穷尽。RealPDE内部Re迁移不能冒充Sim-to-Real。因此不作“所有路线失败”的结论。

## 物理坐标与数值审计

逐一检查训练/校准/开发全部HDF5网格：dx=+0.003421664、dy=−0.003421664，dt≈0.02。行号方向与物理y相反。physics/advection/helmholtz阶段已按正确方向计算涡量与散度；前面明确名为图像网格代理的指标保留原定义。没有把图像梯度代理包装为物理涡量。

旧环境强制`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`，使batch1与旧batch12缓存略不同。初始A用batch2复现旧缓存；之后所有阶段显式override=0，候选和对照一起在FP32重算。任何精度变化都不计作方法收益。

所有耦合方法有序边际最大差为0。NumPy独立双重求和重算CRPS、ES、VS、泛函与Brier，已保存首/末窗口整场样本；最大数值差 **{maxerr:.3g}**。Hodge离散伴随、投影残差和幂等性另存机制审计。独立推理入口重放误差见inference_audit.json，推理只接收已观测历史，无标签或评测缓存依赖。

## 成本与使用

预载模型，A100，batch1，完整四步P + 49风险头 + 32条四步路径 + 耦合，10次计时；没有包含磁盘I/O和评分：

{table(pd.DataFrame(lat['benchmarks']))}

仍需32条扩散路径，没有实现低成本采样突破。毫秒级小差异受测量噪声影响，不能宣称复杂版本更快。

从Workspace加载推理入口：

```python
import sys
sys.path.insert(0, 'joint_structure_20261117')
from inference import JointPredictor
predictor = JointPredictor()
# observed: CUDA tensor [1,2,20,32,64]，物理u/v单位
samples = predictor.predict(observed, method='smooth')  # [32,2,20,32,64]
```

完整复现入口、阶段协议和缓存说明见../../joint_structure_20261117/README.md。输出均在本实验目录，未替换旧发布权重或推送外部仓库。

## 文献边界

ECC的“固定边际后按原ensemble秩重排”是已有方法，本轮将其作为强对照，而非新贡献：[Schefzik等，2013](https://arxiv.org/abs/1302.7149)。附件所提DLL在输入条件化低秩函数基上训练系数扩散：[论文](https://arxiv.org/abs/2602.04139)、[作者实现](https://github.com/sungwpark/dll-no)。本轮没有训练或复现DLL，不能宣称优于DLL；低秩形式或Gaussian系数也不能单独作为创新依据。

可继续推进的研究问题是：学习观测条件下、与PDE误差动力学相符的联合依赖算子，在保持边际能力时稳定击败这里的空间修正强对照，并修复MVPE/TKE取舍。本轮的贡献是可复核正向可行性证据和一批明确的反例，尚非完成论文。
'''
    (OUT/'REPORT.md').write_text(text)
    print('REPORT',OUT/'REPORT.md',flush=True)

if __name__=='__main__':main()
