from task_error import *

def paired(df,method,reference):
    pivot=df.pivot(index='file',columns='model',values='score');a=pivot[method].to_numpy();b=pivot[reference].to_numpy();rng=np.random.default_rng(42);ids=rng.integers(0,len(a),(4000,len(a)));boot=(1-a[ids].mean(1)/b[ids].mean(1))*100
    return dict(gain=(1-a.mean()/b.mean())*100,low=np.quantile(boot,.025),high=np.quantile(boot,.975),wins=int((a<b).sum()),n=len(a))

def run():
    sel=json.loads((OUT/'selection.json').read_text());chosen=sel['chosen'];ci=[];summaries={};bounds=[];alltr=[];cal=load(OUT/'calibration.pt');calpred=load(OUT/'calibration'/'predictions.pt')
    for split in ['calibration','development','late','future']:
        sm=pd.read_csv(OUT/split/'summary.csv',index_col=0);summaries[split]=sm;tr=pd.read_csv(OUT/split/'trajectories.csv');tr['split']=split;alltr.append(tr)
        for method in sel['models']:
            for reference in ['base','mean','direct/X+P/16','task_affine','direct/task_stats/8']:
                ci.append(dict(split=split,method=method,reference=reference,**paired(tr,method,reference)))
        d=load(OUT/f'{split}.pt');bb=load(OUT/split/'bounds.pt');pp=load(OUT/split/'predictions.pt');unit=load(OUT/'models.pt')['unit'].sqrt().cpu()
        for name,error,bound in [('norm_envelope',bb['absolute_error'],bb['empirical_bound']),('direct_task_envelope',bb['absolute_error'],bb['task_envelope']),('selected_task_envelope',(pp[chosen]-d['jy']).abs(),(calpred[chosen]-cal['jy']).abs().amax(0,keepdim=True).expand_as(d['jy']))]:
            ok=error<=bound+1e-12
            for j,task in enumerate(TASKS):
                byfile=pd.DataFrame(dict(file=[m['file'] for m in d['meta']],covered=ok[:,j].numpy(),bound=(bound[:,j]/unit[j]).numpy(),relative=(bound[:,j]/error[:,j].clamp_min(1e-12)).numpy())).groupby('file').agg({'covered':'mean','bound':'mean','relative':'median'})
                bounds.append(dict(split=split,method=name,task=task,coverage=byfile.covered.mean(),normalized_radius=byfile.bound.mean(),median_bound_error_ratio=byfile.relative.median()))
            simultaneous=pd.DataFrame(dict(file=[m['file'] for m in d['meta']],ok=ok.all(1).numpy())).groupby('file').ok.agg(['mean','all'])
            bounds.append(dict(split=split,method=name,task='joint',coverage=simultaneous['mean'].mean(),whole_trajectory_fraction=simultaneous['all'].mean()))
    ci=pd.DataFrame(ci);ci.to_csv(OUT/'paired_gains.csv',index=False);bdf=pd.DataFrame(bounds);bdf.to_csv(OUT/'empirical_bounds.csv',index=False);tr=pd.concat(alltr);tr.to_csv(OUT/'all_trajectories.csv',index=False);tr.groupby(['split','model','group']).score.mean().to_csv(OUT/'by_re.csv')
    modelnames=['base','mean','task_persistence','task_affine','direct/task_stats/8','direct/P/16','direct/X+P/16','direct/F_last/16','direct/F_det4_1500/16','direct/F_det4_6000/16','direct/X+P+F_last/16','direct/X+P+F_last/32','direct/kinematic/16','direct/task_weighted/16','taylor/X+P+F_last/16','taylor_no_q/X+P+F_last/16','pod/X+P+F_last/16','pod_q/X+P+F_last/16']
    lines=['# 物理任务误差：小型有符号读出与精确离散分解','',f'校准集冻结选择为 **{chosen}**。任务仍是完整历史20→20，无新增观测。输出是4个任务量的修正估计；原预测场P没有改变，不能把这些收益写成整场MSE或自由滚动改善。','',
        '主指标为四项任务平方误差除以各自训练基线MSE后取平均，再按轨迹平均。任务固定为窗口平均平面动能、时间脉动能、中心差分平均涡量平方的一半、右侧1/4区域平均u。最后一项是运输相关代理，不是升阻力或完整通量。全部物理量在数据存储坐标/速度单位下计算，没有擅自标记SI单位。','',
        '**主要取舍：** 原选择模型的宏指标改善59.02/63.84/45.13%，收益集中于脉动能（88–92%）和涡量统计（95–97%）；区域平均u反而退步15.38/1.83/16.46%。不能宣称四任务一致改善。简单X/P读出也有接近收益，扩散F没有独有优势。','',
        '## 所有冻结候选','', '| 方法 | 读出系数 | cal主分数 | dev改善% | late改善% | future改善% |','|---|---:|---:|---:|---:|---:|']
    for name in modelnames:
        vals=[summaries[s].loc[name,'gain_percent'] for s in ['development','late','future']]
        lines.append(f"| {name} | {sel['readout_parameters'][name]} | {summaries['calibration'].loc[name,'score']:.4f} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} |")
    lines+=['','系数数目不包含训练拟合的PCA投影和POD基底，完整模型体积另存。所有直接16PC模型有68个读出系数。32PC直接对照132个；Taylor分解119个；POD272个，另加3个未解析二次项常数。确定性F控制在原P以外需要额外四次主干调用，耗时不能视为免费。','',
        'task_persistence、task_affine和direct/task_stats/8是在初步结果诊断后补充的强对照，过程明确记录在SIMPLE_CONTROL_ADDENDUM.md，原选择模型和原始模型文件保留。前者直接用J(X)；12系数模型为每个任务只用自身J(X),J(P)；36系数模型联合读取全部8个任务统计。不允许根据它们的开发成绩改写原选择。','',
        '## 选择模型的配对轨迹比较','', '| 时间段 | 对照 | 改善% | 95%描述性CI | 胜出轨迹 |','|---|---|---:|---|---:|']
    for r in ci[(ci.method==chosen)&(ci.split!='calibration')].to_dict('records'):lines.append(f"| {r['split']} | {r['reference']} | {r['gain']:.2f} | [{r['low']:.2f}, {r['high']:.2f}] | {r['wins']}/{r['n']} |")
    lines+=['','## 分任务结果','', '| 时间段 | 任务 | 原P MSE | 选择模型MSE | 相对改善% |','|---|---|---:|---:|---:|']
    for split in ['development','late','future']:
        for task in TASKS:
            a=summaries[split].loc['base',task+'_mse'];b=summaries[split].loc[chosen,task+'_mse'];lines.append(f'| {split} | {task} | {a:.6g} | {b:.6g} | {(1-b/a)*100:.2f} |')
    lines+=['','## 机制与物理边界','',
        '对前三项二次任务，离散恒等式为J(P+R)−J(P)=L(P,R)+Q(R)；区域平均u是线性的。这里的一阶敏感项是任务导数，不是由闭合PDE解出的伴随。单独忽略二阶项、只保留低秩场残差、增加训练平均未解析残差能量的结果均保留。Taylor两项使用同一线性读出时，在Q未触发非负截断处与直接读出代数等价，不能把相同评分包装为新的机制收益。','',
        '运动学特征包含散度和物质加速度，以及三个可见扰动方向的任务加权投影。对3D流动的二维测量，它们不是应当严格为零的闭合Navier–Stokes残差；不能用其大小直接声称预测误差上界。两组物理描述符与P/F采用同16PC读出比较。运动学描述符还包含J(X),J(P)，因此它的收益不能全部归因于物理缺陷，新增12/36系数任务统计对照用于排除这一解释。','',
        '非线性任务J(条件均值场)一般不等于条件任务均值；二阶能量项可以解释为什么点场较平滑却仍能另行预测能量统计。该分解不赋予预测任务量一个唯一对应的未来场。未额外裁剪最终任务值；选择模型在三个评价段没有负能量，P-only在late有1个负能量，丢掉Q的诊断有1/2/2个，保留为失效记录。','',
        '元数据核对覆盖train/cal/dev68轨迹：使用实际x增量和负y增量，t间隔约0.02。真实HDF5没有压力、出平面速度、闭合边界数据或离散误差常数。官方数据卡与本地interface也说明压力没有测量；这只限制本实验能建立的证书，不证明物理证书路线普遍不可能。[官方数据卡](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data/blob/main/README.md)，[官方物理场景](https://realpdecompetition.github.io/#data)。当前manifest已经排除数据卡指出的重复7575_0文件。','',
        '## 确定性不等式与经验包络','',
        '已验证Cauchy–Schwarz给出的二次任务界sqrt(2J(P)) rho + rho²/2；rho是相应残差/中心化残差/涡量残差的RMS范数。区域平均u由区域残差RMS界定。实际rho依赖未知Y，所以上述恒等式/不等式本身不是可部署误差证书。','',
        '可部署诊断将rho替换为10条校准轨迹最大值；也比较直接任务绝对误差最大值和选择模型误差最大值。下表列全部4任务同窗落入包络的比例；完整逐任务覆盖/归一化半径/整轨迹通过率在empirical_bounds.csv。校准轨迹同时参与选模，不假设交换性，不声称90%共形、条件覆盖或跨工况保证。','',
        '| 时间段 | 残差范数包络覆盖 | 直接任务包络覆盖 | 修正后任务包络覆盖 |','|---|---:|---:|---:|']
    for split in ['development','late','future']:
        q=bdf[(bdf.split==split)&(bdf.task=='joint')].set_index('method');lines.append(f"| {split} | {100*q.loc['norm_envelope','coverage']:.1f}% | {100*q.loc['direct_task_envelope','coverage']:.1f}% | {100*q.loc['selected_task_envelope','coverage']:.1f}% |")
    lines+=['','高覆盖伴随明显宽度代价：开发段范数包络的四任务平均半径，分别为训练基线RMSE的12.44/2.99/6.57/8.80倍；直接任务包络为2.68/1.80/1.99/3.07倍。按轨迹汇总的包络/实际绝对误差中位比约15–41倍。开发段范数包络仍有1/12轨迹出现至少一次违例。因此没有把接近100%的经验覆盖写成紧致或可靠的物理误差证书。','',
        '## 审计、成本与复现','',
        '独立NumPy计算四任务、对称极化的一阶项、二阶项和真实残差范数界；解析线性旋转场检验有符号坐标的curl；公开history-only接口逐窗口重放全部模型，干预未来标签不改输出。具体误差和通过状态见audit.json。原始实际batch1预测与已审计主动观测缓存精确一致；所有拟合投影仅用训练轨迹。','',
        '训练819窗口/46轨迹，cal90/10；dev108/12，late120和future72使用相同12条开发轨迹的后续时间。数据有多轮历史复用，bootstrap按轨迹，只作描述性探索；不能写成新数据确认。代码、预先协议、选模网格、预测、逐窗/逐轨迹/Re统计都保留。']
    if (OUT/'audit.json').exists():lines+=['', '审计摘要：`'+json.dumps(json.loads((OUT/'audit.json').read_text()),ensure_ascii=False)+'`。']
    if (OUT/'timing.csv').exists():
        timing=pd.read_csv(OUT/'timing.csv');lines+=['','| 方法 | batch1中位毫秒 |','|---|---:|'];lines += [f"| {r['method']} | {r['median']:.3f} |" for r in timing.to_dict('records')];lines+=['','A10080GB，CPU物理输入和CPU任务量输出，6窗口×5交错重复，不含初始化。这里不生成分位头/联合样本，因此不应直接拿此毫秒数解释先前联合分布接口的速度。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,split in zip(axes,['development','late','future']):
        sm=summaries[split]
        for model,label in [('mean','Mean bias'),('direct/X+P/16','XP'),(chosen,'Cal-selected'),('direct/task_weighted/16','Task-weighted')]:
            values=[100*(1-sm.loc[model,t+'_mse']/sm.loc['base',t+'_mse']) for t in TASKS];ax.plot(range(4),values,'o-',label=label)
        ax.axhline(0,color='grey',lw=.7);ax.set_xticks(range(4));ax.set_xticklabels(['KE','TKE','Enstrophy','Region u'],rotation=20);ax.set_title(split);ax.grid(alpha=.15)
    axes[0].set_ylabel('Task MSE improvement vs J(P), %');axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'task_gains.png',dpi=160);fig.savefig(OUT/'task_gains.pdf');plt.close(fig)
    dump(OUT/'completion_manifest.json',dict(full_goal_complete=False,task_pilot='evaluated; audit status explicit',chosen=chosen,protocol_sha256=digest(Path(__file__).with_name('PROTOCOL.md')),additional_controls_protocol_sha256=digest(Path(__file__).with_name('SIMPLE_CONTROL_ADDENDUM.md')),sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},audit=json.loads((OUT/'audit.json').read_text()) if (OUT/'audit.json').exists() else None))
    print('REPORT',chosen,flush=True)

if __name__=='__main__':run()
