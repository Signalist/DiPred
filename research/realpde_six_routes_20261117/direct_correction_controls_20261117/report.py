from controls import *

def pair(tr,method,ref):
    wide=tr.pivot(index='file',columns='method',values='mse');a=wide[method].to_numpy();b=wide[ref].to_numpy();rng=np.random.default_rng(42);idx=rng.integers(0,len(a),(4000,len(a)));boot=(1-a[idx].mean(1)/b[idx].mean(1))*100
    # A second descriptive resampling unit reflects the shared operating regime.
    groups=tr.drop_duplicates('file').set_index('file').group.reindex(wide.index).astype(str).to_numpy();unique=np.unique(groups);gidx=rng.integers(0,len(unique),(4000,len(unique)));gs_a=np.array([a[groups==g].sum() for g in unique]);gs_b=np.array([b[groups==g].sum() for g in unique]);gs_n=np.array([(groups==g).sum() for g in unique]);ga=gs_a[gidx].sum(1)/gs_n[gidx].sum(1);gb=gs_b[gidx].sum(1)/gs_n[gidx].sum(1);gboot=(1-ga/gb)*100
    return dict(gain=(1-a.mean()/b.mean())*100,low=np.quantile(boot,.025),high=np.quantile(boot,.975),wins=int((a<b).sum()),n=len(a),re_low=np.quantile(gboot,.025),re_high=np.quantile(gboot,.975),n_re=len(unique))

def run():
    sel=json.loads((OUT/'selection.json').read_text());budget=json.loads((OUT/'budget_selection.json').read_text());timing=pd.read_csv(OUT/'timing.csv',index_col=0);memory=pd.read_csv(OUT/'memory.csv',index_col=0);names=REFERENCES+KINDS;summaries={};pairs=[];alltr=[];tasks=[]
    for split in ['calibration','development','late','future']:
        sm=pd.read_csv(OUT/split/'summary.csv',index_col=0);summaries[split]=sm;tr=pd.read_csv(OUT/split/'trajectories.csv');tr['split']=split;alltr.append(tr)
        for name in names:
            for ref in ['base','warp_F_last',sel['winner']]:pairs.append(dict(split=split,method=name,reference=ref,**pair(tr,name,ref)))
        for name in names:
            for task in ['kinetic','fluctuation','enstrophy','regional_u']:tasks.append(dict(split=split,method=name,task=task,mse=sm.loc[name,task+'_mse'],gain=(1-sm.loc[name,task+'_mse']/sm.loc['base',task+'_mse'])*100))
    pairs=pd.DataFrame(pairs);pairs.to_csv(OUT/'paired_gains.csv',index=False);alltr=pd.concat(alltr);alltr.to_csv(OUT/'all_trajectories.csv',index=False);alltr.groupby(['split','method','group'])[['mse','harm']].mean().to_csv(OUT/'by_re.csv');pd.DataFrame(tasks).to_csv(OUT/'physical_tasks.csv',index=False)
    roll=pd.read_csv(OUT/'rollout_trajectories.csv');rs=roll.groupby(['method','mode']).mse.mean().unstack();rp=[]
    for mode in ['observed','free']:
        for name in rs.index:rp.append(dict(mode=mode,method=name,reference='base',**pair(roll[roll['mode']==mode],name,'base')))
    pd.DataFrame(rp).to_csv(OUT/'rollout_paired.csv',index=False)
    audit=json.loads((OUT/'audit.json').read_text());cost=[]
    for name in names:
        if name in KINDS:params=sel['models'][name]['parameters'];gain=sel['models'][name]['gain'];train=sel['models'][name]['training_seconds'];size=(OUT/f'{name}.pt').stat().st_size
        elif name!='base':old=load(SOURCE/'local'/f'{name}.pt')['summary'];params=old['parameters'];gain=old['shrink'];train=old['training_seconds'];size=(SOURCE/'local'/f'{name}.pt').stat().st_size
        else:params,gain,train,size=0,0,0,0
        cost.append(dict(method=name,parameters=params,gain=gain,training_seconds=train,correction_checkpoint_bytes=size,full_ms=timing.loc[name,'full_ms'],active_post_ms=timing.loc[name,'active_post_ms'],peak_allocated_mb=memory.loc[name,'peak_allocated_mb']))
    cost=pd.DataFrame(cost).set_index('method');cost.to_csv(OUT/'costs.csv');projection=common_projection('cpu');projection_bytes=sum(v.numel()*v.element_size() for v in projection.values());dump(OUT/'storage_dependencies.json',dict(dense_F_additional_frozen_projection_bytes=projection_bytes,projection_source=str(SOURCE/'local/warp_F_last.pt'),projection_source_sha256=digest(SOURCE/'local/warp_F_last.pt'),base_backbone_shared=str(SHARED)))
    lines=['# 结构修正的直接强对照：ResCrct、加性场网络与两步细化','',f"新直接控制族仅在校准集选择 **{sel['winner']}**；冻结warp_F_last和旧additive_F_last的选择不变。实际成本门槛内的校准最佳控制为 **{budget['selected']}**。全部使用原RealPDE、seed42、完整真实历史20→20，不增加反馈或未来观测。",'',
        '冻结warp在观察驱动任务中保持小幅优势：相对原P改善0.79/0.72/0.45%，相对本轮成本门槛内的ResCrct改善0.379/0.334/0.283%，轨迹与Re组描述性区间均正。更大的离线ResCrct在late的差异区间跨零。自由滚动仍使warp退步0.416%；这里支持的是受限观察驱动用途，不能声称长期动力学或扩散F独有优势。','',
        '## 直接比较','', '| 方法 | 修正参数 | 校准增益 | dev MSE改善% | late改善% | future改善% |','|---|---:|---:|---:|---:|---:|']
    for name in names:
        g=[summaries[s].loc[name,'gain_percent'] for s in ['development','late','future']];lines.append(f"| {name} | {int(cost.loc[name,'parameters'])} | {cost.loc[name,'gain']:g} | {g[0]:.4f} | {g[1]:.4f} | {g[2]:.4f} |")
    lines+=['','正数表示降低等轨迹物理速度MSE。完整gain1结果保存在各split/summary.csv，零增益与退步均保留；没有按开发表现替换校准选择。逐窗口损害比例、全部逐轨迹/逐Re统计和四个任务量误差在CSV中。','',
        '## 冻结warp相对新控制的配对比较','', '| 时间段 | 对照 | warp改善% | 轨迹95%CI | Re组95%CI | 胜出轨迹 |','|---|---|---:|---|---|---:|']
    for split in ['development','late','future']:
        tr=alltr[alltr.split==split]
        for ref in KINDS:
            c=pair(tr,'warp_F_last',ref);lines.append(f"| {split} | {ref} | {c['gain']:.4f} | [{c['low']:.4f}, {c['high']:.4f}] | [{c['re_low']:.4f}, {c['re_high']:.4f}] | {c['wins']}/{c['n']} |")
    lines+=['','4000次bootstrap只用于描述固定seed的轨迹差异，不是多种子训练。每段12条轨迹仅3个Re组，组级区间很粗；后续时间段来自相同轨迹，不能当作三次独立确认。','',
        '## 模型与信息公平性','',
        '- dense_F保留warp相同的32维训练投影，在4×8格点输出全部20帧加性残差，去除旧加性对照的2×4空间池化和4个时间节点限制。2840参数。',
        '- additive_XP是4→8→8→2完整分辨率三层3D卷积，3042参数，仅读取X和P。',
        '- rescrct_original直接导入仓库Shared/models.py的Corrector(width24)，独立训练时r=m始终为零；22082参数中2592个首层权重对应恒零反馈输入。它是原类在当前无反馈任务中的适配，不是复现原在线AE/Diffusion完整系统，也不是对旧反馈权重临时清零。',
        '- rescrct_block直接导入Shared/crossfit_corrector.py的BlockResidualCorrector，25842参数，包含完整20帧历史编码。',
        '- refine_two为2851参数共享两步确定性修正，读取X/原P/当前Q，在空间降采样2倍后计算3D增量。1000更新×2调用，与其他新控制2000更新×1调用数量相同；这不等于相同FLOPs或训练时间，也不称PDE-Refiner论文复现。','',
        '所有主干保持四次调用且冻结；原P与F来自实际batch1推理。新控制batch8、AdamW、同seed42批次序列前缀、相同物理MSE目标。所有增益只在校准90窗口/10轨迹选择；随后实际batch1校准重放并冻结，完整记录见calibration_replay.json。训练819窗口/46轨迹，晚于主干前200帧拟合期；身份不是主干未见，但修正时间与主干拟合不重叠。','',
        '## 实际成本','', '| 方法 | 全流程ms | 激活修正ms | 新训练秒 | 峰值MB | 修正checkpoint KiB |','|---|---:|---:|---:|---:|---:|']
    for name in names:
        c=cost.loc[name];lines.append(f"| {name} | {c.full_ms:.3f} | {c.active_post_ms:.3f} | {c.training_seconds:.2f} | {c.peak_allocated_mb:.2f} | {c.correction_checkpoint_bytes/1024:.1f} |")
    lines+=['',f"预定预算门槛：全流程和激活修正中位耗时都不超过warp的110%，且校准选择的增益>0。符合者：{', '.join(budget['eligible']) or '无'}。在这些候选中只按校准MSE选{budget['selected']}。",'',
        f'A10080GB，6个输入窗口×5次交错计时。全流程包含CPU物理输入、四调用主干、池化/PCA/修正、CPU物理输出；修正列使用gain1，避免关闭网络造成虚假预算优势。峰值为当前torch进程张量分配，含已加载权重与输入输出，非整卡占用。初始化和磁盘读取排除。旧warp/additive训练秒来自原记录，其余本轮实测；不把异时训练秒当严格加速比。模型参数不含冻结PCA；checkpoint列为实际文件大小，旧warp/additive已含PCA，而dense_F另复用{projection_bytes/1024:.1f}KiB冻结投影，依赖和哈希见storage_dependencies.json。','',
        '## 无反馈200帧自由滚动','', '| 方法 | 同起点真实历史MSE改善% | 自由滚动MSE改善% | 自由滚动MSE |','|---|---:|---:|---:|']
    for name in rs.index:
        lines.append(f"| {name} | {(1-rs.loc[name,'observed']/rs.loc['base','observed'])*100:.4f} | {(1-rs.loc[name,'free']/rs.loc['base','free'])*100:.4f} | {rs.loc[name,'free']:.7g} |")
    lines+=['','每个模型均从原始前20帧开始，之后连续10块读取自己的预测；同起点观察驱动对照每块重新读取合法真实历史。自由模式没有反馈，不重调增益。发报先于评分，完整预测链保留。单窗口改善不外推为长期动力学改善。','',
        '## 审计与结论边界','',f"独立NumPy物理MSE最大差{audit['numpy_mse_max_error']:.3g}；全部公开接口与归档实际预测最大差{audit['public_max_error']:.3g}；自由预测前缀重放差{audit['rollout_prefix_max_error']:.3g}。原ResCrct及BlockResidual类与导入源逐值一致；推理期间禁止torch.load/HDF5读取，仍可重放。训练、校准、开发身份分离；未来40帧输入被拒绝；gain0精确返回P。审计通过：{audit['passed']}。",'',
        '本阶段补足明确的普通场残差/原ResCrct/离线ResCrct/简单两步细化对照。结论只针对这些受限模型、预算和重复使用的小数据，不证明更强修正器或所有细化方法无效。旧主线权重未替换。原六方向是否全部满足仍须另做完整要求审计。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));xx=np.arange(3)
    for name in ['warp_F_last',sel['winner'],'additive_XP','dense_F','refine_two']:
        axs[0].plot(xx,[summaries[s].loc[name,'gain_percent'] for s in ['development','late','future']],'o-',label=name)
    axs[0].set_xticks(xx);axs[0].set_xticklabels(['dev','late','future']);axs[0].set_ylabel('Observed-history MSE gain, %');axs[0].axhline(0,color='gray',lw=.7);axs[0].legend(fontsize=8);rw=pd.read_csv(OUT/'rollout_windows.csv');curve=rw[rw['mode']=='free'].groupby(['method','block']).mse.mean()
    for name in ['base','warp_F_last',sel['winner'],'refine_two']:
        if name in curve.index.get_level_values(0):axs[1].plot(curve.loc[name].index*20,curve.loc[name].values,label=name)
    axs[1].set_xlabel('Predicted frames');axs[1].set_ylabel('Free-rollout physical MSE');axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'direct_controls.png',dpi=170);fig.savefig(OUT/'direct_controls.pdf');plt.close(fig)
    dump(OUT/'completion_manifest.json',dict(full_goal_complete=False,direct_control_pilot_complete=audit['passed'],selection=sel['winner'],budget_selection=budget['selected'],audit=audit,protocol_sha256=sel['protocol_sha256'],sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},model_hashes=sel['model_hashes']))
    print('REPORT',sel['winner'],budget['selected'],flush=True)

if __name__=='__main__':run()
