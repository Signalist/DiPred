"""Finish saved-output analysis and concise Chinese report, no new forwards."""
from collections import defaultdict
from .core import *
from .verify import csvread


def bootstrap(values):
    v=np.asarray(values,dtype=float);rng=np.random.default_rng(123)
    estimates=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
    return dict(delta=float(v.mean()),ci95_low=float(np.quantile(estimates,.025)),ci95_high=float(np.quantile(estimates,.975)),
                improved_trajectories=int((v<0).sum()),trajectories=len(v))


def main():
    verification=read(OUT/'VERIFICATION.json');assert verification['passed']
    rows=csvread(OUT/'DEVELOPMENT.csv');selection,tr,macro=select(rows)
    assert selection==read(OUT/'SELECTION.json')
    lookup={(r['arm'],r['step'],r['trajectory_id'],r['channel']):r for r in tr}
    comparisons=[];trajectory_diffs=[]
    names=sorted({r['trajectory_id'] for r in tr})
    for step in (250,500):
        for a,b in [('C0','native'),('C1','native'),('C1','C0')]:
            for ch in ('overall','u','v'):
                for metric in METRICS:
                    ds=[]
                    for name in names:
                        left=lookup[a,step,name,ch];right=lookup[b,0 if b=='native' else step,name,ch]
                        ds.append(left[metric]-right[metric])
                        if metric in ('crps','energy','mse','smse','std_variance'):
                            trajectory_diffs.append(dict(a=a,b=b,step=step,trajectory_id=name,Re=left['Re'],channel=ch,metric=metric,delta=ds[-1]))
                    comparisons.append(dict(a=a,b=b,step=step,channel=ch,metric=metric,**bootstrap(ds)))
    csvwrite(OUT/'PAIRED_COMPARISONS.csv',comparisons);csvwrite(OUT/'PAIRED_TRAJECTORY.csv',trajectory_diffs)
    grouped=defaultdict(list)
    for r in tr:grouped[(r['arm'],r['step'],r['channel'],r['Re'])].append(r)
    byRe=[dict(zip(('arm','step','channel','Re'),k),trajectories=len(rr),**{m:float(np.mean([r[m] for r in rr])) for m in METRICS}) for k,rr in grouped.items()]
    csvwrite(OUT/'DEVELOPMENT_RE.csv',byRe)
    delta=csvread(OUT/'DELTA_RMS.csv');dg=defaultdict(list)
    for r in delta:dg[(r['arm'],r['step'],int(r['label']))].append(r)
    ds=[dict(arm=k[0],step=k[1],label=k[2],delta_rms=float(np.mean([float(r['delta_rms']) for r in rr]))) for k,rr in dg.items()]
    csvwrite(OUT/'DELTA_SUMMARY.csv',ds)
    costs=read(OUT/'COST.json');logs={a:[json.loads(t) for t in (OUT/f'train_{a}.jsonl').read_text().splitlines()] for a in ARMS}
    costs['optimization_step_seconds']={a:sum(r['seconds'] for r in logs[a]) for a in ARMS}
    costs['development_evaluation_seconds']=sum(r['development_seconds'] for r in costs['checkpoint_evaluation'])
    costs['native_baseline_groups']=36;costs['native_baseline_paths']=1152
    costs['trained_candidate_eval_groups']=144;costs['trained_candidate_eval_paths']=4608
    costs['XPF_condition_windows']=173;costs['XPF_primary_backbone_calls']=692
    costs['test_groups']=0;costs['no_train_restarts']=True
    costs['deployment_extra']='Per release 4 native calls+frozen XPF head for W, plus previous-window 32-member frozen native shadow forecast for E. Current sampled backbone runs unchanged four iterations with lightweight correction.'
    costs['cache_bytes']=sum(p.stat().st_size for p in CACHE.glob('*.pt'))
    dump(OUT/'COST.json',costs)
    training=read(OUT/'TRAINING_COMPLETE.json')
    n=next(r for r in macro if r['arm']=='native' and r['channel']=='overall')
    ms={(r['arm'],r['step'],r['channel']):r for r in macro}
    table=['|模型|步数|经验CRPS|fair32 CRPS|路径ES|原MSE|SMSE|标准化方差|90%标准化宽度|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,step in [('native',0)]+[(a,s) for a in ARMS for s in (250,500)]:
        r=ms[arm,step,'overall']
        table.append(f"|{arm}|{step}|{r['crps']:.9f}|{r['fair_crps']:.9f}|{r['energy']:.9f}|{r['mse']:.9g}|{r['smse']:.9f}|{r['std_variance']:.9f}|{r['std_width90']:.9f}|")
    lines=['# Residual & Uncertainty Conditioned Diffusion：中文决策报告','',
      '**情形A：C0、C1均未超过开发原生经验CRPS，科学选择均为原生。固定实验结束，未启动测试推理或后续训练。**', '',
      '本轮C0/C1各seed42、500次优化更新，总计1000，训练8成员fair CRPS、开发32成员经验CRPS。参数/输入/实际噪声/轨迹抽样计划完全对齐；没有训练重启。旧Trust审计与全部旧文件保持原样。', '',
      '## 固定开发结果','']+table+['',
      '评分先在每条原轨迹内平均3窗口，再对12条开发轨迹等权。候选仅原生、250、500；不以MSE保护条件覆盖科学选择。初始化最后G输出层为0，完整四步与原生一致；不作为额外候选。','',
      'C0最佳非零250步相对原生CRPS差为+7.5677e-6；C1最佳非零250步差为+2.1483e-5。500步C1的CRPS略低于C0约1.4420e-5，但两者仍退步于原生，且C1路径ES、原/标准化MSE均不占优，不能称条件收益。','',
      '32成员fair敏感性下，250步两模型略低于原生；按事先规定的经验CRPS选模仍回退原生，不利用fair结果重新选择。各非零候选的路径ES和原MSE均上升。分通道u的CRPS降低，v升高；总体均值的微小变化不能支持全面概率改善。','',
      '标准化集合方差相对原生：250步C0/C1约+3.52%/+3.50%，500步约+5.48%/+5.32%；90%宽度也扩大，因此本次没有通过压缩分布获得CRPS收益。方差扩大同样不代表质量提升。','',
      '逐轨迹差、u/v和Re组、10000次seed123的原轨迹配对bootstrap在PAIRED_COMPARISONS/PAIRED_TRAJECTORY/DEVELOPMENT_RE中。区间为12条已用轨迹的探索性描述，不把窗口、成员、像素或一个训练种子当独立原轨迹，不声称多重比较确认。','',
      '## 方法与因果边界','',
      'E保留两通道20帧有符号残差，等于上一冻结原生32成员均值减当前已观测历史，除冻结通道std；对应物理时间严格一致。训练固定E绝对值99.5%裁剪阈值1.679229856。E不是当前未来标签，也不是观测噪声。','',
      'W由冻结原生零噪声99/66/33/0的点预测和隐藏特征，以及冻结Selected16QuantileHead原始5%/95%残差界取得半宽。保留实际float16隐藏舍入、训练统计和[-10,10]处理；Selected16选用6个base槽不依赖curl几何，已与原特征路径逐值验证。没有虚构真实几何、没有在线校准，也没有重训XPF。','',
      '训练137窗口估计W通道RMS分别0.228806108、0.616460025，再固定[0,10]裁剪；不按窗口归一化。同时广播每通道W整体均值，整体风险一起增大时输入确实改变。未强制风险越高修正越强。','',
      'G为211→16(1×1)→16(3×3)→40(1×1)，6392参数、最后层零初始化、固定缩放0.1。输入当前噪声状态、完整H、E、W、up2的48通道空间特征、W整体均值和标签。C0对应E/W槽全零；C1全用。每步clean_new=clean_base+Delta后继续同一采样更新，不修改历史、不做邻域凸组合、不做最终共同偏差平移、不跨成员混合。','',
      '仅G与native.net.out.weight/bias（17320参数）训练，Adam两组lr3e-4/1e-5。其余主干及XPF冻结。C0/C1输出层最终参数变化RMS分别0.000222684555、0.000220972536；前后输出层哈希和非out冻结哈希详见TRAINING_COMPLETE及VERIFICATION。','',
      '## 测试、解释及成本','',
      '训练前12项门禁和绝对风险结构测试通过，包括真实E时间身份、XPF特征/权重等价、未来污染输入和发布预测不变、关闭与初始化精确复现、fair8显式值/梯度/差分、各去噪步clean梯度、非零合成G上游梯度、out梯度、训练统计唯一来源、参数/实际噪声相同。完整评价/报告/科学选模合成流程在优化前跑通。','',
      '两模型都无非零科学选择，因此不运行正式测试或G_off/E_off/W_off候选诊断；这些解释问题在本轮没有选中模型可回答，不以未选模型作事后部署证据。Delta RMS仍来自固定候选原有评价前向，见DELTA_SUMMARY，证明训练后条件修正已进入四步但不证明有效机制。','',
      f"新增W条件准备（173窗、每窗4主干调用及冻结XPF）：{costs['XPF_cost']['train']['seconds']+costs['XPF_cost']['development']['seconds']:.3f}秒，含数据/缓存I/O和少量等价核验。新FP32开发基线36组/1152路径：{costs['baseline_development_seconds']:.3f}秒。1000配对更新及候选评价墙钟{costs['training_wall_seconds']:.3f}秒；优化步骤C0/C1计时{costs['optimization_step_seconds']['C0']:.3f}/{costs['optimization_step_seconds']['C1']:.3f}秒，候选评价合计{costs['development_evaluation_seconds']:.3f}秒。新缓存{costs['cache_bytes']}字节（约{costs['cache_bytes']/2**30:.3f}GiB）。这些不是Codex服务费用。", '',
      '运行时W额外需要4次冻结原生调用和XPF头；E仍依赖上一窗口完整32成员影子预测。本轮没有正面增益证明这些成本值得。缓存W避免每次优化重复XPF；这不是新模型自身反馈闭环。','',
      '## 对九个研究问题的回答','',
      '1. E/W确实以合法条件进入逐步干净状态修正；绝非最终统一平移。',
      '2. 只更新G与native.net.out，输出层变化真实非零，其余主干和XPF哈希不变。',
      '3. C1没有可选择地优于同预算C0；500步小CRPS差伴随ES/MSE代价，不说明E、W各自有效。',
      '4. 两者未超过完全未修改的FP32 DiPred，科学选择原生。',
      '5. 集合方差/宽度扩大，收益不是压缩概率分布；没有完整概率收益。',
      '6. XPF额外4次主干调用当前没有收益证据支持部署价值。',
      '7. 整体W风险上升的绝对信息被固定训练尺度及均值广播保留。',
      '8. fair8训练排除自配对，empirical32开发为选择口径；fair32只作敏感性，不能比较跨K绝对分数或事后改选。',
      '9. 不建议本证据下自动扩大Backbone微调；本固定方案归档，若未来新立项须新的独立验证、更多种子及预设选择，而本轮不启动。','',
      '正负结论均仅针对本次定义及固定预算。未证实历史残差/XPF完全无价值，未证明物理机制、联合路径一致性或论文级创新。旧反复使用轨迹仅探索性。本轮严格停止。']
    (OUT/'DECISION_REPORT.md').write_text('\n'.join(lines)+'\n')
    paths=[p for d in (CODE,OUT) for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_HASHES.json']
    dump(OUT/'ARTIFACT_HASHES.json',{str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)})
    print('ANALYSIS REPORT HASHES COMPLETE; no new model calls',flush=True)

if __name__=='__main__':main()
