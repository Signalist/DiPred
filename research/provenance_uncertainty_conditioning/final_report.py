"""Build Chinese evidence report from finalized, verified statistics; no new experiment."""
from core import *
from collections import Counter


def mdtable(rows,columns):
    lines=['| '+' | '.join(label for key,label in columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
    for r in rows:
        values=[]
        for key,label in columns:
            v=r.get(key,'')
            if isinstance(v,(float,np.floating)):v=f'{v:.7g}'
            values.append(str(v))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)+'\n'


def numeric(rows):
    for row in rows:
        for key in QUAL+['mean','ci98_333_low','ci98_333_high','development_restart_crps','seam_rms','seam_increment_mse','parent_seam_rms','acf_rmse','temporal_corr_difference_B','restart_excess_ed_u']:
            if row.get(key) not in ('',None):row[key]=float(row[key])
    return rows


def main():
    assert (OUT/'VERIFICATION.json').exists() and (OUT/'ANALYSIS.json').exists()
    assert not (OUT/'DECISION_REPORT.md').exists()
    selection=json.loads((OUT/'selection.json').read_text());candidates=read_csv(OUT/'selection_candidates.csv')
    summ=numeric(read_csv(OUT/'scores_summary.csv'));paired=numeric(read_csv(OUT/'paired_intervals.csv'))
    dev=numeric(read_csv(OUT/'development_nonzero_paired.csv'));dv=numeric(read_csv(OUT/'development_summary.csv'))
    train=json.loads((OUT/'TRAINING_COMPLETE.json').read_text());cost=json.loads((OUT/'EVALUATION_COST.json').read_text());ver=json.loads((OUT/'VERIFICATION.json').read_text());pre=json.loads((OUT/'PREFLIGHT.json').read_text())
    invalid=OUT/'invalid_direct_training';bad=json.loads((invalid/'TRAINING_COMPLETE.json').read_text());badlog=(invalid/'evaluate.log').read_text().splitlines()
    invalid_done=sum(1 for _ in (invalid/'evaluation_manifest.jsonl').open());invalid_native=sum(1 for _ in (invalid/'extension_manifest.jsonl').open())
    totals=Counter(r['step'] for r in selection);allzero=all(r['step']==0 for r in selection)
    candidate_rows=[]
    for arm in CFG['arms']:
        for seed in CFG['seeds']:
            own=[r for r in candidates if r['arm']==arm and int(r['seed'])==seed and int(r['step'])>0]
            best=min(own,key=lambda r:(float(r['development_restart_crps']),int(r['step'])))
            baseline=next(float(r['development_restart_crps']) for r in candidates if r['arm']==arm and int(r['seed'])==seed and r['step']=='0')
            ratios={key:float(best[key]) for key in best if key.startswith(('mse_k','smse_k'))}
            candidate_rows.append(dict(arm=arm,seed=seed,best_nonzero_step=int(best['step']),nonzero_crps=float(best['development_restart_crps']),change_percent=100*(float(best['development_restart_crps'])/baseline-1),worst_mse_percent=100*(max(ratios.values())-1),worst_guard=max(ratios,key=ratios.get),eligible=best['eligible'],scientific_step=next(r['step'] for r in selection if r['arm']==arm and r['seed']==seed and r['policy']=='scientific'),guarded_step=next(r['step'] for r in selection if r['arm']==arm and r['seed']==seed and r['policy']=='guarded')))
    csvwrite(OUT/'best_nonzero_disclosure.csv',candidate_rows)
    quality=[r for r in summ if r['channel']=='overall' and r['policy']=='native' and r['view'] in ['tail','joint']]
    baselinepair=[r for r in paired if r['policy']=='native' and r['contrast']=='S minus B' and r['channel']=='overall' and r['metric'] in ['crps','energy','mse','smse']]
    increments=[r for r in dev if r['seed']=='mean3' and r['channel']=='overall' and r['metric']=='crps' and r['view']=='tail' and r['step'] in ['250','1000'] and r['contrast'] in ['C_mask minus C_aug','C_unc minus C_mask']]
    report=f'''# 生成历史来源与父集合不确定性条件化：研究报告

本轮执行完成：符合原任务的三臂×种子42/43/44共九次训练，每次1000步，使用完整32成员经验CRPS，45个checkpoint；随后完成48个历史测试组和480个后续窗口组、统计及审计。实际起始HEAD=`{CFG['head']}`，分支dynamics；最终提交号见Git包含本报告的提交及交付消息。

**执行完成与经验收益分开判断。** 两类开发选择的步数分布为{dict(totals)}。'''
    if allzero:report+='''科学候选与MSE保护候选的18项选择全部回退step0，意味着本次固定架构和预算下没有获得可选择的开发CRPS增益。正式选择模型的重启预测等于原生S，真实历史直接预测等于原生B；这些零差不证明来源或不确定性信息有价值，也不能证明该信息在所有其他参数化下都无效。所有36个非零候选的开发结果单独披露，没有用回退掩盖非零网络。
'''
    report+='''
## 训练范围纠错和探索性边界

此前实现误把k0直接样本加入训练，260/1000更新不满足任务“每个训练样本k∈{2,5,10}”要求。该批9000更新、45权重、原协议/配置/代码、开发结果和中止评价完整保存在 `invalid_direct_training/`，不计为有效九次训练，也不进入任何有效选择/统计。纠正仅恢复原任务训练域；1760参数、lr=.001、Adam、初始化种子、1000步、loss、候选和选择门槛不变，没有根据结果重新搜索超参。详见PROTOCOL_CORRECTION.md。

有效批次在训练前重新固定协议并重新预检。训练1000共同样本中k2/5/10为356/297/347，覆盖46条原训练轨迹；k0只用于直接预测预检、开发保护和正式评价。全部三臂使用同一抽样顺序与实际噪声，三种子只改变对齐的适配器初始化。错误批次冻结原生A/B/C/S不依赖任何适配权重，锁定manifest并核对哈希后允许复用；错误适配权重和分数不复用。

历史测试及后续时段在先前研究与该范围纠错前均已有探索使用。所有结果都属于探索性，不能称新盲测；修复本身不恢复独立性。只有3个测试Re组，按12条轨迹bootstrap不能代表跨全部Re推广。

## 数据、模型与完整集合目标

数据来自 `DiPred+ResCrct/OLD/REALPDE/realpdebench-competition/data/train_real/`，u/v空间`::2,::2`得到32×64。46/12/12原轨迹及Re隔离沿用旧划分，train/dev/test历史起点400/420、两独立重复。后续测试固定历史440至820，步长20，共20窗口×12轨迹×2重复。源HDF5完整哈希、原缓存/成员/噪声身份在data_sources、config、evaluation/extension manifests中，未使用500000系列缓存。

子模型保留原生20帧历史→联合20帧输出、四步99/66/33/0去噪。原主干、父生成器及归一化不更新。相同1760参数适配器输入101槽：标准化历史40、来源mask20、父离散度40、去噪步1；卷积101→8(1×1)、8→8(3×3)、8→40(1×1)，中间SiLU，最后层零初始化。每次去噪都把适配量加入历史条件，原DynamicsNet重新计算history.diff/scale，四步完整图训练。

C_aug屏蔽mask/std；C_mask仅mask有效；C_unc两者有效。mask仅末k位置=1；std为父完整32成员前k帧的逐点总体标准差除冻结通道std，放到末k位置；真实位置0仅表示无生成来源集合离差，不意味着物理无不确定性。每个父成员保持原时空身份并对应自己的一个子噪声，未先均值化父集合。输入函数不接受truth。

训练使用完整32成员、完整子20帧标准化经验CRPS：mean|p_i−y|−sum_ij|p_i−p_j|/(2M²)。成员距离项实际参与梯度，无Monte Carlo小成员近似、无共同平移简化loss。每次1000步Adam，lr=.001，batch1个完整成员块；无新增正则、JVP目标或误差监督。父离散度在当前未来标签读取前计算，未来仅进入损失/评分。

物理索引（0基）：H=[start,start+20)，混合H'=H[k:]+A_i[:k]，子完整输出[start+20+k,start+40+k)。tail取子前20−k，对齐B后20−k；joint为同成员A前k+子前20−k，对齐原完整20未来；direct用全真实H、零mask/std及原B噪声，与原B完整20比较。正式全部预测先记hash并发布，再读评分truth；训练和开发标签未进入发布条件。

## 预检和实现证据

'''
    report+=f"有效PREFLIGHT通过：完整CRPS显式成员对数值误差{pre['checks']['explicit_crps_value_error']:.3g}、梯度误差{pre['checks']['explicit_crps_gradient_error']:.3g}，gradcheck通过；真实网络成员距离项适配器梯度范数{pre['checks']['pair_term_adapter_gradient_norm']:.6g}，四次适配输出梯度RMS均非零。24组零初始化预测满足maxabs≤2e−6；参数/初值对齐、时间/成员/噪声、mask/std位置、未来污染和冻结主干检查通过。非零固定诊断适配器改变成员间差，非最终共同平移。\n\n"
    report+='''EVALUATION_TESTS验证完整评分、逐lead汇总、显式时间协方差和拼接成员；PLAN_CHECKS验证528组覆盖、目标帧可用、Re隔离及145920个旧+后续原生种子唯一。tests.log、evaluation_tests.log及相应JSON保留。最初Python3.8路径API错误在任何优化前退出，产物单独保存在invalid_direct_training/preupdate_failure/，兼容修复后预检，未隐藏失败。

## 开发选择和非零候选披露

每个arm/seed在0/250/500/750/1000步保留候选。科学选择最小开发restart-tail CRPS，轨迹等权、三个k等权；保护选择另要求各k重启尾段及direct的原MSE/SMSE均≤对应原生均值1.01，且CRPS严格改善才可离开step0。这个1%规则是描述性保护界，不是统计非劣证明。

下表“最佳非零”仅为审查四个预定非零候选，不是新增正式选择。原生选择CRPS为'''+f"{float(candidates[0]['development_restart_crps']):.9f}。\n\n"
    report+=mdtable(candidate_rows,[('arm','臂'),('seed','种子'),('best_nonzero_step','最佳非零步'),('nonzero_crps','开发CRPS'),('change_percent','相对原生%'),('worst_mse_percent','最差MSE增幅%'),('worst_guard','保护项'),('eligible','合格'),('scientific_step','科学步'),('guarded_step','保护步')])
    report+='''
![开发候选的概率质量与点预测代价](development_tradeoffs.png)

完整其他指标包括ES、原MSE/SMSE、方差、宽度和时间均值场CRPS在development.csv及development_summary.csv；36个非零候选逐轨迹/Re/通道及与原生、同一步臂间的配对差在development_nonzero_*。同一步数的来源/不确定性增量如下（平均三种子后按原轨迹bootstrap，98.333%区间）：

'''+mdtable(increments,[('step','步'),('k','k'),('contrast','差'),('mean','CRPS差'),('ci98_333_low','下界'),('ci98_333_high','上界')])
    report+='''
局部开发差异不等于独立确认，也不能把C_mask相对C_aug的小改善当作超过原生模型。所有模型、选择/权重hash先冻结，再评分历史测试/后续窗口；未根据测试追加候选或改变规则。

## 冻结选择的正式质量

下面为原生B/S的全部探索测试/后续窗口、两通道整体均值。若选择step0，C_aug/C_mask/C_unc两种政策及各seed对应S（tail/joint）或B（direct）逐值相同；完整政策/seed表仍全部输出，不把相同结果重复解释为独立证据。

'''+mdtable(quality,[('split','划分'),('k','k'),('view','视图'),('arm','基线'),('crps','CRPS'),('energy','完整ES'),('mse','原MSE'),('smse','SMSE'),('standardized_variance','标准化方差'),('standardized_width90','标准化90%宽度'),('time_mean_crps','时间均值CRPS')])
    report+='''
这里CRPS/ES/SMSE及standardized列用冻结训练尺度；mse/variance/width90保留原尺度。完整ES覆盖该视图所有时间/空间，不由逐lead ES平均替代。时间均值场CRPS先对每条路径沿物理时间平均，再评价。

S相对B的轨迹配对差（正值为S分数更大，98.333%区间）：

'''+mdtable(baselinepair,[('split','划分'),('k','k'),('view','视图'),('metric','指标'),('mean','S−B'),('ci98_333_low','下界'),('ci98_333_high','上界')])
    report+='''
适配器−S、C_mask−C_aug、C_unc−C_mask、适配器−B以及direct适配器−B的全部seed/policy/原轨迹/Re/通道区间见paired_*。step0相对S和臂间差为0意味着未获经验增益，不能当作统计功效证明信息无用。选择适配器相对B的tail/joint差就是上表的S−B，直接预测差为0；不能宣称原始20→20预测任务改善。

## 接缝、多样性、时间结构与分布诊断

'''
    seam=[r for r in quality if r['view']=='joint']
    report+=mdtable(seam,[('split','划分'),('k','k'),('arm','组'),('seam_rms','接缝增量RMS'),('seam_increment_mse','增量对真值MSE'),('parent_seam_rms','父自身增量RMS'),('acf_rmse','ACF误差'),('temporal_corr_difference_B','相对B时间相关差')])
    report+='''
接缝是同成员生成前缀末帧到子首帧的标准化增量；与真值增量误差及父自身k−1→k增量共同解释，不要求独立成员逐点相等。时间结构同时保存u/v各lag ACF、逐成员中心化20×20时间协方差/相关及B/C独立采样对照；矩阵文件在temporal缓存，经manifest索引，评分表保留Frobenius差。

冻结选择为step0时，成员方差、90%宽度和相关结构按构造保持原生S/B，说明回退安全，不证明学到非零适配器同时提升质量与多样性。非零候选在开发表有spread/width代价，但并未以未选网络构造测试改善结论。32成员和固定时间投影限制高维依赖识别。

重启分布指标为独立ED_U(B_tail,预测_tail)−ED_U(B_tail,C_tail)，保留负数；B/C独立于父和子。原生B对自身的U留空，耦合的适配/S不套独立U。全指标见scores_*及paired_*；分布距离不能替代真值评分，未要求独立随机路径相等。

## 八个科学问题

1. **同等适配训练是否足够改善重启？** 在本次1760参数、1000步完整集合CRPS协议下，开发选择未获得可用增益；不支持“已经足够”。这限定于该架构/数据/优化预算。
2. **来源标记是否有独立收益？** 正式科学选择未超过C_aug；开发同一步比较存在的小差异仅属探索，不满足预设跨测试/后续时段/种子/Re稳定增量标准，不能宣称来源条件有效。
3. **父离散度是否有额外价值？** 正式选择未超过C_mask；未证实增量。不能以显式输入了std就宣称网络利用了有益不确定性机制。
4. **完整概率质量与必要点指标是否同时改善？** 没有选出的非零网络实现全面提升；保护回退保留原生指标。所有CRPS/ES/MSE/SMSE取舍及不合格保护项均披露。
5. **是否保留多样性与时间相关？** 正式回退与原生逐值一致，因此保留；这不是学得的结构改善，也不能从方差非零推出所有时空相关准确。
6. **是否跨Re、通道、种子稳定？** 没有稳定正收益。保留三种子、u/v、三个Re、逐轨迹/窗口/lead表；种子及像素没有被当作独立物理样本。
7. **是否优于B，还是仅改善S？** 选择模型重启等于S，连改善S也未实现；与B差异按对应tail/joint正式表报告。全真实历史直接预测等于B。不能把两次生成成本称为单次原生任务的同预算改进。
8. **是否有足够证据启动物理时间一致性训练？** 没有。上一轮JVP仅说明部分输入响应可局部近似，不证明一致性loss的收益，本轮没有新增JVP目标或一致性正则。固定九次结果结束，不自动扩容、加步、加loss或选测试集。

## 统计单位、成本与可复核性

每条原轨迹内先平均窗口与两随机重复；每seed单报，mean3先在同一原轨迹内平均三个训练seed，再对12轨迹等权bootstrap10000次seed123。95%和三k Bonferroni98.333%区间全部保留，不把窗口、成员、像素或训练seed当独立轨迹。不声称所有指标/Re/通道的全族同时覆盖。

'''
    report+=f"有效训练+开发评价{train['seconds']:.3f}秒（含数据预载入时间另见training_process_start），正式生成/评分{cost['wall_seconds']:.3f}秒；预检{pre['seconds']:.3f}秒。有效训练峰值allocated显存{train['peak_memory_allocated']/2**30:.3f}GiB，正式评价{cost['peak_memory_allocated']/2**30:.3f}GiB。有效新缓存{cost['cache_bytes']:,}bytes={cost['cache_bytes']/2**30:.3f}GiB。\n\n"
    report+=f"新增原生extension路径{cost['native_extension_full20_paths']}，复用纠错批次冻结原生路径{cost['reused_native_extension_full20_paths']}；有效正式新增非零适配路径{cost['adapter_full20_paths']}（step0直接复用）。数据/原缓存预载、权重加载、评分、写盘与生成不可混作部署延迟。每个重启预测仍需要父+子两次四步，共8个网络前向，单次原生B只需4次；三k可共享父。训练的9000步完整32成员四步前向及反向成本不等于9000次推理。\n\n"
    report+=f"额外作废批次训练+开发{bad['seconds']:.3f}秒、9000更新和45checkpoint，已完成{invalid_done}个评价组、生成{invalid_native}组原生extension；其最后进度/中止堆栈见invalid_direct_training/evaluate.log。所有这部分成本都不能隐藏在有效实验的预算或部署时延中，计算总开销见RESOURCE_ACCOUNTING.json。原生缓存可重用，但错误适配实验不作为正式证据。\n\n"
    report+=f"VERIFICATION通过：有效9000更新、45checkpoint和18选择复算，528组身份/输入/noise及先发布后真值事件，所有{ver['all_lead_rows_checked']:,}个lead明细索引/归约，9组固定轨迹/Re/窗口的{ver['audit_score_rows']}评分复算（最大绝对差{ver['audit_score_maxabs']:.3g}），{ver['crps_bootstrap_intervals_checked']}个CRPS轨迹区间复核。旧结果与checkpoint哈希保持不变。审计子集用于数值核验，不替代完整528组研究统计。\n\n"
    report+='''复跑命令见research/provenance_uncertainty_conditioning/README.md；实际执行命令见EXECUTED_COMMANDS.json；文件清单及hash见REVIEW_FILES.md/ARTIFACT_SHA256.json。大型原生与时间结构缓存留Workspace/runs，45个小适配checkpoint、日志、压缩明细、报告与全部hash随Git交付。

## 最终区分与停止

**执行完成：** 符合训练k约束的固定九次实验、两类冻结选择、全部正式窗口、统计、数值/身份复核完成；作废批次单独归档。

**经验收益：** 当前固定实验未获得可选择的重启CRPS改善，正式模型回退原生。没有部署改进结论。

**机制支持：** 来源/离散度的增量价值未得到稳定支持；共同训练数据/容量控制使比较可解释，但零选择不能识别信息本身是否有用。

**未识别因素：** 轻量历史扰动参数化、固定步数/学习率、训练窗口覆盖、模型表达与历史截断可能影响结果；本轮不通过改超参补救，也不据负结果证明PDE机制。

**仍需独立验证：** 若未来另立项，需要新的独立原轨迹/Re及独立选择过程；既有历史测试/延伸时段不能提供新盲测确认。本轮按协议结束，不自动开始下一研究阶段。
'''
    (OUT/'DECISION_REPORT.md').write_text(report)
    print('WROTE Chinese report from verified evidence; all selections zero:',allzero)

if __name__=='__main__':main()
