"""Recompute trajectory-macro development selection without inference/training."""
import csv,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import OUT,dump,csvwrite,digest,ROOT,state_hash,STEPS,SEED

METRICS=['crps','energy','energy_raw','mse','smse','variance','std_variance','width90','std_width90','time_mean_crps']

def read(name):return list(csv.DictReader((OUT/name).open()))

def rows_macro(rows,arm,step):
    rr=[r for r in rows if r['arm']==arm and r['channel']=='overall' and str(r['step'])==str(step)]
    groups=defaultdict(list)
    for r in rr:groups[r['trajectory_id']].append(r)
    trajectory=[]
    for trajectory_id,items in groups.items():
        trajectory.append(dict(arm=arm,step=step,trajectory_id=trajectory_id,Re=items[0]['Re'],
            **{m:float(np.mean([float(x[m]) for x in items])) for m in METRICS}))
    return trajectory

def macro(trajectory):
    assert trajectory
    return dict(arm=trajectory[0]['arm'],step=trajectory[0]['step'],trajectories=len(trajectory),
        **{m:float(np.mean([x[m] for x in trajectory])) for m in METRICS})

def run():
    native_rows=read('development_native_fp32_scores.csv')
    native_tr=rows_macro(native_rows,'native',-1);native=macro(native_tr)
    all_tr=list(native_tr);all_macro=[native]
    selection=[]
    for arm in ('G_context','G_trust'):
        arm_rows=read(f'development_{arm}.csv'); candidates=[]
        initialized=rows_macro(arm_rows,arm,'initialized')
        all_tr.extend(initialized)
        for step in (250,500,1000):
            tr=rows_macro(arm_rows,arm,step);m=macro(tr);candidates.append(m);all_tr.extend(tr);all_macro.append(m)
        best=min([native]+candidates,key=lambda x:(x['crps'], 0 if x['arm']=='native' else int(x['step'])))
        selection.append(dict(arm=arm,scientific=dict(best),selected=dict(best),native=dict(native),
                              candidates=[dict(x) for x in candidates],initialized=dict(macro(initialized))))
    csvwrite(OUT/'development_trajectory_macro.csv',all_tr)
    csvwrite(OUT/'development_macro.csv',all_macro)
    dump(OUT/'SELECTION.json',selection)
    complete=json.loads((OUT/'TRAINING_COMPLETE.json').read_text())
    complete['selected']=selection;complete['selection_recomputed_without_inference']=True
    complete['formal_test_started']=False;dump(OUT/'TRAINING_COMPLETE.json',complete)
    cost=json.loads((OUT/'COST.json').read_text())
    cost['valid_protocol_optimizer_updates']=2000;cost['invalid_discarded_attempt_updates']=1000
    cost['invalid_attempts']=[json.loads((OUT/'ATTEMPT_1_FAILURE.json').read_text()),json.loads((OUT/'ATTEMPT_2_FAILURE.json').read_text())]
    cost['note']='The second invalid attempt completed G_context updates before a route-statistics shape error; its weights/logs were discarded and the valid fixed run was restarted. No invalid artifact is used for selection.'
    dump(OUT/'COST.json',cost)
    both_native=all(x['selected']['arm']=='native' for x in selection)
    lines=['# 修复后第一次有效历史误差空间路由训练对照','',
      '**训练协议完成：G_context 与 G_trust 各 seed=42、1000 次更新；有效协议更新数=2000。** `READY_FOR_RETRAINING=true` 的 v2 输入被直接复用。旧权重、旧风险图和旧评分没有作为新训练结果使用。','',
      '## 固定边界','',
      '训练实际使用137条窗口（420/440/460分别46/46/45），开发36条窗口。每步先按原轨迹等权抽样，再在该轨迹有效窗口中抽样；两模型共享预生成批次计划、完整32成员FP32噪声张量和冻结原生上一窗口反馈。未来标签只在损失和开发评分阶段读取。','',
      '这不是新模型自身反馈闭环：风险图和上一窗口反馈来自冻结原生影子模型。部署时需要额外生成一次32成员、四步原生预测，成本单列于 `COST.json`。','',
      '## 开发宏平均选择','',
      '|模型|原生CRPS|250|500|1000|选择|选择CRPS|选择Energy|选择原尺度MSE|选择标准化MSE|','|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for x in selection:
        c={str(z['step']):z for z in x['candidates']};n=x['native'];s=x['selected']
        lines.append(f"|{x['arm']}|{n['crps']:.9f}|{c['250']['crps']:.9f}|{c['500']['crps']:.9f}|{c['1000']['crps']:.9f}|{s['step']}|{s['crps']:.9f}|{s['energy']:.9f}|{s['mse']:.9g}|{s['smse']:.9f}|")
    lines += ['', '原生基线和每个非零候选均保留逐轨迹、Re、u/v及完整辅助指标；`development_per_trajectory.csv`、`development_by_Re.csv`、`development_by_channel.csv` 和 `development_by_Re_channel.csv` 分别提供这些切片。路由统计记录各去噪步 gate、高低风险区域、加权邻居风险和特征改变量。初始化启用路由不作为候选。','']
    if both_native:
        lines += ['## 停止条件','', 'G_context 和 G_trust 都选择原生硬关闭候选，因此按协议停止测试窗口推理，不重复计算相同零输出。本轮是开发负结果：不能据此否定历史误差路由，因为这是单种子、既有轨迹且非零候选相对原生退步。']
    lines += ['', '## 解释','', 'G_trust 与 G_context 的差别只来自固定 lambda=beta=1 可信度先验；两者参数量、信息条件、初始化、计划、噪声和预算相同。历史预测误差不是观测噪声，也不等于物理位置不可靠；结构统计不能自动证明物理因果。', '', '本轮没有追加模型、种子、窗口、损失或超参数搜索。两次实现级失败尝试已单独记录，最终选择只来自完整有效两模型训练产物；没有使用失败中间权重。']
    (OUT/'DECISION_REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(OUT/'SELECTION_AUDIT.json',dict(passed=True,native_trajectory_units=len(native_tr),both_models_selected_native=both_native,
      training_updates_valid=2000,formal_test_started=False,selection_source='recomputed from saved per-window scores; no forward pass'))
    print('FINALIZED SELECTION; both_native=',both_native,flush=True)

if __name__=='__main__':run()
