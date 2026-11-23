"""Fixed C0/C1 run; no retries after any optimizer update."""
import time
from collections import defaultdict
from .core import *
from research.trust_causal_audit_v2.model import native_sample


def make_noise(entries):
    cache={};manifest=[];rs=records()
    for item in entries:
        key=item['key'];group=(key.split,key.trajectory_id)
        if group not in cache:
            z,ids=batch_noise(key,[r['file'] for r in rs[key.split]])
            cache[group]=z.cpu()
        else:
            ids=[dict(split=key.split,trajectory_id=key.trajectory_id,history_start=key.start,member_id=m,noise_seed=m,
                noise_tensor_hash=tensor_hash(cache[group][i]),convention='original_full_split_batch_draw_then_identity_slice') for i,m in enumerate(MEMBERS)]
        for identity in ids:
            identity.pop('batch_files',None)
            manifest.append(identity)
    return cache,manifest


def schedule(entries):
    groups=defaultdict(list)
    for i,e in enumerate(entries):groups[e['key'].trajectory_id].append(i)
    names=sorted(groups);rng=np.random.default_rng(20261123);plan=[]
    for step in range(1,501):
        name=names[int(rng.integers(len(names)))];i=groups[name][int(rng.integers(len(groups[name])))]
        ids=rng.choice(32,size=8,replace=False).tolist()
        plan.append(dict(step=step,index=i,key=entries[i]['cache_key'],trajectory_id=name,start=entries[i]['key'].start,
                         member_indices=ids,member_ids=[MEMBERS[j] for j in ids]))
    return plan


def checkpoint(model,arm,step):
    path=OUT/'weights'/f'{arm}_{step}.pt';path.parent.mkdir(exist_ok=True)
    assert not path.exists()
    torch.save(dict(arm=arm,step=step,correction={k:v.detach().cpu() for k,v in model.correction.state_dict().items()},
                   out={k:v.detach().cpu() for k,v in model.native.net.out.state_dict().items()}),path)
    return dict(arm=arm,step=step,path=str(path.relative_to(ROOT)),sha256=digest(path),out_state_hash=state_hash(model.native.net.out),
                correction_state_hash=state_hash(model.correction))


def freeze_hash(native,exclude_out=False):
    value={k:tensor_hash(v) for k,v in native.state_dict().items() if not exclude_out or not k.startswith('net.out.')}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def evaluate(native,model,arm,step,entries,scales,noise,store=False,enabled=True,ablate=None):
    rows=[];amplitudes=[];outputs=[];started=time.perf_counter()
    for item in entries:
        key=item['key'];z=noise[(key.split,key.trajectory_id)].cuda()
        inputs=condition_inputs(item,scales,z,'C0' if arm=='native' else arm)
        if ablate=='E':inputs['E']=torch.zeros_like(inputs['E'])
        if ablate=='W':inputs['W']=torch.zeros_like(inputs['W']);inputs['W_mean']=torch.zeros_like(inputs['W_mean'])
        trace=[]
        p=native_sample(native,inputs['history'],z) if arm=='native' else model.sample(**inputs,enabled=enabled,trace=trace)
        # Predictions are generated and hashed before target is fetched for score.
        output_hash=tensor_hash(p)
        target=item['target']
        for r in score(p,target,{k:v.cuda() for k,v in native_norm().items()}):
            rows.append(dict(arm=arm,step=step,trajectory_id=key.trajectory_id,Re=item['Re'],start=key.start,
                             output_sha256=output_hash,**r))
        for call in trace:
            amplitudes.append(dict(arm=arm,step=step,trajectory_id=key.trajectory_id,start=key.start,label=call['label'],
                 delta_rms=float(call['delta'].double().square().mean().sqrt()),member_std=float(p.std(0,unbiased=False).mean())))
        if store:outputs.append(dict(key=item['cache_key'],paths=p.cpu(),sha256=output_hash))
    if store:torch.save(outputs,CACHE/f'{entries[0]["key"].split}_{arm}_{step}_paths.pt')
    return rows,amplitudes,time.perf_counter()-started


def native_norm():
    ck=torch.load(ROOT/'reliability/checkpoints/current.pt',map_location='cpu',weights_only=False)
    return {k:v.reshape(2,1,1,1) for k,v in ck['normalization'].items()}


def report(rows,selection,cost,training,test_rows=None):
    tr,macro=trajectory_summary(rows)
    n=next(r for r in macro if r['arm']=='native' and r['channel']=='overall')
    lines=['# 残差与XPF不确定性条件的干净状态去噪修正','',
       f"有效更新：C0/C1各{training['steps_each']}步，共{training['updates']}；seed42，训练8成员fair CRPS，开发32成员经验CRPS。新增G6392参数，只有原生net.out的17320参数解冻。",'',
       '|模型|候选|开发CRPS|32 fair CRPS|路径ES|原MSE|SMSE|标准化方差|90%标准化宽度|',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in macro:
        if r['channel']=='overall':lines.append(f"|{r['arm']}|{r['step']}|{r['crps']:.9f}|{r['fair_crps']:.9f}|{r['energy']:.9f}|{r['mse']:.9g}|{r['smse']:.9f}|{r['std_variance']:.9f}|{r['std_width90']:.9f}|")
    lines+=['','科学选择：'+json.dumps(selection,ensure_ascii=False),'',
      'E是上一冻结原生32成员预测减当前已观测历史的完整有符号残差；W是冻结XPF16对当前原生零噪声预测的原始残差分位半宽。W训练固定尺度和广播均值保留整体风险，不做窗口内均值/方差归一化；不使用在线校准。', '',
      '所有条件均来自发布时已知历史和冻结影子模型；每个99/66/33/0去噪调用使用base_clean+Delta继续原采样。C0与C1同结构/同初始化/同out训练范围/同实际噪声，C0的E/W槽零。未采用空间邻域凸组合或最终共同平移。']
    c0=next(x for x in selection if x['arm']=='C0');c1=next(x for x in selection if x['arm']=='C1')
    if all(x['chosen_arm']=='native' for x in selection):
        conclusion='情形A：两个模型均没有超过开发原生基线。固定预算的部分微调＋干净状态修正没有可选择增益；不证明E/W完全无价值。按协议停止测试推理。'
    elif c1['crps']>=c0['crps']:
        conclusion='情形B或局部结果：C1未超过C0，未观察到E+W条件超出普通修正及输出层微调的增量。各自是否优于原生须分别看科学选择。'
    else:conclusion='C1开发CRPS优于C0，但不能仅凭这一点宣称完整概率质量或E/W各自独立有效。必须同时核对ES、MSE、通道/Re及成员多样性，并用冻结测试和无训练消融辅助解释。'
    lines+=['','## 决策', '',conclusion,'',
      '完整候选逐轨迹、u/v、Re、成员距离项及四步Delta RMS见开发评分和分析表；输出层参数变化与冻结层哈希见TRAINING_COMPLETE。输入消融只在固定少量开发样本执行，关G/E/W的分布偏移不能当因果识别。','',
      '## 九个问题','',
      '1. E/W进入G输入并逐步改变干净状态；初始化最后层零时Delta为0，训练后实测幅度见表。',
      '2. 仅native.net.out.weight/bias及G训练；其余主干、归一化和XPF全部冻结。',
      '3. C1−C0由科学选择和配对表报告；E/W联合比较不能分别识别各自收益。',
      '4. 各自对完全未修改FP32 DiPred比较，不能把胜过C0等同部署收益。',
      '5. 方差/区间宽度和成员距离变化需与CRPS/全路径ES一起看，收缩本身不等于改善。',
      '6. 当前W每窗口需额外4次冻结原生调用和一个XPF头，且历史E需要上一32成员影子预测。成本尚不支持无条件部署推荐。',
      '7. 训练固定W尺度及整体均值广播已通过整体风险上升测试，没有强制风险越大修正越大。',
      '8. 8成员训练fair CRPS用K(K−1)排除自配对；32成员评价经验CRPS用K²，另报32fair敏感性；不混用不同K绝对分数。',
      '9. 一种子、旧轨迹仅探索性；收益及取舍不充分时不建议扩大主干微调。即使正面，只建议未来独立多种子确认，本轮不启动。','',
      '测试轨迹若执行均为反复使用的旧12轨迹start480，非独立盲测。GPU秒/缓存字节是计算成本，不是Codex服务费用。','',
      '成本：'+json.dumps(cost,ensure_ascii=False)]
    return '\n'.join(lines)+'\n'


def synthetic_pipeline_test():
    # Exercise actual summary, selection, score field coverage and full Chinese
    # report before the first optimizer update, including both-native branch.
    rows=[]
    for arm,step in [('native',0),('C0',250),('C0',500),('C1',250),('C1',500)]:
        for trajectory in ('a','b'):
            for start in (420,440):
                for ch in ('overall','u','v'):
                    rows.append(dict(arm=arm,step=step,trajectory_id=trajectory,Re='100',start=start,channel=ch,
                       **{m:1.+(0.1 if arm!='native' else 0) for m in METRICS}))
    sel,tr,macro=select(rows);assert all(s['chosen_arm']=='native' for s in sel)
    text=report(rows,sel,dict(test=True),dict(steps_each=500,updates=1000))
    assert '情形A' in text and '九个问题' in text and len(tr)==30
    # Exercise nonzero selection and ties too; MSE protection cannot override.
    for r in rows:
        if r['arm']=='C1':r['crps']=.9;r['mse']=100
    sel,_,_=select(rows);assert next(x for x in sel if x['arm']=='C1')['step']==250
    assert 'C1开发CRPS' in report(rows,sel,dict(test=True),dict(steps_each=500,updates=1000))
    return True
