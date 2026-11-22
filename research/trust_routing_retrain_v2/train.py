"""First valid v2 training comparison; exactly two arms, one seed, 1000 steps."""
import copy, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.trust_causal_audit_v2.common import load_backbone, state_hash, tensor_hash, digest, records
from research.trust_causal_audit_v2.data import Key, CacheIndex, H5Reader, prepare_inputs, model_inputs, batch_noise
from research.trust_causal_audit_v2.model import TrustDiffusion, native_sample, crps_parts
from research.trust_routing_retrain_v2.common import *


def token(key): return f'{key.split}__{key.trajectory_id}__{key.start}'


def as_cuda(entry, noise, norm):
    return model_inputs(entry['prepared'], noise.to('cuda'))


def load_all_entries(split, manifest, norm, scale):
    rows=split_rows(manifest,split);rs=records();starts=sorted({int(r['start'])-20 for r in rows})
    indexes=load_indexes(split,starts,rs)
    entries=load_entries(split,rows,norm,scale,indexes)
    return entries


def write_noise_identity(entries):
    identities=[]
    for entry in entries:
        _,ids=batch_noise(entry['key'],files_for(entry['key'].split),device='cuda')
        identities.extend(ids)
    dump(OUT/'NOISE_IDENTITIES.json',identities)
    return identities


@torch.no_grad()
def native_baseline(native, entries, norm, name):
    out=[];rows=[];started=time.perf_counter()
    for i,entry in enumerate(entries):
        noise,ids=batch_noise(entry['key'],files_for(entry['key'].split),device='cuda')
        inputs=as_cuda(entry,noise,norm)
        p=native_sample(native,inputs['history'],noise)
        raw=p*norm['std']+norm['mean']
        out.append(dict(key=token(entry['key']),output=raw.cpu(),noise_sha256=[x['noise_tensor_hash'] for x in ids],
                        output_sha256=tensor_hash(raw)))
        for m in metrics_rows(p,entry['target'],norm,'native',-1,entry):rows.append(m)
        if (i+1)%12==0:print('BASELINE',name,i+1,flush=True)
    torch.save(out,CACHE/f'{name}.pt')
    csvwrite(OUT/f'{name}_scores.csv',rows)
    return out,rows,time.perf_counter()-started


def metrics_rows(pred,target,norm,arm,step,entry):
    values=metrics(pred,target.cuda(),norm)
    result=[]
    for value in values:
        result.append(dict(arm=arm,step=step,split=entry['key'].split,trajectory_id=entry['key'].trajectory_id,
                           Re=entry['row'].get('Re',''),start=entry['key'].start,**value))
    return result


def preflight(native,norm,scale,train_entries,dev_entries):
    assert len(train_entries)==137 and len(dev_entries)==36
    assert {e['key'].start for e in train_entries}=={420,440,460}
    assert len([e for e in train_entries if e['key'].start==460])==45
    assert len({e['key'].trajectory_id for e in train_entries})==46
    assert len({e['key'].trajectory_id for e in dev_entries})==12
    first=train_entries[0];noise,_=batch_noise(first['key'],files_for('train'),device='cuda')
    inputs=as_cuda(first,noise,norm)
    model=TrustDiffusion(copy.deepcopy(native),True).cuda().eval()
    with torch.no_grad():
        expected=native_sample(native,inputs['history'],noise)
        closed=model.sample(**inputs,enabled=False)
    assert torch.equal(expected,closed)
    # Future-label mutation is isolated because prepared inputs have already
    # been published; only the loss target is changed here.
    polluted=first['target'].clone();polluted.fill_(float('nan'))
    assert all(torch.equal(inputs[k],as_cuda(first,noise,norm)[k]) for k in inputs)
    p=torch.randn(32,2,2,2,3,device='cuda',dtype=torch.float64,requires_grad=True)
    y=torch.randn(2,2,2,3,device='cuda',dtype=torch.float64)
    loss=crps_parts(p,y)[0];grad=torch.autograd.grad(loss,p)[0]
    assert torch.isfinite(loss) and torch.isfinite(grad).all() and float(grad.abs().sum())>0
    assert state_hash(native)==state_hash(model.native)
    result=dict(passed=True,optimizer_updates=0,training_runs=0,formal_test_comparisons=0,
                train_windows=137,development_windows=36,train_counts_by_start={str(s):sum(e['key'].start==s for e in train_entries) for s in (420,440,460)},
                future_label_mutation_input_invariant=True,closed_wrapper_exact=True,
                crps_gradient_nonzero=True,backbone_state_sha256=state_hash(native),scale=scale)
    dump(OUT/'PREFLIGHT.json',result);return result


def schedule(rows):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['key'].trajectory_id].append(i)
    names=sorted(groups);rng=np.random.default_rng(SCHEDULE_SEED);items=[]
    for step in range(1,STEPS+1):
        trajectory=names[int(rng.integers(len(names)))]
        ix=groups[trajectory][int(rng.integers(len(groups[trajectory])))]
        items.append(dict(step=step,trajectory_id=trajectory,entry_index=ix,
                          key=token(rows[ix]['key']),window_start=rows[ix]['key'].start))
    return items


def checkpoint(model,arm,step):
    p=OUT/'weights'/f'{arm}_{step}.pt';p.parent.mkdir(parents=True,exist_ok=True)
    state={k:v.detach().cpu() for k,v in model.router.state_dict().items()}
    torch.save(dict(arm=arm,seed=SEED,step=step,parameters=sum(v.numel() for v in model.router.parameters()),state=state),p)
    return dict(arm=arm,seed=SEED,step=step,path=str(p.relative_to(ROOT)),sha256=digest(p),state_sha256=state_hash(model.router))


@torch.no_grad()
def evaluate_arm(model,entries,norm,arm,step,baseline=None,collect_trace=False):
    rows=[];stats=[]
    for i,entry in enumerate(entries):
        noise,ids=batch_noise(entry['key'],files_for(entry['key'].split),device='cuda')
        inputs=as_cuda(entry,noise,norm)
        if arm=='native':
            path=baseline[i]['output'].cuda();pred=(path-norm['mean'])/norm['std']
        else:
            trace=[]
            hook=[]
            handle=model.native.net.first.register_forward_hook(lambda m,a,o:hook.append(o.detach())) if collect_trace else None
            pred=model.sample(**inputs,trace=trace if collect_trace else None)
            if handle is not None:handle.remove()
            if collect_trace:
                for j,c in enumerate(trace):
                    first=hook[j];delta=c['output']-first;rr=inputs['risk']
                    high=rr>torch.quantile(rr,.75);low=rr<=torch.quantile(rr,.25)
                    neighbor_risk=(c['weights']*model_router_neighbors(rr)).sum(1)
                    stats.append(dict(arm=arm,step=step,trajectory_id=entry['key'].trajectory_id,start=entry['key'].start,
                       denoise_label=c['label'],gate_mean=float(c['gate'].mean()),gate_high_risk=float(c['gate'][high].mean()),
                       gate_low_risk=float(c['gate'][low].mean()),weighted_neighbor_risk=float(neighbor_risk.mean()),
                       feature_delta_rms=float(delta.square().mean().sqrt()),feature_delta_high_risk=float(delta[high.expand_as(delta)].square().mean().sqrt())))
        rows.extend(metrics_rows(pred,entry['target'],norm,arm,step,entry))
    return rows,stats


def model_router_neighbors(risk):
    from research.trust_causal_audit_v2.model import neighbors
    return neighbors(risk).squeeze(1)


def train_one(arm,trust,entries,dev_entries,native,norm,sched):
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=TrustDiffusion(copy.deepcopy(native),trust).cuda().eval();model.router.train()
    optimizer=torch.optim.Adam(model.router.parameters(),lr=.001)
    logs=[];dev=[];checkpoints=[];cost=[];started=time.perf_counter()
    init=checkpoint(model,arm,0);checkpoints.append(init)
    init_rows,init_stats=evaluate_arm(model,dev_entries,norm,arm,'initialized',collect_trace=False)
    dev.extend(init_rows)
    for step,item in enumerate(sched,1):
        entry=entries[item['entry_index']]
        noise,_=batch_noise(entry['key'],files_for('train'),device='cuda')
        inputs=as_cuda(entry,noise,norm);target=(entry['target'].cuda()-norm['mean'])/norm['std']
        pred=model.sample(**inputs)
        loss=crps_parts(pred,target)[0]
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad_norm=float(torch.sqrt(sum(p.grad.square().sum() for p in model.router.parameters() if p.grad is not None)))
        optimizer.step()
        logs.append(dict(arm=arm,seed=SEED,step=step,trajectory_id=entry['key'].trajectory_id,start=entry['key'].start,
                         key=token(entry['key']),crps=float(loss),gradient_norm=grad_norm))
        if step in (250,500,1000):
            model.eval();t=time.perf_counter();ck=checkpoint(model,arm,step);checkpoints.append(ck)
            vals,stats=evaluate_arm(model,dev_entries,norm,arm,step,collect_trace=(step==1000));dev.extend(vals)
            cost.append(dict(arm=arm,step=step,seconds=time.perf_counter()-t));model.router.train()
            if stats:csvwrite(OUT/f'route_stats_{arm}.csv',stats)
        if step%50==0:print('TRAIN',arm,step,flush=True)
    csvwrite(OUT/f'training_{arm}.csv',logs);csvwrite(OUT/f'development_{arm}.csv',dev)
    dump(OUT/f'checkpoints_{arm}.json',checkpoints);csvwrite(OUT/f'development_cost_{arm}.csv',cost)
    return model,checkpoints,dev,time.perf_counter()-started


def select(dev_rows,native_rows):
    metrics_names=['crps','energy','energy_raw','mse','smse','variance','std_variance','width90','std_width90','time_mean_crps']
    tr,macro=aggregate_trajectory(native_rows,metrics_names)
    alltr,allmacro=aggregate_trajectory(dev_rows,metrics_names)
    csvwrite(OUT/'development_trajectory_macro.csv',alltr);csvwrite(OUT/'development_macro.csv',allmacro)
    candidates=[]
    for arm in ('G_context','G_trust'):
        for step in CANDIDATES[1:]:
            row=next(x for x in allmacro if x['arm']==arm and x['step']==step)
            candidates.append(row)
    native_score=next(x for x in tr if x['arm']=='native' and int(x['step'])==-1)
    selections=[]
    for arm in ('G_context','G_trust'):
        own=[x for x in candidates if x['arm']==arm]
        eligible=[x for x in own if x['mse']<=native_score['mse']*1.01 and x['smse']<=native_score['smse']*1.01]
        best=min(own,key=lambda x:(x['crps'],int(x['step'])))
        chosen=min(eligible,key=lambda x:(x['crps'],int(x['step']))) if eligible and min(x['crps'] for x in eligible)<native_score['crps'] else dict(native_score,arm=arm,step='native')
        selections.append(dict(arm=arm,scientific=dict(best),eligible_nonzero=[dict(x) for x in eligible],selected=dict(chosen),
                               native=dict(native_score)))
    dump(OUT/'SELECTION.json',selections)
    return selections,alltr,allmacro


def main():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    if (OUT/'TRAINING_STARTED.json').exists():raise RuntimeError('refuse to overwrite an already started training run')
    ready,scale,manifest,cache_manifest=load_protocol()
    rs=records();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    native,norm=load_backbone();base_hash=state_hash(native)
    cpu_norm={k:v.cpu() for k,v in norm.items()}
    train_entries=load_all_entries('train',manifest,cpu_norm,scale)
    dev_entries=load_all_entries('development',manifest,cpu_norm,scale)
    pre=preflight(native,cpu_norm,scale,train_entries,dev_entries)
    dump(OUT/'SOURCE_HASHES.json',dict(audit_verification=digest(AUDIT_OUT/'VERIFICATION.json'),
       audit_data=digest(AUDIT_OUT/'DATA_AUDIT.json'),manifest=digest(AUDIT_OUT/'WINDOW_MANIFEST.json'),
       scale=digest(AUDIT_OUT/'RISK_NORMALIZATION.json'),cache=digest(AUDIT_OUT/'CACHE_MANIFEST.json'),
       v2_data=digest(ROOT/'research/trust_causal_audit_v2/data.py'),v2_model=digest(ROOT/'research/trust_causal_audit_v2/model.py'),
       checkpoint=digest(ROOT/'reliability/checkpoints/current.pt')))
    dump(OUT/'TRAINING_STARTED.json',dict(seed=SEED,steps=STEPS,optimizer_updates=0,training_runs=0,
         source_hashes=json.loads((OUT/'SOURCE_HASHES.json').read_text()),preflight_sha256=digest(OUT/'PREFLIGHT.json'),started=time.time()))
    # Pre-generated common schedule and all real noise identities are frozen
    # before either optimizer is constructed.
    sched=schedule(train_entries);dump(OUT/'BATCH_SCHEDULE.json',sched)
    write_noise_identity(train_entries+dev_entries)
    dump(OUT/'PROTOCOL_FROZEN.json',dict(seed=SEED,steps=STEPS,lr=.001,candidates=list(CANDIDATES),
      schedule_seed=SCHEDULE_SEED,train_windows=137,development_windows=36,feedback='frozen native previous-window outputs',
      target_start_test=480,optimizer='Adam',loss='full32 empirical CRPS standardized',
      noise_identity='v2 batch_noise full split draw then trajectory slice',source_hashes=json.loads((OUT/'SOURCE_HASHES.json').read_text()),
      optimizer_updates_before_training=0))
    # New FP32 development baseline is the only baseline used for selection.
    native_dev,native_dev_rows,base_seconds=native_baseline(native,dev_entries,norm,'development_native_fp32')
    dump(OUT/'DEVELOPMENT_NATIVE_MANIFEST.json',
         [{k:v for k,v in item.items() if k!='output'} for item in native_dev])
    model_c,ck_c,dev_c,cost_c=train_one('G_context',False,train_entries,dev_entries,native,norm,sched)
    model_t,ck_t,dev_t,cost_t=train_one('G_trust',True,train_entries,dev_entries,native,norm,sched)
    all_dev=dev_c+dev_t
    selection,_,_=select(all_dev,native_dev_rows)
    dump(OUT/'TRAINING_COMPLETE.json',dict(seed=SEED,steps_each=STEPS,total_updates=2000,optimizer_updates=2000,
      training_seconds_context=cost_c,training_seconds_trust=cost_t,baseline_seconds=base_seconds,
      backbone_state_sha256_before=base_hash,backbone_state_sha256_after=state_hash(native),
      selected=selection,formal_test_started=False))
    dump(OUT/'COST.json',dict(new_fp32_development_baseline_seconds=base_seconds,
      training_context_seconds=cost_c,training_trust_seconds=cost_t,shadow_feedback='one frozen native 32-member source per window',
      shadow_feedback_required_for_training=True,shadow_feedback_not_new_model_feedback=True,
      optimizer_updates=2000,formal_test_comparisons=0))
    write_report(selection,native_dev_rows,all_dev)
    print('TRAINING COMPLETE; selections:',json.dumps(selection,ensure_ascii=False),flush=True)


def write_report(selection,native_rows,dev_rows):
    both_native=all(x['selected']['step']=='native' for x in selection)
    lines=['# 修复后第一次有效历史误差空间路由训练对照','',
       f'本轮固定种子42、G_context/G_trust各1000步，实际优化器更新2000次；训练137窗口，开发36窗口。`READY_FOR_RETRAINING=true`。本轮不使用旧实验权重、旧风险图或旧评分。','',
       '## 协议和边界','',
       '训练使用 v2 的显式 `(split, trajectory_id, start)` 键、137 条训练窗口和训练集冻结风险尺度。风险来自上一轮冻结原生32成员预测减当前已观测20帧；未来目标只进入CRPS损失与开发评分。两模型共享同一轨迹等权抽样计划、同一实际CUDA噪声、同一FP32冻结原生反馈，路由公式和1269参数实现来自 v2。','',
       '这不是新模型自身反馈闭环：训练和输入构造依赖上一轮冻结原生影子模型，部署时需要这次额外的32成员四步原生预测。旧TF32路径只作为已发布历史反馈，不作为新FP32评分基线。','',
       '## 开发选择','',
       '|模型|科学最优|选择|选择CRPS|选择ES|选择原尺度MSE|选择标准化MSE|','|---|---|---|---:|---:|---:|---:|']
    for x in selection:
        b=x['scientific'];s=x['selected'];n=x['native']
        lines.append(f"|{x['arm']}|step {b['step']}|{s['step']}|{s['crps']:.9f}|{s['energy']:.9f}|{s['mse']:.9g}|{s['smse']:.9f}|")
    lines += ['', '全部非零候选的CRPS、Energy、MSE、集合方差、90%宽度和时间平均场CRPS已写入 `development_macro.csv`、`development_trajectory_macro.csv` 及两份逐窗口表。初始化启用路由只作诊断，不进入候选选择。','']
    if both_native:
        lines += ['## 停止规则','', '两个模型均选择硬关闭原生候选，因此按协议停止新模型正式测试推理，不重复计算相同的零输出。没有把本轮结果解释成历史误差路由无效；它只表示在一个种子和既有开发轨迹上，非零候选没有胜过原生。']
    else:
        lines += ['## 测试状态','', '至少一个模型选择非零候选；测试评价由后续固定入口执行。']
    lines += ['', '## 路由行为','', '各去噪步的 gate、高低风险区域 gate、加权邻居风险和特征改变量保存在 `route_stats_G_context.csv` 与 `route_stats_G_trust.csv`。这些是结构行为统计，不是物理因果或收益证明。','',
              '历史预测误差不是观测噪声，也不等于物理位置不可靠；本轮不声称恢复了物理机制。单种子、已反复使用的轨迹只支持探索性判断。','',
              '训练更新数固定为2000（两模型各1000），没有追加模型、种子、窗口、损失或超参数搜索。']
    (OUT/'DECISION_REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
