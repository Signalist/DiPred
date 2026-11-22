"""Fixed two-arm experiment: ordinary and history-error-guided routing."""
import csv, hashlib, json, os, sys, time, copy
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reliability.networks import DynamicsDiffusion20
from research.error_guided_spatial_routing.model import TrustDiffusion,crps
from research.error_guided_spatial_routing.prepare import build as prepare_build,DEFAULT_CACHE

OUT=ROOT/"results/error_guided_spatial_routing"
CACHE=Path(os.environ.get("DIPRED_TRUST_CACHE",str(DEFAULT_CACHE)))
MEMBERS=tuple(range(500000,500032));SEED=42;STEPS=1000


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)


def native_model():
    ck=torch.load(ROOT/"reliability/checkpoints/current.pt",map_location="cuda",weights_only=False)
    m=DynamicsDiffusion20(ck["dynamics_scale"]).cuda().eval();m.load_state_dict(ck["predictor"])
    m.requires_grad_(False)
    norm={k:v.cuda() for k,v in ck["normalization"].items()}
    return m,norm,ck


def noise_for(h):
    vals=[]
    for s in MEMBERS:
        g=torch.Generator(device="cuda").manual_seed(s)
        vals.append(torch.randn(h.shape,device=h.device,dtype=h.dtype,generator=g))
    return torch.cat(vals)


def aux_for(history, previous, norm):
    # The context arm sees the same frozen previous prediction context; only
    # G_trust additionally applies the explicit risk bias to neighbor logits.
    pm=previous.mean(0)
    spread=previous.std(0,unbiased=False).mean(0)
    return torch.stack((
        ((pm[:,-1]-history[:,-1])/norm["std"][0,:,0,0,0].view(2,1,1)).mean(0),
        ((pm.mean(1)-history.mean(1))/norm["std"][0,:,0,0,0].view(2,1,1)).mean(0),
        spread.mean(0)/norm["std"][0,:,0,0,0].mean()), 0)


def crps_metrics(paths,truth,norm):
    p=paths.double();y=truth.double();std=norm["std"][0].double()
    z=p/std;zy=y/std
    e=p.mean(0)-y;ez=e/std
    d=(p-y[None]).flatten(1).square().mean(1).sqrt().mean()
    pair=(p[:,None]-p[None]).flatten(2).square().mean(2).sqrt().mean()/2
    es=d-pair
    width=torch.quantile(p,.95,0)-torch.quantile(p,.05,0)
    return {"crps":float(crps(z,zy)),"energy":float(es),"mse":float(e.square().mean()),
            "smse":float(ez.square().mean()),"variance":float(p.var(0,unbiased=False).mean()),
            "std_variance":float(z.var(0,unbiased=False).mean()),"width90":float(width.mean()),
            "std_width90":float((width/std).mean()),
            "time_mean_crps":float(crps(z.mean(2),zy.mean(1)))}


def rows_from(path): return torch.load(path,map_location="cpu",weights_only=False)["rows"]


def load_prepared(split):
    return rows_from(CACHE/f"{split}_windows.pt")


def cache_paths(split,start):
    return torch.load(CACHE/f"{split}_{start}_native.pt",map_location="cpu",weights_only=False)["paths"]


def make_batch(row, split, norm, prev_cache):
    h=(row["history"].cuda()-norm["mean"][0])/norm["std"][0]
    y=(row["truth"].cuda()-norm["mean"][0])/norm["std"][0]
    previous=prev_cache[(row["file"],row["start"]-20)].cuda()
    aux=aux_for(row["history"].cuda(),previous,norm).expand(32,-1,-1,-1)
    risk=row["risk"].cuda().expand(32,-1,-1,-1)
    return h[None].expand(32,-1,-1,-1,-1),y,aux,risk


@torch.no_grad()
def evaluate_model(model, rows, split, norm, prev_cache, step, arm_name=None):
    out=[]
    for row in rows:
        h,y,aux,risk=make_batch(row,split,norm,prev_cache); y_raw=row["truth"].cuda()
        z=noise_for(h[:1])
        if step==0: paths=cache_paths(split,row["start"])[int(row["trajectory_index"])].cuda()
        else: paths=model.sample(h,aux,risk,z)*norm["std"]+norm["mean"]
        m=crps_metrics(paths,y_raw,norm)
        out.append(dict(split=split,arm=arm_name or ("native" if step==0 else model.trust and "G_trust" or "G_context"),
                        step=step,file=row["file"],Re=row["Re"],start=row["start"],**m))
    return out


def train_arm(name, trust, train_rows, dev_rows, norm, prev_cache, native):
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=TrustDiffusion(copy.deepcopy(native).cuda(),trust).cuda()
    model.eval();model.router.train()
    opt=torch.optim.Adam(model.router.parameters(),lr=.001)
    rng=np.random.default_rng(20261122)
    schedule=rng.integers(len(train_rows),size=STEPS).tolist()
    logs=[];dev=[];weights=[];start_time=time.time()
    for step in range(0,STEPS+1):
        if step==0 or step in (250,500,1000):
            model.eval();vals=evaluate_model(model,dev_rows,"development",norm,prev_cache,step if step else 0,name)
            dev.extend(vals)
            ckpt=OUT/"weights"/f"{name}_{step}.pt";ckpt.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"arm":name,"step":step,"trust":trust,"state":model.router.state_dict()},ckpt)
            weights.append({"arm":name,"step":step,"path":str(ckpt.relative_to(ROOT)),"sha256":digest(ckpt)})
            model.router.train()
        if step==STEPS:break
        row=train_rows[schedule[step]]
        h,y,aux,risk=make_batch(row,"train",norm,prev_cache);z=noise_for(h[:1])
        pred=model.sample(h,aux,risk,z)
        loss=crps(pred,y)
        opt.zero_grad(set_to_none=True);loss.backward()
        gn=float(torch.sqrt(sum((p.grad.square().sum() for p in model.router.parameters() if p.grad is not None))))
        opt.step()
        logs.append({"arm":name,"step":step+1,"file":row["file"],"start":row["start"],"crps":float(loss),"gradient_norm":gn})
        if (step+1)%50==0: print("TRAIN",name,step+1,flush=True)
    write_csv(OUT/f"training_{name}.csv",logs);write_csv(OUT/f"development_{name}.csv",dev)
    return model,weights,dev,time.time()-start_time


def choose(dev):
    result={}
    for arm in ("G_context","G_trust"):
        vals=[r for r in dev if r["arm"]==arm]
        scores={s:float(np.mean([r["crps"] for r in vals if r["step"]==s])) for s in (0,250,500,1000)}
        result[arm]=min(scores,key=lambda s:(scores[s],s));result[arm+"_scores"]=scores
    (OUT/"development_selection.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


@torch.no_grad()
def test_and_counterfactual(models, selection, test_rows, norm, prev_cache):
    rows=[]
    for row in test_rows:
        h,y,aux,risk=make_batch(row,"test",norm,prev_cache);y_raw=row["truth"].cuda();z=noise_for(h[:1])
        native_paths=cache_paths("test",480)[int(row["trajectory_index"])].cuda()
        rows.append(dict(arm="native",step=0,file=row["file"],Re=row["Re"],start=480,**crps_metrics(native_paths,y_raw,norm)))
        for arm,step in selection.items():
            if arm not in ("G_context","G_trust"):continue
            # Step 0 is the separately reported native, fusion-off baseline;
            # do not treat a trainable router initialization as a new model.
            if int(step)==0: continue
            p=models[arm][step].sample(h,aux,risk,z)*norm["std"]+norm["mean"]
            rows.append(dict(arm=arm,step=step,file=row["file"],Re=row["Re"],start=480,**crps_metrics(p,y_raw,norm)))
            if arm=="G_trust" and step:
                shifted=risk.clone();shifted[:,:,1:]=risk[:,:,:-1]
                q=models[arm][step].sample(h,aux,shifted,z)*norm["std"]+norm["mean"]
                rows.append(dict(arm="G_trust_shifted_map",step=step,file=row["file"],Re=row["Re"],start=480,**crps_metrics(q,y_raw,norm)))
    write_csv(OUT/"test_scores.csv",rows);return rows


def report(dev,test,selection,costs):
    def mean(rs,arm,key,step=None):
        x=[float(r[key]) for r in rs if r["arm"]==arm and (step is None or int(r["step"])==int(step))]
        return float(np.mean(x)) if x else None
    lines=["# 历史预测误差指导的空间可信度路由：决策报告","",
           "本实验在 `trust` 分支固定冻结 DiPred 主干，严格按上一窗口预测→真值揭示→误差地图→下一窗口预测执行。训练只使用 46 条训练轨迹、起点 420/440/460，开发使用 12 条开发轨迹，正式评价固定为历史起点 480（500–519），属于探索性验证。","",
           "## 方法与审计","",
           "G_context 与 G_trust 使用同一 3×3 局部特征融合器、相同初始化和 Adam 预算；G_trust 仅在邻域 logits 上减去固定 λ=1 的归一化历史误差地图。融合强度由 sigmoid 限制在 0.25 内，边界使用 replicate padding。每个成员独立保持时空身份，四个去噪步骤都调用路由器。主干权重未训练。","",
           f"路由参数数目为 {json.loads((OUT/'PREFLIGHT.json').read_text())['router_parameters']}，每个模型最多 {STEPS} 次更新，训练耗时记录见 `cost.json`。开发选择：{json.dumps(selection,ensure_ascii=False)}。","",
           "## 开发与历史测试结果","",
           "|模型|开发CRPS|测试CRPS|测试Energy|测试原尺度MSE|测试标准化MSE|",
           "|---|---:|---:|---:|---:|---:|"]
    native_dev=mean(dev,"G_context","crps",0)
    lines.append(f"|native|{native_dev}|{mean(test,'native','crps')}|{mean(test,'native','energy')}|{mean(test,'native','mse')}|{mean(test,'native','smse')}|")
    for arm in ("G_context","G_trust","G_trust_shifted_map"):
        step=selection.get(arm,0)
        lines.append(f"|{arm}|{mean(dev,arm,'crps',step) if arm in selection else None}|{mean(test,arm,'crps',step)}|{mean(test,arm,'energy',step)}|{mean(test,arm,'mse',step)}|{mean(test,arm,'smse',step)}|")
    lines += ["","## 科学结论","",
              "上一窗口误差地图是近期模型失败强度的描述量，不是真实观测噪声、未来误差或物理不可靠位置。只有当 G_trust 相对 G_context 在完整 CRPS、Energy Score、MSE、成员宽度及轨迹/Re 分组上同时保持合理，才能支持显式路由的增量；单一平均分数下降不足以证明机制。错位地图结果只作实现行为诊断，不能单凭此证明因果。","",
              "本轮只有一个训练种子，历史测试已被既往研究使用；结果不能声称跨种子稳定，也不能代表新盲测。运行时需要额外一次冻结原生上一窗口预测来构造地图，成本与一次 32 成员四步原生路径生成相同。"]
    (OUT/"DECISION_REPORT.md").write_text("\n".join(lines)+"\n")


def main(mode="all"):
    OUT.mkdir(parents=True,exist_ok=True)
    if mode in ("prepare","all"): prepare_build(CACHE)
    if mode=="prepare":return
    if not (OUT/"PREFLIGHT.json").exists():
        os.system(f"{sys.executable} {Path(__file__).with_name('tests.py')}")
    native,norm,ck=native_model()
    train_rows=load_prepared("train");dev_rows=load_prepared("development");test_rows=load_prepared("test")
    # Previous native outputs are read-only and indexed by trajectory/start.
    prev_cache={}
    for split,rows in (("train",train_rows),("development",dev_rows),("test",test_rows)):
        for r in rows:
            key=(r["file"],r["start"]-20)
            if key not in prev_cache:
                allpaths=cache_paths(split,r["start"]-20)
                idx=next(i for i,q in enumerate(rows) if q["file"]==r["file"])
                prev_cache[key]=allpaths[idx]
    models={};all_dev=[];cost={}
    for name,trust in (("G_context",False),("G_trust",True)):
        m,w,d,t=train_arm(name,trust,train_rows,dev_rows,norm,prev_cache,native)
        models[name]={int(x["step"]):m for x in w};all_dev.extend(d);cost[name]=t
    sel=choose(all_dev)
    selected_models={}
    for arm,step in sel.items():
        if arm.endswith("_scores"):continue
        item=next(x for x in (OUT/"weights").glob(f"{arm}_{step}.pt"))
        m=TrustDiffusion(DynamicsDiffusion20(ck["dynamics_scale"]).cuda().eval(),arm=="G_trust").cuda();m.native.load_state_dict(ck["predictor"])
        m.router.load_state_dict(torch.load(item,map_location="cuda",weights_only=False)["state"]);m.eval();selected_models[arm]={step:m}
    test=test_and_counterfactual(selected_models,sel,test_rows,norm,prev_cache)
    write_csv(OUT/"development_all.csv",all_dev)
    (OUT/"cost.json").write_text(json.dumps(cost,indent=2)+"\n")
    report(all_dev,test,sel,cost)
    manifest={"objective":"previous-window error-guided spatial routing","seed":SEED,"steps":STEPS,
              "cache":str(CACHE),"checkpoint_sha256":digest(ROOT/"reliability/checkpoints/current.pt"),
              "protocol_sha256":digest(ROOT/"research/error_guided_spatial_routing/PROTOCOL.md"),
              "selection":sel,"test_rows":len(test),"utc":time.time()}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__": main(sys.argv[1] if len(sys.argv)>1 else "all")
