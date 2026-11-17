"""One shared monotone quantile function for regional means and maxima.

No P bypass or X reference: each arm observes only its advertised feature slots
plus the public query geometry. Ensemble paths are reused across all queries.
"""
from common import *
from torch import nn
from torch.nn import functional as TF
from functorch import combine_state_for_ensemble,vmap
import math,time,csv,argparse
ARMS=('X','P','F','XPF')
TRAIN_RECTS=[(0,32,0,64),(0,16,0,32),(0,16,32,64),(16,32,0,32),(16,32,32,64),
 (8,24,16,48),(0,32,0,32),(0,32,32,64),(0,16,0,64),(16,32,0,64),(0,8,16,48),(24,32,16,48)]
NEW_RECTS=[(4,20,8,40),(12,28,24,56),(4,28,8,56),(8,24,0,24),(8,24,40,64),(2,14,20,44),(18,30,20,44),(6,26,12,52)]
TRAIN_TIMES=[(0,5),(5,10),(10,20),(0,20)]
NEW_TIMES=[(0,10),(5,15),(10,15),(15,20)]
def queries(train=False):
    values=[]
    for category,rects,times in [('seen',TRAIN_RECTS,TRAIN_TIMES),('new_space',NEW_RECTS,TRAIN_TIMES),('new_time',TRAIN_RECTS,NEW_TIMES),('new_both',NEW_RECTS,NEW_TIMES)]:
        if train and category!='seen':continue
        for r in rects:
            for t in times:values.append(dict(id=len(values),category=category,rect=list(r),time=list(t)))
    return values

def target_queries(field,qs):
    out=[]
    for q in qs:
        a,b,c,d=q['rect'];t,u=q['time'];part=field[:,:,t:u,a:b,c:d]
        out.append(torch.cat((part.mean((2,3,4)),part.amax((2,3,4))),1))
    return torch.stack(out,1)

def pool(field,rect):
    a,b,c,d=rect;v=field[...,a:b,c:d]
    return torch.cat((v.mean((-2,-1)),v.std((-2,-1)),v.amax((-2,-1))),1)
@torch.no_grad()
def inputs(data,qs):
    x=data['x'].flatten(1,2);p=data['p'].flatten(1,2);f=data['f'];saved={};out=[]
    for q in qs:
        r=tuple(q['rect'])
        if r not in saved:saved[r]=torch.cat([pool(z,r) for z in (x,p,f)],1)
        a,b,c,d=r;t,u=q['time'];geo=x.new_tensor([a/32,b/32,c/64,d/64,t/20,u/20,(b-a)*(d-c)/2048,(u-t)/20]).expand(len(x),8)
        out.append(torch.cat((saved[r],geo),1))
    return torch.stack(out,1)
class QueryHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(392,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,28))
    def components(self,x,mask):
        slots=torch.cat((x[:,:120]*mask[0],x[:,120:240]*mask[1],x[:,240:384]*mask[2],x[:,384:]),1)
        out=self.net(slots);return out[:,:4],TF.softplus(out[:,4:].reshape(-1,4,6))/6
    def forward(self,x,mask,tau):
        center,weights=self.components(x,mask);return center+(weights*bases(tau)[:,None]).sum(2)
    def quantiles(self,x,mask,levels):
        c,w=self.components(x,mask);return c[None]+torch.einsum('bck,qk->qbc',w,bases(levels))
def mask(arm):return torch.tensor((arm in ('X','XPF'),arm in ('P','XPF'),arm in ('F','XPF')),device='cuda',dtype=torch.float32)
@torch.no_grad()
def prepare():
    setup();folder=OUT/'query';folder.mkdir(exist_ok=True)
    for split in ('train','calibration','development'):
        path=folder/(split+'.pt')
        if path.exists():continue
        d=cache(split);qs=queries(split=='train');z=inputs(d,qs);y=target_queries(d['y'],qs)
        if split=='train':
            mu=z[:,:,:384].mean((0,1));std=z[:,:,:384].std((0,1)).clamp_min(.05)
            thresholds=torch.quantile(y[:,:,2:].flatten(0,1),torch.tensor([.5,.75,.9],device='cuda'),dim=0)
            save(dict(mean=mu.cpu(),std=std.cpu(),thresholds=thresholds.cpu()),folder/'statistics.pt')
        stats=load(folder/'statistics.pt','cuda');z[:,:,:384]=((z[:,:,:384]-stats['mean'])/stats['std']).clamp(-10,10)
        save(dict(z=z.cpu(),y=y.cpu(),queries=qs,meta=d['meta']),path)
        print('QUERY_DATA',split,z.shape,flush=True)

def train():
    setup();folder=OUT/'query';d=load(folder/'train.pt','cuda');x=d['z'].flatten(0,1);y=d['y'].flatten(0,1)
    if all((folder/(a+'.pt')).exists() for a in ARMS):return
    torch.manual_seed(42);template=QueryHead().cuda();initial={k:v.detach().clone() for k,v in template.state_dict().items()};models=[]
    for _ in ARMS:
        m=QueryHead().cuda();m.load_state_dict(initial);models.append(m)
    fn,params,buf=combine_state_for_ensemble(models);params=tuple(v.requires_grad_() for v in params);names=[k for k,_ in template.named_parameters()]
    opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-5);rng=torch.Generator(device='cuda').manual_seed(42);masks=torch.stack([mask(a) for a in ARMS]);logs=[];begin=time.monotonic();steps=4000
    for step in range(1,steps+1):
        ids=torch.randint(len(x),(256,),device='cuda',generator=rng);tau=torch.rand(256,device='cuda',generator=rng).clamp(.002,.998)
        opt.zero_grad(set_to_none=True);q=vmap(fn,in_dims=(0,0,None,0,None))(params,buf,x[ids],masks,tau)
        e=y[ids][None]-q;t=tau[None,:,None];losses=torch.maximum(t*e,(t-1)*e).mean((1,2));assert torch.isfinite(losses).all();losses.sum().backward()
        size=sum(p.grad.flatten(1).square().sum(1) for p in params).sqrt();factor=(1/(size+1e-6)).clamp(max=1)
        for p in params:p.grad.mul_(factor.reshape(4,*([1]*(p.ndim-1))))
        opt.param_groups[0]['lr']=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/steps));opt.step()
        if step==1 or step%200==0:
            logs.append(dict(step=step,loss=losses.tolist(),elapsed=time.monotonic()-begin));print('QUERY_TRAIN',logs[-1],flush=True)
    init_hash=hashlib.sha256(b''.join(initial[k].cpu().numpy().tobytes() for k in sorted(initial))).hexdigest()
    for i,a in enumerate(ARMS):save(dict(model={k:v[i].detach().cpu() for k,v in zip(names,params)},seed=42,arm=a,steps=steps,parameters=sum(p.numel() for p in template.parameters()),initial_sha256=init_hash,logs=logs),folder/(a+'.pt'))

@torch.no_grad()
def cdf(q,thresholds,levels):
    # [Q,N,2] x [K,2] -> [N,K,2], piecewise-linear CDF + endpoint atoms.
    out=[]
    for t in thresholds:
        ix=(q<=t[None,None]).sum(0);i=(ix-1).clamp(0,len(levels)-2)
        a=q.gather(0,i[None]).squeeze(0);b=q.gather(0,(i+1)[None]).squeeze(0)
        prob=levels[i]+(levels[i+1]-levels[i])*(t[None]-a)/(b-a).clamp_min(1e-9)
        prob=torch.where(ix==0,torch.zeros_like(prob),torch.where(ix==len(levels),torch.ones_like(prob),prob))
        out.append(1-prob.clamp(0,1))
    return torch.stack(out,1)

def contain_pairs(qs):
    result=[]
    for a in qs:
        for b in qs:
            ra,rb=a['rect'],b['rect'];ta,tb=a['time'],b['time']
            if a['id']==b['id']:continue
            if ra[0]>=rb[0] and ra[1]<=rb[1] and ra[2]>=rb[2] and ra[3]<=rb[3] and ta[0]>=tb[0] and ta[1]<=tb[1]:result.append((a['id'],b['id']))
    return result

@torch.no_grad()
def evaluate():
    setup();folder=OUT/'query';d=load(folder/'development.pt','cuda');cal=load(folder/'calibration.pt','cuda');thresholds=load(folder/'statistics.pt','cuda')['thresholds'];levels=torch.arange(1,100,device='cuda')/100
    n,k=d['y'].shape[:2];rows=[];preds={};qs=d['queries'];std=norm()['std'].flatten().repeat(2);mean=norm()['mean'].flatten().repeat(2)
    for a in ARMS:
        m=QueryHead().cuda().eval();m.load_state_dict(load(folder/(a+'.pt'))['model']);x=d['z'].flatten(0,1)
        chunks=[m.quantiles(x[i:i+1024],mask(a),levels) for i in range(0,len(x),1024)]
        q=torch.cat(chunks,1);y=d['y'].flatten(0,1);assert (q[1:]>=q[:-1]-1e-6).all()
        sc=crps(q,y,levels)*std;cov=((y>=q[4])&(y<=q[94])).float();width=(q[94]-q[4])*std
        event=cdf(q[:,:,2:],thresholds,levels);truth=y[:,None,2:]>thresholds[None];brier=(event-truth.float()).square()
        assert (event[:,1:]<=event[:,:-1]+1e-6).all()
        physical_y=(y*std+mean).cpu()
        sc,cov,width,event,truth,brier=[v.cpu() for v in (sc,cov,width,event,truth,brier)]
        for i,meta in enumerate(d['meta']):
            for j,query in enumerate(qs):
                ix=i*k+j
                for channel in range(2):
                    for stat in range(2):
                        h=channel+stat*2;row=dict(method=a,**meta,query=j,category=query['category'],channel=channel,statistic=('mean','max')[stat],crps=float(sc[ix,h]),coverage=float(cov[ix,h]),width=float(width[ix,h]),target=float(physical_y[ix,h]))
                        if stat:
                            for t in range(3):row.update({f'brier{t}':float(brier[ix,t,channel]),f'prob{t}':float(event[ix,t,channel]),f'event{t}':int(truth[ix,t,channel])})
                        rows.append(row)
        preds[a]=dict(quantiles=q.reshape(99,n,k,4).cpu(),event=event.reshape(n,k,3,2).cpu())
        print('QUERY_EVAL',a,'crps',sc.mean(0).tolist(),'brier',brier.mean((0,1)).tolist(),flush=True)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (folder/'per_query.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
    violations={};pairs=contain_pairs(qs);aa,bb=zip(*pairs)
    for a,v in preds.items():
        e=v['event'];delta=e[:,aa]-e[:,bb];violations[a]=dict(pairs=len(pairs),fraction=float((delta>1e-6).float().mean()),mean_excess=float(delta.clamp_min(0).mean()),max_excess=float(delta.max()))
    dump(folder/'inclusion_violations.json',violations)
    save(dict(predictions=preds,levels=levels.cpu(),thresholds=thresholds.cpu(),targets=d['y'].cpu(),queries=qs,meta=d['meta']),folder/'predictions.pt')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','train','evaluate']);args=ap.parse_args();globals()[args.stage]()
