"""Development-only center decomposition and prediction-defined local groups."""
from common import *
from risk import RiskHead,ARMS,features
import csv, argparse
sys.path.insert(0,str(W))
from fair_f_p_pf_xf_sharedref.model import FairQuantileHead
LEVELS=torch.arange(1,100,device='cuda')/100

def gradients(p):
    return (torch.nn.functional.pad(p.diff(dim=-1).square(),(0,1,0,0))+torch.nn.functional.pad(p.diff(dim=-2).square(),(0,0,0,1))+1e-12).sqrt()
def groups(p,threshold):
    b,c,t,h,w=p.shape;ones=torch.ones_like(p,dtype=torch.bool);out={'all':ones}
    for name,start,end in [('lead01_05',0,5),('lead06_10',5,10),('lead11_20',10,20)]:
        a=torch.zeros_like(ones);a[:,:,start:end]=True;out[name]=a
    for i in range(2):
        for j in range(2):
            a=torch.zeros_like(ones);a[:,:,:,i*16:(i+1)*16,j*32:(j+1)*32]=True;out[f'quadrant{i}{j}']=a
    a=gradients(p)>=threshold[None,:,None,None,None];out['high_gradient']=a;out['low_gradient']=~a
    return out

def latest(arm):
    root=W/('runs/fair_f_p_pf_xf_sharedref_20261116' if arm in ('F','P') else 'runs/fair_x_xp_xpf_20261113')
    ck=load(root/'seed_42'/f'{arm}.pt');m=FairQuantileHead().cuda().eval().requires_grad_(False);m.load_state_dict(ck['model']);return m

@torch.no_grad()
def old_quantiles(m,arm,split):
    qs=[]
    for s in range(0,161,20):
        d=load(W/f'runs/fair_x_xp_xpf_20261113/inputs/{split}/{s:04d}.pt','cuda')
        x=d['x'];p=d['p'];f=d['f'];mask=torch.tensor((float(arm in ('P','XPF')),float(arm in ('F','XPF'))),device='cuda').expand(len(x),2)
        inputx=x if arm in ('X','XPF') else torch.zeros_like(x)
        o=m.decoder(m.encode(inputx,p,f,mask));center=o[:,:2]+x[:,:,-1:]
        weights=torch.nn.functional.softplus(o[:,2:].reshape(len(x),2,6,20,32,64))/6
        qs.append((center.cpu(),weights.cpu()))
    return torch.cat([v[0] for v in qs]).cuda(),torch.cat([v[1] for v in qs]).cuda()

@torch.no_grad()
def run(tag='main'):
    setup();tr=cache('train');dev=cache('development');cal=cache('calibration');std=norm()['std'];stats=load(SHARED)['statistics']
    threshold=torch.quantile(gradients(tr['p'])[...,::2,::2].permute(1,0,2,3,4).flatten(1),.8,dim=1)
    donors={s:tr['f'][torch.arange(len(d['x']),device='cuda')*7%len(tr['f'])] for s,d in [('development',dev),('calibration',cal)]}
    save(threshold.cpu(),OUT/'gradient_threshold.pt');del tr
    rows=[];snapshots=[]
    sources=[('old_'+a,None) for a in ('X','P','F','XPF')] if tag=='main' else []
    sources += [(p.stem,p) for p in sorted((OUT/'risk'/tag).glob('*.pt')) if p.stem!='progress']
    for name,path in sources:
        old=name.startswith('old_');arm=name[4:] if old else name
        if old:
            m=latest(arm);c,w=old_quantiles(m,arm,'calibration');dc,dw=old_quantiles(m,arm,'development')
            def quant(data,ids,split):
                center,weight=(c,w) if split=='calibration' else (dc,dw)
                return center[ids][None]+torch.einsum('bckthw,qk->qbcthw',weight[ids],bases(LEVELS))
        else:
            ck=load(path);m=RiskHead(ck['width'],ck.get('symmetric',False)).cuda().eval().requires_grad_(False);m.load_state_dict(ck['model'])
            def quant(data,ids,split):return m.quantiles(*features(data,arm,ids,donors[split]),LEVELS)
        scores=[]
        for i in range(0,len(cal['x']),4):
            ids=torch.arange(i,min(i+4,len(cal['x'])),device='cuda');q=quant(cal,ids,'calibration')
            scores.append(torch.maximum(q[4]-cal['y'][ids],cal['y'][ids]-q[94]).cpu())
        offset=torch.quantile(torch.cat(scores).permute(1,0,2,3,4).flatten(1).cuda(),.9,dim=1)
        for i in range(0,len(dev['x']),4):
            ids=torch.arange(i,min(i+4,len(dev['x'])),device='cuda');q=quant(dev,ids,'development');y=dev['y'][ids];p=dev['p'][ids]
            assert (q[1:]>=q[:-1]-1e-6).all()
            if not old:assert torch.equal(q[49],p)
            variants={'raw':q}
            if old:variants['center_aligned']=q-q[49:50]+p[None]
            masks=groups(p,threshold)
            for mode,qq in variants.items():
                sc=crps(qq,y,LEVELS)*std;lo,hi=qq[4],qq[94]
                mse=((qq[49]-y)*std).square();width=(hi-lo)*std
                cov=((y>=lo)&(y<=hi)).float();interval=width+20*((lo-y).clamp_min(0)+(y-hi).clamp_min(0))*std
                metrics=dict(crps=sc,mse=mse,coverage=cov,width=width,interval=interval)
                for group,mask in masks.items():
                    count=mask.sum((2,3,4)).clamp_min(1)
                    values={k:((v*mask).sum((2,3,4))/count).cpu().numpy() for k,v in metrics.items()}
                    for j,index in enumerate(ids.tolist()):
                        for channel in range(2):
                            if int(mask[j,channel].sum())==0:continue
                            rows.append(dict(method=name,mode=mode,region=group,channel=channel,**dev['meta'][index],**{k:float(v[j,channel]) for k,v in values.items()}))
                if i==0:
                    snapshots.append(dict(method=name,mode=mode,q=qq[:,0,:,::5,::8,::16].cpu(),y=y[0,:,::5,::8,::16].cpu(),crps=sc[0,:,::5,::8,::16].cpu()))
            lo=q[4]-offset[None,:,None,None,None];hi=q[94]+offset[None,:,None,None,None];mid=(lo+hi)/2
            lo=torch.minimum(lo,mid);hi=torch.maximum(hi,mid)
            metrics=dict(coverage=((y>=lo)&(y<=hi)).float(),width=(hi-lo)*std,interval=((hi-lo)+20*((lo-y).clamp_min(0)+(y-hi).clamp_min(0)))*std)
            for group,mask in masks.items():
                count=mask.sum((2,3,4)).clamp_min(1);values={k:((v*mask).sum((2,3,4))/count).cpu().numpy() for k,v in metrics.items()}
                for j,index in enumerate(ids.tolist()):
                    for channel in range(2):
                        if int(mask[j,channel].sum())==0:continue
                        rows.append(dict(method=name,mode='global_cqr',region=group,channel=channel,**dev['meta'][index],**{k:float(v[j,channel]) for k,v in values.items()}))
        print('RISK_EVAL',tag,name,flush=True)
        if old:del c,w,dc,dw
    folder=OUT/'risk'/tag
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (folder/'per_group.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    save(dict(levels=LEVELS.cpu(),std=std.cpu(),snapshots=snapshots),folder/'snapshots.pt')
    dump(folder/'evaluation.json',dict(rows=len(rows),split='development',seed=42,group_threshold=threshold.tolist(),quantile_levels=99,cqr_split='calibration'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='main');a=ap.parse_args();run(a.tag)
