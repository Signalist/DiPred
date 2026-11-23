"""State-conditioned residual propagation, using the frozen original residual cache."""
import argparse, csv, hashlib, json, random, sys
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def dump(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def table(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

class Corrector(nn.Module):
    def __init__(self, conditioned):
        super().__init__(); self.conditioned=conditioned
        # 6 physical channels (Y,Yhat,R) or zero-filled Y/Yhat slots, plus x/y/time.
        self.net=nn.Sequential(nn.Conv2d(9,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,2,1))
    def forward(self,y,yhat,r,lead):
        n,_,h,w=r.shape; dev=r.device
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=dev),torch.linspace(-1,1,w,device=dev),indexing='ij')
        coords=torch.stack([xx,yy])[None].expand(n,-1,-1,-1)
        step=((lead.float()-2)/18*2-1)[:,None,None,None].expand(n,1,h,w)
        if not self.conditioned: y=torch.zeros_like(y); yhat=torch.zeros_like(yhat)
        return r + self.net(torch.cat([y,yhat,r,coords,step],1))

def load_truth(data, records, start):
    vals=[]
    for r in records:
        with h5py.File(data/r['file'],'r') as f: vals.append(np.stack([f[c][start:start+20,::2,::2] for c in ('u','v')]).astype('float32'))
    return torch.from_numpy(np.stack(vals))

def load_split(args,split):
    records=json.loads((args.out/'protocol.json').read_text())['records'][split]
    rs=[]; ys=[]
    for start in (400,420):
        e=torch.load(args.cache/f'{split}_{start}.pt',map_location='cpu',weights_only=False)['residual']
        y=load_truth(args.data,records,start+20)
        assert e.shape==(len(records),32,2,20,32,64) and y.shape==(len(records),2,20,32,64)
        rs.append(e); ys.append(y)
    # trajectory, window, member, channel, frame, y, x
    return torch.stack(rs,1), torch.stack(ys,1)

def evaluate(model,r,y,scale,shuffle=None):
    model.eval(); out=[]
    with torch.no_grad():
      for i in range(r.shape[0]):
        sums=torch.zeros(2,device=r.device,dtype=torch.float64)
        for wi in range(2):
          prev=r[i,wi]; truth=y[i,wi,None].expand(32,-1,-1,-1,-1)
          if shuffle is not None: prev=prev[torch.randperm(32,device=r.device)]
          for t in range(19):
            yt=truth[:,:,t]; yp=yt+prev[:,:,t]; pred=model(yt/scale,yp/scale,prev[:,:,t]/scale,torch.full((32,),t+2,device=r.device))*scale
            sums+=(pred-prev[:,:,t+1]).double().square().mean((0,2,3))
        out.append((sums/38).cpu().numpy())
    return np.asarray(out)

def protocol(args):
    manifest=args.out.parent/'residual_dynamics/cache_manifest.json'; original_protocol=args.out.parent/'residual_dynamics/protocol.json'
    base=json.loads(manifest.read_text())
    # Preserve the original split and cache identities, and record the state reconstruction.
    records={s:json.loads(original_protocol.read_text())['records'][s] for s in ('train','development','test')}
    dump(args.out/'protocol.json',dict(version=1,base_cache=str(args.cache),base_cache_manifest=str(manifest),base_cache_manifest_sha256=sha(manifest),starts=[400,420],member_seeds=list(range(500000,500032)),records=records, residual_definition='R=Yhat-Y, original data scale', target='R[t+1]', input_indices='Y[t], Yhat[t]=Y[t]+R[t], R[t]; no target truth in input', phase='A offline oracle single-step; not deployment-valid', models={'R-only':'9-channel network with Y/Yhat slots zeroed','State-conditioned':'same 9-channel network with Y/Yhat'}, cache_entries=len(base)))

def fit(args):
    device='cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(42); np.random.seed(42); random.seed(42)
    tr,ty=load_split(args,'train'); dv,dy=load_split(args,'development'); tr,ty,dv,dy=[x.to(device) for x in (tr,ty,dv,dy)]
    scale=float(tr.square().mean().sqrt()); logs=[]; selected={}; params={}
    for name,cond in [('R-only',False),('State-conditioned',True)]:
      torch.manual_seed(42); torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None
      m=Corrector(cond).to(device); params[name]=sum(p.numel() for p in m.parameters()); opt=torch.optim.Adam(m.parameters(),lr=.001); rng=torch.Generator(device=device).manual_seed(20261119); best=float('inf')
      for step in range(1,args.steps+1):
        ti=torch.randint(tr.shape[0],(16,),generator=rng,device=device); wi=torch.randint(2,(16,),generator=rng,device=device); mi=torch.randint(32,(16,),generator=rng,device=device); fi=torch.randint(19,(16,),generator=rng,device=device)
        rr=tr[ti,wi,mi]; yy=ty[ti,wi]
        prev=torch.stack([rr[i,:,fi[i]] for i in range(16)]); targ=torch.stack([rr[i,:,fi[i]+1] for i in range(16)]); yt=torch.stack([yy[i,:,fi[i]] for i in range(16)]); yp=yt+prev
        pred=m(yt/scale,yp/scale,prev/scale,fi+2)*scale; loss=(pred-targ).square().mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step%250==0 or step==args.steps:
          val=float(evaluate(m,dv,dy,scale).mean()); logs.append(dict(model=name,step=step,batch_loss=float(loss),development_mse=val)); print('FIT',name,step,val,flush=True)
          if val<best: best=val; selected[name]=step; torch.save({'model':m.state_dict(),'scale':scale,'step':step,'conditioned':cond},args.out/(name.replace('-','_').replace(' ','_')+'.pt'))
    dump(args.out/'training.json',dict(seed=42,batch_seed=20261119,steps=args.steps,batch_size=16,optimizer='Adam lr=.001',scalar_train_rms=scale,parameters=params,selected_steps=selected,log=logs,device=device))

def score(args):
    p=json.loads((args.out/'protocol.json').read_text()); rows=[]; summaries=[]
    for split in ('development','test'):
      r,y=load_split(args,split); device='cuda' if torch.cuda.is_available() else 'cpu'; r,y=r.to(device),y.to(device)
      for name in ('R-only','State-conditioned'):
        ck=torch.load(args.out/(name.replace('-','_').replace(' ','_')+'.pt'),map_location=device,weights_only=False); m=Corrector(ck['conditioned']).to(device); m.load_state_dict(ck['model']); vals=evaluate(m,r,y,ck['scale']);
        for i,rec in enumerate(p['records'][split]): rows.append(dict(split=split,trajectory=rec['file'],model=name,mse=float(vals[i].mean()),mse_u=float(vals[i,0]),mse_v=float(vals[i,1]),windows=2,members=32,pairs_per_member_per_window=19))
        summaries.append(dict(split=split,model=name,trajectories=len(vals),mse=float(vals.mean()),mse_u=float(vals[:,0].mean()),mse_v=float(vals[:,1].mean())))
      # persistence is exact no-training baseline
      vals=[]
      for i in range(r.shape[0]):
        d=(r[i,:,:,:,1:]-r[i,:,:,:,:-1]).double().square().mean((0,1,3,4,5)).cpu().numpy(); vals.append(d)
      vals=np.asarray(vals); 
      for i,rec in enumerate(p['records'][split]): rows.append(dict(split=split,trajectory=rec['file'],model='Persistence',mse=float(vals[i].mean()),mse_u=float(vals[i,0]),mse_v=float(vals[i,1]),windows=2,members=32,pairs_per_member_per_window=19))
      summaries.append(dict(split=split,model='Persistence',trajectories=len(vals),mse=float(vals.mean()),mse_u=float(vals[:,0].mean()),mse_v=float(vals[:,1].mean())))
    table(args.out/'per_trajectory.csv',rows); dump(args.out/'summary.json',summaries)

def main():
 p=argparse.ArgumentParser(); p.add_argument('stage',choices=['protocol','fit','score','all']); p.add_argument('--workspace',type=Path,default=REPO.parent/'Forecast/Runtime'); p.add_argument('--data',type=Path); p.add_argument('--cache',type=Path); p.add_argument('--out',type=Path,default=REPO/'results/corrector'); p.add_argument('--steps',type=int,default=1000); a=p.parse_args(); a.data=a.data or REPO.parents[1]/'OLD/REALPDE/realpdebench-competition/data/train_real'; a.cache=a.cache or a.workspace/'runs/residual_dynamics_20261119'; a.out.mkdir(parents=True,exist_ok=True)
 for s in ('protocol','fit','score') if a.stage=='all' else (a.stage,): globals()[s](a)
if __name__=='__main__': main()
