"""Raw HDF5 to fixed 50%-budget held-out-Re table, using released weights only.

Requires this repository's reliability.networks module, the public RealPDE
train_real HDF5 files and unpacked primary_assets.tar.gz. No training caches.
"""
from pathlib import Path
import sys,json,argparse,hashlib
import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

class Head(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(32,16,1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,7,1))
 def forward(self,z,p,gain):
  if gain==0:return p
  c=torch.tanh(F.adaptive_avg_pool2d(self.net(z),(2,4)))*gain*p.new_tensor([1,1,1,.25,.25,.25,.25])[None,:,None,None];nt,ny,nx=p.shape[-3:];coeff=F.interpolate(c,(ny,nx),mode='bilinear',align_corners=True);t,y,x=torch.meshgrid(torch.linspace(-1,1,nt,device=p.device),torch.linspace(-1,1,ny,device=p.device),torch.linspace(-1,1,nx,device=p.device),indexing='ij');grid=torch.stack([x,y,t],-1)[None]+coeff[:,:3].permute(0,2,3,1)[:,None]*p.new_tensor([2/(nx-1),2/(ny-1),2/(nt-1)]);q=F.grid_sample(p,grid,padding_mode='border',align_corners=True);return q*(1+coeff[:,3:5,None])+coeff[:,5:7,None]

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def projfeature(f,b):
 z=f.reshape(len(f),-1,4,8).permute(0,2,3,1).double();z=((z-b['mean'])/b['scale'])@b['basis']/b['pc_scale'];return z.permute(0,3,1,2).float().clamp(-10,10)

def choose(meta,priority):
 mask=np.zeros(len(meta),dtype=bool);d=pd.DataFrame(meta)
 for _,g in d.groupby('start',sort=True):
  order=sorted(g.index,key=lambda i:(-float(priority[i]),meta[i]['file']));mask[order[:int(np.floor(.5*len(order)+.5))]]=True
 return mask

def main(args):
 sys.path.insert(0,str(args.repository));from reliability.networks import DynamicsDiffusion20
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(42);np.random.seed(42)
 device='cuda' if torch.cuda.is_available() else 'cpu';assets=Path(args.assets);manifest=json.loads((assets/'asset_manifest.json').read_text());assert digest(args.checkpoint)==manifest['backbone_sha256']
 for path,sha in manifest['files'].items():assert digest(assets/path)==sha,path
 ck=torch.load(args.checkpoint,map_location=device,weights_only=False);net=DynamicsDiffusion20(ck['dynamics_scale']).to(device).eval();net.load_state_dict(ck['predictor']);net.requires_grad_(False);mean=ck['normalization']['mean'].to(device);std=ck['normalization']['std'].to(device);records=json.loads((assets/'manifest.json').read_text())['files']['test'];shared=torch.load(assets/'routers_shared.pt',map_location='cpu',weights_only=False);meta=[];xx=[];pp=[];ff=[];yy=[]
 with torch.no_grad():
  for rec in records:
   with h5py.File(Path(args.data)/rec['file'],'r') as f:
    for start in range(0,161,20):
     raw=torch.from_numpy(np.stack([f[c][start:start+40,::2,::2] for c in ['u','v']]).astype('float32')).to(device);x=(raw[:,:20][None]-mean)/std;tap=[];handle=net.net.up2.register_forward_hook(lambda module,args,out:tap.append(F.adaptive_avg_pool2d(out,(4,8)).flatten(1).detach()))
     try:p=net.rollout(x,500000,sign=0)
     finally:handle.remove()
     assert len(tap)==4;xx.append(x.cpu());pp.append(p.cpu());ff.append(tap[-1].cpu());yy.append(((raw[:,20:][None]-mean)/std).cpu());meta.append(dict(file=rec['file'],group=str(rec['group']),start=start))
  x=torch.cat(xx).to(device);p=torch.cat(pp).to(device);f=torch.cat(ff).to(device);des=torch.cat([F.adaptive_avg_pool3d(x,(4,4,8)).flatten(1),F.adaptive_avg_pool3d(p,(4,4,8)).flatten(1),f],1).cpu().double();bp=shared['projection'];z=((des-bp['mean'])/bp['scale'])@bp['basis']/bp['pcscale'];issued=[]
  for seed in [42,123,2025]:
   c=torch.load(assets/f'warp_{seed}.pt',map_location=device,weights_only=False);head=Head().to(device).eval();head.load_state_dict(c['model']);priority=(torch.cat([torch.ones_like(z[:,:1]),z],1)@shared['routers'][str(seed)]['benefit']['coef']).numpy();mask=choose(meta,priority);q=torch.cat([head(projfeature(f[i:i+8],c['projection']),p[i:i+8],c['gain']).cpu() for i in range(0,len(p),8)]);issued.append((seed,mask,q))
  # Only now use truth for scoring.
  y=torch.cat(yy).to(device);base=(((p-y)*std)**2).mean((1,2,3,4)).cpu().numpy();rows=[]
  for seed,mask,q in issued:
   e=(((q.to(device)-y)*std)**2).mean((1,2,3,4)).cpu().numpy()
   for i,m in enumerate(meta):rows.append(dict(seed=seed,**m,benefit=float(e[i] if mask[i] else base[i]),random_expected=float((base[i]+e[i])/2),action=int(mask[i])))
 df=pd.DataFrame(rows);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);df.to_csv(out/'raw_windows.csv',index=False);tr=df.groupby(['seed','file','group'])[['benefit','random_expected']].mean().reset_index();summary=tr.groupby('seed')[['benefit','random_expected']].mean();summary['gain_percent']=(1-summary.benefit/summary.random_expected)*100;summary.to_csv(out/'raw_summary.csv');print(summary.to_string(),flush=True)
 if args.reference:
  reference=pd.read_csv(args.reference);maxdiff=max(abs(float(summary.loc[s,'gain_percent'])-float(reference[(reference.split=='test')&(reference.seed==s)&(reference.quota==.5)].gain_percent.iloc[0])) for s in [42,123,2025]);print('max gain percent difference',maxdiff);assert maxdiff<1e-5

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--repository',type=Path,required=True,help='DiPred checkout containing reliability/networks.py');parser.add_argument('--data',type=Path,required=True,help='RealPDE train_real directory');parser.add_argument('--assets',type=Path,required=True,help='Unpacked primary_assets.tar.gz');parser.add_argument('--checkpoint',type=Path,required=True,help='Original reliability/checkpoints/current.pt');parser.add_argument('--out',type=Path,required=True);parser.add_argument('--reference',type=Path);main(parser.parse_args())
