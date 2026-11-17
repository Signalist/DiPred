"""Tiny local heads: smooth warp versus ordinary additive residual."""
from structure import *
from local_oracle import local_transform
from torch import nn
import math
FOLDER=OUT/'local'
KINDS=['X','X+P','F_last','F_mid0','X+P+F_last','X+P+F_det4_6000','F_det4_1500','F_det4_6000']

def raw_features(d,kind):
    parts=[]
    for k in kind.split('+'):
        if k in ('X','P'):
            v=d[k.lower()].cuda().flatten(1,2);v=F.adaptive_avg_pool2d(v,(4,8))
        else:v=d['features'][k].cuda().reshape(len(d['x']),-1,4,8)
        parts.append(v)
    return torch.cat(parts,1)

def fit_projection(z):
    a=z.permute(0,2,3,1).reshape(-1,z.shape[1]).double();mu=a.mean(0);sd=a.std(0).clamp_min(.01);a=(a-mu)/sd
    cov=a.T@a/len(a);s,v=torch.linalg.eigh(cov);basis=v[:,-32:];scale=s[-32:].clamp_min(1e-4).sqrt()
    return dict(mean=mu,scale=sd,basis=basis,pc_scale=scale)

def apply_projection(z,proj):
    a=z.permute(0,2,3,1).double();a=((a-proj['mean'])/proj['scale'])@proj['basis']/proj['pc_scale']
    return a.permute(0,3,1,2).float().clamp(-10,10)

class Head(nn.Module):
    def __init__(self,style):
        super().__init__();self.style=style
        self.net=nn.Sequential(nn.Conv2d(32,16,1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,7 if style=='warp' else 8,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def coefficients(self,z):return torch.tanh(F.adaptive_avg_pool2d(self.net(z),(2,4)))
    def forward(self,z,p,shrink=1.):
        if shrink==0:return p
        c=self.coefficients(z)*shrink
        if self.style=='warp':return local_transform(p,c*p.new_tensor(LIMITS)[None,:,None,None])
        return p+.25*F.interpolate(c.reshape(len(p),2,4,2,4),p.shape[-3:],mode='trilinear',align_corners=True)

@torch.no_grad()
def errors(head,z,p,y,shrink):
    return torch.cat([mse(head(z[i:i+8],p[i:i+8],shrink),y[i:i+8]) for i in range(0,len(p),8)])

def train(fraction=1.,initial=False):
    FOLDER.mkdir(exist_ok=True);d=load(OUT/'train.pt');cal=load(OUT/'calibration.pt')
    p=d['p'].cuda();y=d['y'].cuda();cp=cal['p'].cuda();cy=cal['y'].cuda();summaries={}
    names=sorted({m['file'] for m in d['meta']});rng=np.random.default_rng(42);rng.shuffle(names);names=names[:max(1,round(len(names)*fraction))]
    ids=torch.tensor([i for i,m in enumerate(d['meta']) if m['file'] in names],device='cuda')
    for kind in ((KINDS[:6] if initial else KINDS) if fraction==1 else ['X+P','F_last','F_det4_6000']):
        raw=raw_features(d,kind);proj=fit_projection(raw[ids]);z=apply_projection(raw,proj);zc=apply_projection(raw_features(cal,kind),proj);del raw
        for style in (('warp','additive') if fraction==1 else ('warp',)):
            name=f'{style}_{kind.replace("+","_")}' + (f'_n{int(fraction*100)}' if fraction!=1 else '')
            if (FOLDER/f'{name}.pt').exists():
                ck=load(FOLDER/f'{name}.pt');summaries[name]=ck['summary'];continue
            torch.manual_seed(42);head=Head(style).cuda();opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-5)
            g=torch.Generator(device='cuda').manual_seed(42);start=time.perf_counter();logs=[]
            for step in range(2000):
                ix=ids[torch.randint(len(ids),(8,),device='cuda',generator=g)]
                lr=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*step/2000))
                for group in opt.param_groups:group['lr']=lr
                opt.zero_grad();q=head(z[ix],p[ix]);loss=mse(q,y[ix]).mean()/norm()['std'].square().mean();assert torch.isfinite(loss)
                loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
                if step%500==0:logs.append(dict(step=step+1,loss=float(loss)))
            head.eval();options={s:aggregate(errors(head,zc,cp,cy,s),cal['meta']) for s in (0.,.25,.5,1.)};shrink=min(options,key=options.get)
            summary=dict(kind=kind,style=style,shrink=shrink,calibration_mse=options[shrink],options=options,parameters=sum(v.numel() for v in head.parameters()),training_seconds=time.perf_counter()-start,seed=42,updates=2000,fraction=fraction,fit_files=names,fit_windows=len(ids))
            save(dict(model=head.state_dict(),projection=proj,summary=summary,training_log=logs),FOLDER/f'{name}.pt');summaries[name]=summary
            print(name,summary,flush=True)
    if (FOLDER/'selection.json').exists():
        original=json.loads((FOLDER/'selection.json').read_text());original['models'].update(summaries)
        original['supplement_note']='Fdet-only controls and 25/50% curves added after initial development evaluation; original selected methods unchanged.'
        dump(FOLDER/'selection.json',original)
    else:
        selected={s:min((k for k in summaries if summaries[k]['style']==s),key=lambda k:summaries[k]['calibration_mse']) for s in ('warp','additive')}
        dump(FOLDER/'selection.json',dict(models=summaries,selected=selected,criterion='trajectory-equal physical MSE on calibration only',development_used=False))

@torch.no_grad()
def evaluate(split):
    d=load(OUT/f'{split}.pt');p=d['p'].cuda();y=d['y'].cuda();rows=[]
    for name in ['base']+[v.stem for v in sorted(FOLDER.glob('*.pt'))]:
        if name=='base':err=mse(p,y)
        else:
            ck=load(FOLDER/f'{name}.pt','cuda');cfg=ck['summary'];head=Head(cfg['style']).cuda().eval();head.load_state_dict(ck['model'])
            z=apply_projection(raw_features(d,cfg['kind']),ck['projection']);err=errors(head,z,p,y,cfg['shrink'])
        for i,m in enumerate(d['meta']):rows.append(dict(**m,method=name,mse=float(err[i])))
    df=pd.DataFrame(rows);df.to_csv(FOLDER/f'{split}_windows.csv',index=False)
    traj=df.groupby(['method','file','group']).mse.mean().reset_index();traj.to_csv(FOLDER/f'{split}_trajectories.csv',index=False)
    summary=traj.groupby('method').mse.mean().to_frame();summary['gain_percent']=(1-summary.mse/summary.loc['base','mse'])*100;summary.to_csv(FOLDER/f'{split}_summary.csv')
    print(summary.to_string(),flush=True)

if __name__=='__main__':
    setup();a=argparse.ArgumentParser();a.add_argument('action',choices=['train','evaluate']);a.add_argument('--split',default='development');a.add_argument('--fraction',type=float,default=1.);a.add_argument('--initial',action='store_true');args=a.parse_args()
    train(args.fraction,args.initial) if args.action=='train' else evaluate(args.split)
