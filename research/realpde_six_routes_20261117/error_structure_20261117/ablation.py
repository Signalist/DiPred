"""Fixed small controls to distinguish conditional correction from smoothing."""
from local_learn import *

@torch.no_grad()
def smooth(p):
    a=p.flatten(0,2)[:,None];a=F.avg_pool2d(F.pad(a,(1,1,1,1),mode='replicate'),3,1)
    return a.reshape_as(p)

def fit():
    d=load(OUT/'train.pt');cal=load(OUT/'calibration.pt');p=d['p'].cuda();y=d['y'].cuda();cp=cal['p'].cuda();cy=cal['y'].cuda()
    torch.manual_seed(42);c=torch.zeros(1,7,2,4,device='cuda',requires_grad=True);opt=torch.optim.Adam([c],lr=.003)
    g=torch.Generator(device='cuda').manual_seed(42);limits=p.new_tensor(LIMITS)[None,:,None,None]
    for _ in range(2000):
        ix=torch.randint(len(p),(8,),device='cuda',generator=g);opt.zero_grad()
        q=local_transform(p[ix],torch.tanh(c).expand(8,-1,-1,-1)*limits);loss=mse(q,y[ix]).mean()/norm()['std'].square().mean();loss.backward();opt.step()
    c=torch.tanh(c.detach())*limits
    options={s:aggregate(mse(cp if s==0 else local_transform(cp,c.expand(len(cp),-1,-1,-1)*s),cy),cal['meta']) for s in (0.,.25,.5,1.)};shrink=min(options,key=options.get)
    sp=smooth(cp);soptions={s:aggregate(mse(cp+s*(sp-cp),cy),cal['meta']) for s in (0.,.05,.1,.25,.5,1.)};beta=min(soptions,key=soptions.get)
    save(dict(c=c,shrink=shrink,beta=beta,static_options=options,smooth_options=soptions),OUT/'simple_controls.pt')
    dump(OUT/'simple_controls.json',dict(shrink=shrink,beta=beta,static_options=options,smooth_options=soptions,protocol='Supplementary diagnosis after first development evaluation; constant56 coefficients fit train only, strength on calibration; uniform3x3 smoothing strength on calibration; no primary model replacement.'))

@torch.no_grad()
def evaluate(split):
    d=load(OUT/f'{split}.pt');ck=load(FOLDER/'warp_F_last.pt','cuda');cfg=ck['summary'];head=Head('warp').cuda().eval();head.load_state_dict(ck['model'])
    z=apply_projection(raw_features(d,'F_last'),ck['projection']);control=load(OUT/'simple_controls.pt','cuda');rows=[]
    for i in range(0,len(d['p']),8):
        p=d['p'][i:i+8].cuda();y=d['y'][i:i+8].cuda();c=head.coefficients(z[i:i+8])*cfg['shrink']*p.new_tensor(LIMITS)[None,:,None,None]
        results={'base':p,'full':local_transform(p,c),'static56':local_transform(p,control['c'].expand(len(p),-1,-1,-1)*control['shrink']),'smooth':p+control['beta']*(smooth(p)-p)}
        for mode,keep in [('geometry',[0,1,2]),('amplitude',[3,4]),('bias',[5,6])]:
            cc=torch.zeros_like(c);cc[:,keep]=c[:,keep];results[mode]=local_transform(p,cc)
        for name,q in results.items():
            err=mse(q,y)
            for j,m in enumerate(d['meta'][i:i+8]):rows.append(dict(**m,method=name,mse=float(err[j])))
    df=pd.DataFrame(rows);df.to_csv(OUT/f'{split}_ablation.csv',index=False)
    summary=df.groupby(['method','file']).mse.mean().groupby('method').mean();print(split,((1-summary/summary['base'])*100).to_dict(),flush=True)

if __name__=='__main__':
    setup();a=argparse.ArgumentParser();a.add_argument('action',choices=['fit','evaluate']);a.add_argument('--split',default='development');args=a.parse_args();fit() if args.action=='fit' else evaluate(args.split)
