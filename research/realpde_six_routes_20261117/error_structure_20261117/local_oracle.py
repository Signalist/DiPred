"""Smooth 2x4 coefficient fields: 56-parameter diagnostic, not a predictor."""
from structure import *

def local_transform(p,c):
    nt,ny,nx=p.shape[-3:]
    coeff=F.interpolate(c,(ny,nx),mode='bilinear',align_corners=True)
    t,y,x=torch.meshgrid(torch.linspace(-1,1,nt,device=p.device),torch.linspace(-1,1,ny,device=p.device),torch.linspace(-1,1,nx,device=p.device),indexing='ij')
    grid=torch.stack([x,y,t],-1)[None]+coeff[:,:3].permute(0,2,3,1)[:,None]*p.new_tensor([2/(nx-1),2/(ny-1),2/(nt-1)])
    q=F.grid_sample(p,grid,padding_mode='border',align_corners=True)
    return q*(1+coeff[:,3:5,None])+coeff[:,5:7,None]

def optimize(p,y,mode,steps=120):
    c=torch.zeros(len(p),7,2,4,device=p.device,requires_grad=True)
    limits=p.new_tensor(LIMITS)[None,:,None,None]
    mask=torch.ones_like(limits)
    if mode=='geometry':mask[:,3:]=0
    if mode=='amplitude':mask[:,:3]=0
    opt=torch.optim.Adam([c],lr=.025);best=mse(p,y).detach();bestc=torch.zeros_like(c)
    for _ in range(steps):
        opt.zero_grad();error=mse(local_transform(p,c*mask),y)
        with torch.no_grad():
            good=error<best;best[good]=error[good];bestc[good]=(c*mask)[good]
        (error/norm()['std'].square().mean()).mean().backward();opt.step()
        with torch.no_grad():c.clamp_(-limits,limits)
    return bestc.detach(),best

def main(split):
    setup();d=load(OUT/f'{split}.pt');rows=[];coefficients={k:[] for k in ('geometry','amplitude','combined')}
    for i in range(0,len(d['p']),8):
        p=d['p'][i:i+8].cuda();y=d['y'][i:i+8].cuda()
        for mode in coefficients:
            c,error=optimize(p,y,mode);coefficients[mode].append(c.cpu())
            for j,m in enumerate(d['meta'][i:i+8]):rows.append(dict(**m,method=mode,mse=float(error[j])))
    save(dict(coefficients={k:torch.cat(v) for k,v in coefficients.items()},meta=d['meta']),OUT/f'{split}_local_oracle.pt')
    df=pd.DataFrame(rows);df.to_csv(OUT/f'{split}_local_oracle.csv',index=False)
    base=aggregate(mse(d['p'].cuda(),d['y'].cuda()),d['meta']);means=df.groupby(['method','file']).mse.mean().groupby('method').mean()
    print('LOCAL ORACLE',split,means.to_dict(),((1-means/base)*100).to_dict(),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--split',default='calibration');args=a.parse_args();main(args.split)
