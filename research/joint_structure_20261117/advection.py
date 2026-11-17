"""Rank-preserving covariance filter along physical flow characteristics."""
from physics import *
import physics as phy
OUT=BASE/'advection'
DT=.02

def names():
    return [f'{kind}_ecc_b{beta}_i{it}' for kind in ('flow','time','reverse') for beta in (25,50,100) for it in (1,3)]

original_transform=phy.transform
def transform(values,z,n,kind,beta,iterations,project,p=None):
    if kind=='smooth':return original_transform(values,z,n,kind,beta,iterations,project,p)
    center=z.mean(0,keepdim=True)
    velocity=p*n['std'][0]+n['mean'][0]
    shift=velocity.permute(1,2,3,0)*velocity.new_tensor([DT/DX*2/63,-DT/DX*2/31])
    if kind=='time':shift=shift*0
    if kind=='reverse':shift=-shift
    yy,xx=torch.meshgrid(torch.linspace(-1,1,32,device='cuda'),torch.linspace(-1,1,64,device='cuda'),indexing='ij');grid=torch.stack((xx,yy),-1)[None].expand(20,-1,-1,-1)
    prev_grid=grid-shift;next_grid=grid+shift;prev_grid[0]=grid[0];next_grid[-1]=grid[-1]
    for _ in range(iterations):
        d=(z-center)*n['std']
        before=torch.cat((d[:,:,:1],d[:,:,:-1]),2).permute(0,2,1,3,4).reshape(N*20,2,32,64)
        after=torch.cat((d[:,:,1:],d[:,:,-1:]),2).permute(0,2,1,3,4).reshape(N*20,2,32,64)
        left=F.grid_sample(before,prev_grid.repeat(N,1,1,1),align_corners=True,padding_mode='border')
        right=F.grid_sample(after,next_grid.repeat(N,1,1,1),align_corners=True,padding_mode='border')
        new=((left+right)/2).reshape(N,20,2,32,64).permute(0,2,1,3,4)
        z=couple(values,center+((1-beta)*d+beta*new)/n['std'])
    return z

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('split',choices=['calibration','development','late']);args=a.parse_args()
    phy.OUT=OUT;phy.names=names;phy.transform=transform;phy.CANDIDATE_PREFIX='flow';phy.CONTROL_PREFIX='time';phy.EXTRAS=['smooth_ecc_b50_i1']
    if args.split!='calibration':phy.EXTRAS.append(json.loads((OUT/'selection.json').read_text())['selected'].replace('flow','reverse'))
    phy.run(args.split)
