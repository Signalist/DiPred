"""Separate rotational and potential uncertainty before rank-preserving filtering."""
from physics import *
import physics as phy
OUT=BASE/'helmholtz_filter'
original_transform=phy.transform

def names():
    return [f'{kind}_ecc_b{beta}_i{it}' for kind in ('sol','pot','smooth') for beta in (25,50,100) for it in (1,3)]

def transform(values,z,n,kind,beta,iterations,project,p=None):
    if kind=='smooth':return original_transform(values,z,n,kind,beta,iterations,project,p)
    center=z.mean(0,keepdim=True)
    for _ in range(iterations):
        d=(z-center)*n['std'];h=project(d);part=h if kind=='sol' else d-h
        flat=part.reshape(N*2*20,1,32,64)
        change=F.avg_pool2d(F.pad(flat,(1,1,1,1),mode='replicate'),3,1).reshape_as(d)-part
        hc=project(change);change=hc if kind=='sol' else change-hc
        z=couple(values,center+(d+beta*change)/n['std'])
    return z

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('split',choices=['calibration','development','late','future']);args=a.parse_args()
    phy.OUT=OUT;phy.names=names;phy.transform=transform;phy.CANDIDATE_PREFIX='sol';phy.CONTROL_PREFIX='smooth';phy.EXTRAS=[]
    if args.split!='calibration':
        chosen=json.loads((OUT/'selection.json').read_text())['selected'];phy.EXTRAS=[chosen.replace('sol','pot')]
        if args.split in ('late','future'):
            link=OUT/f'{args.split}_data.pt'
            if not link.exists():link.symlink_to(BASE/f'{args.split}_data.pt')
    phy.run(args.split)
