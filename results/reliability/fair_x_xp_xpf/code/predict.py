"""Portable raw 499-quantile inference using published weights and DiPred."""
import argparse
import sys
from pathlib import Path
import torch
from model import FairQuantileHead


def find_repo():
    here=Path(__file__).resolve()
    for parent in here.parents:
        if (parent/'reliability/networks.py').exists(): return parent
    raise RuntimeError('Place this script under a DiPred checkout')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',type=Path,required=True,help='torch tensor [B,2,20,32,64] in original dataset units')
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--arm',choices=('X','XP','XPF'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    repo=find_repo();sys.path.insert(0,str(repo))
    from reliability.networks import DynamicsDiffusion20
    from reliability.features import forecast_features
    from reliability.backbone import configure
    configure();device=torch.device(args.device)
    shared=torch.load(repo/'reliability/checkpoints/current.pt',map_location=device,weights_only=False)
    history=torch.load(args.history,map_location=device,weights_only=False).float()
    if history.ndim!=5 or history.shape[1:]!=(2,20,32,64):raise ValueError('Invalid input history')
    mean,std=shared['normalization']['mean'],shared['normalization']['std']
    h=(history-mean)/std
    predictor=DynamicsDiffusion20(shared['dynamics_scale']).to(device).eval().requires_grad_(False)
    predictor.load_state_dict(shared['predictor'])
    stats=shared['statistics'];p=torch.zeros_like(h)
    f=torch.zeros(len(h),48,*h.shape[-2:],device=device)
    with torch.no_grad():
        if args.arm=='XP':
            point=predictor.rollout(h,500000,sign=0)
            p=((point-stats['base_mean'][:,:2])/stats['base_std'][:,:2]).clamp(-10,10)
        elif args.arm=='XPF':
            point,all_features=forecast_features(predictor,h)
            p=((point-stats['base_mean'][:,:2])/stats['base_std'][:,:2]).clamp(-10,10)
            f=((all_features[:,-48:].half().float()-stats['last_mean'])/stats['last_std']).clamp(-10,10)
        mask=torch.tensor({'X':(0.,0.),'XP':(1.,0.),'XPF':(1.,1.)}[args.arm],device=device).expand(len(h),2)
        head=FairQuantileHead().to(device).eval().requires_grad_(False)
        head.load_state_dict(torch.load(args.weights,map_location=device,weights_only=False)['model'])
        levels=torch.arange(1,500,device=device,dtype=h.dtype)/500
        quantiles=head.quantiles(h,p,f,mask,levels)*std+mean
    torch.save(dict(quantiles=quantiles.cpu(),levels=levels.cpu(),units='original dataset scale',arm=args.arm),args.output)

if __name__=='__main__':main()
