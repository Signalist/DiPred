"""Generate training-only X, actual frozen DiPred P/F, and targets for risk heads."""
import argparse,hashlib,json
import numpy as np
import torch
from common import OUT,Scenario,configure,predict,records

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('scenario');parser.add_argument('--windows',type=int,default=2048);args=parser.parse_args()
    configure();scenario=Scenario(args.scenario);train=[r for r in records(args.scenario) if r['split']=='train']
    checkpoint=OUT/'backbones'/args.scenario/'model.pt'
    model=scenario.network().eval().requires_grad_(False)
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model'])
    folder=OUT/'risk_data'/args.scenario;folder.mkdir(parents=True,exist_ok=True)
    done=folder/'complete.json'
    if done.exists():return
    rng=np.random.default_rng(20261114)
    selected=[train[i] for i in rng.integers(len(train),size=args.windows)]
    starts=[int(rng.integers((r['frames']-40)//20+1))*20 for r in selected]
    specs={'x':(args.windows,2,20,32,64),'p':(args.windows,2,20,32,64),
           'y':(args.windows,2,20,32,64),'f':(args.windows,48,32,64)}
    arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=np.float32,shape=shape) for k,shape in specs.items()}
    fsum=torch.zeros(48,device='cuda',dtype=torch.float64);fsq=torch.zeros_like(fsum);count=0
    for start in range(0,args.windows,8):
        stop=min(start+8,args.windows);rs=selected[start:stop];ts=starts[start:stop]
        x=scenario.batch(rs,ts)
        # P/F are fully issued from X before Y is loaded.
        p,f=predict(model,x,scenario.mask,with_features=True)
        y=scenario.batch(rs,ts,True)
        for key,value in (('x',x),('p',p),('f',f),('y',y)):arrays[key][start:stop]=value.cpu().numpy()
        z=f.double();fsum+=z.sum((0,2,3));fsq+=(z*z).sum((0,2,3));count+=len(f)*32*64
        if start%128==0:print('RISK_DATA',args.scenario,start,flush=True)
    for arr in arrays.values():arr.flush()
    mean=fsum/count;std=(fsq/count-mean*mean).clamp_min(0).sqrt().clamp_min(.05)
    result=dict(windows=args.windows,ids=[r['id'] for r in selected],starts=starts,seed=20261114,
                f_mean=mean.cpu().tolist(),f_std=std.cpu().tolist(),
                backbone_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                issuance='P and final denoising feature before loading future target')
    done.write_text(json.dumps(result,indent=2)+'\n')
    print('RISK_DATA_COMPLETE',args.scenario,flush=True)

if __name__=='__main__':main()
