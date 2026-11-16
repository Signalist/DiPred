"""Train one state diffusion predictor per scenario, using training trajectories only."""
import argparse,hashlib,json,math,time
import numpy as np
import torch
from common import OUT,Scenario,configure,records,save_atomic


def main():
    parser=argparse.ArgumentParser();parser.add_argument('scenario');parser.add_argument('--steps',type=int,default=2000);args=parser.parse_args()
    configure();torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    scenario=Scenario(args.scenario);train=[r for r in records(args.scenario) if r['split']=='train']
    folder=OUT/'backbones'/args.scenario;folder.mkdir(parents=True,exist_ok=True)
    model=scenario.network().train();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    rng=np.random.default_rng(42);first=0;elapsed=0.;logs=[]
    progress=folder/'progress.pt'
    if progress.exists():
        saved=torch.load(progress,map_location='cuda',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['rng']
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        first=saved['step'];elapsed=saved['elapsed'];logs=saved['logs']
    start_time=time.monotonic()
    for step in range(first+1,args.steps+1):
        rs=[train[i] for i in rng.integers(len(train),size=4)]
        starts=[int(rng.integers((r['frames']-40)//20+1))*20 for r in rs]
        x=scenario.batch(rs,starts);y=scenario.batch(rs,starts,True)
        t=torch.randint(4,(len(x),),device='cuda');a=model.abar[t,None,None,None,None]
        noise=torch.randn_like(y)*scenario.mask;noisy=(a.sqrt()*y+(1-a).sqrt()*noise)*scenario.mask
        prediction=model.net(noisy,t*33,x)*scenario.mask
        loss=(((prediction-y).square()/(1-a))*scenario.mask).sum()/(len(x)*scenario.channels*20*32*64)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite diffusion training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.param_groups[0]['lr']=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/args.steps))
        optimizer.step()
        if step==1 or step%100==0:
            logs.append(dict(step=step,loss=float(loss)));print('BACKBONE',args.scenario,step,float(loss),flush=True)
        if step%250==0:
            save_atomic(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),logs=logs,
                elapsed=elapsed+time.monotonic()-start_time),progress)
    elapsed+=time.monotonic()-start_time
    save_atomic(dict(model=model.state_dict(),scenario=args.scenario,seed=42,steps=args.steps,
        optimizer='AdamW',train_ids=[r['id'] for r in train],normalization=scenario.info,
        logs=logs,seconds=elapsed,parameters=sum(p.numel() for p in model.parameters())),folder/'model.pt')
    progress.unlink(missing_ok=True)
    print('BACKBONE_COMPLETE',args.scenario,elapsed,flush=True)

if __name__=='__main__':main()
