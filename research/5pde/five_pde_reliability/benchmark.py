"""Single-trajectory latency including scenario backbone and calibration."""
import argparse,json,time,sys
import torch
from common import OUT,Scenario,configure,predict,records
from evaluate import ARMS,NOISE_ARMS,MODES,ALPHA,LEVELS,MASKS,load_head,issued
from calibration_grid import GridCalibrator,MedianCalibrator
from distribution_grid import rearrange_quantiles

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('scenario');args=parser.parse_args()
    configure();name=args.scenario;sc=Scenario(name);r=next(r for r in records(name) if r['split']=='development')
    x=sc.batch([r],[0]);backbone=sc.network().eval().requires_grad_(False)
    backbone.load_state_dict(torch.load(OUT/'backbones'/name/'model.pt',map_location='cuda',weights_only=False)['model'])
    results=[]
    for seed in (42,123,2026):
        for arm in ARMS:
            head,state,mean,std=load_head(name,arm,seed)
            cal=torch.load(OUT/'calibration'/name/f'{arm}_{seed}.pt',map_location='cuda',weights_only=False)
            mask=torch.tensor(MASKS[arm],device='cuda').expand(1,2)
            for mode in MODES:
                online=GridCalibrator(ALPHA.cuda(),cal['initial'],1);median=MedianCalibrator(cal['median'],1)
                def compute():
                    if arm=='X':p=torch.zeros_like(x);f=torch.zeros(1,48,32,64,device='cuda')
                    elif arm=='XP':p=predict(backbone,x,sc.mask);f=torch.zeros(1,48,32,64,device='cuda')
                    else:
                        p,f=predict(backbone,x,sc.mask,True);f=((f-mean)/std).clamp(-10,10)
                    q=head.quantiles(x,p,f,mask,LEVELS.cuda().float())
                    if mode=='fixed_cqr':
                        lo,hi=q[:249],q[250:].flip(0);delta=cal['fixed'][:,None,:,None,None,None]
                        a,b=lo-delta,hi+delta;invalid=a>b;mid=(a+b)/2
                        q=torch.cat((torch.where(invalid,mid,a),q[249:250],torch.where(invalid,mid,b).flip(0)))
                        q=rearrange_quantiles(q)[0]
                    elif mode=='online':
                        a,b=online.issue(q[:249],q[250:].flip(0));mid=median.issue(q[249]);q=torch.cat((a,mid[None],b.flip(0)))
                        online.pending=None;median.pending=None;q=rearrange_quantiles(q)[0]
                    return (q*sc.std+sc.mean)[:,:,:sc.channels]
                for _ in range(2):compute()
                torch.cuda.synchronize();samples=[]
                for _ in range(5):
                    start=time.perf_counter();compute();torch.cuda.synchronize();samples.append(time.perf_counter()-start)
                results.append(dict(arm=arm,seed=seed,calibration=mode,seconds=sum(samples)/len(samples),samples=samples))
                print('BENCHMARK',name,arm,seed,mode,results[-1]['seconds'],flush=True)
    for arm in NOISE_ARMS:
        seed=500000
        cal=torch.load(OUT/'calibration'/name/f'{arm}_{seed}.pt',map_location='cuda',weights_only=False)
        for mode in MODES:
            online=GridCalibrator(ALPHA.cuda(),cal['initial'],1);median=MedianCalibrator(cal['median'],1)
            def compute_noise():
                q=issued(sc,backbone,None,[r],[0],arm,None,None)
                if mode=='fixed_cqr':
                    lo,hi=q[:249],q[250:].flip(0);delta=cal['fixed'][:,None,:,None,None,None]
                    a,b=lo-delta,hi+delta;invalid=a>b;mid=(a+b)/2
                    q=torch.cat((torch.where(invalid,mid,a),q[249:250],torch.where(invalid,mid,b).flip(0)))
                    q=rearrange_quantiles(q)[0]
                elif mode=='online':
                    a,b=online.issue(q[:249],q[250:].flip(0));mid=median.issue(q[249]);q=torch.cat((a,mid[None],b.flip(0)))
                    online.pending=None;median.pending=None;q=rearrange_quantiles(q)[0]
                return (q*sc.std+sc.mean)[:,:,:sc.channels]
            compute_noise();torch.cuda.synchronize();samples=[]
            for _ in range(3):
                start=time.perf_counter();compute_noise();torch.cuda.synchronize();samples.append(time.perf_counter()-start)
            results.append(dict(arm=arm,seed=seed,calibration=mode,seconds=sum(samples)/len(samples),samples=samples))
            print('BENCHMARK',name,arm,seed,mode,results[-1]['seconds'],flush=True)
    path=OUT/'benchmarks'/f'{name}.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(0),results=results),indent=2)+'\n')

if __name__=='__main__':main()
