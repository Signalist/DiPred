"""Time single-trajectory prediction including the specified calibration mode."""
import json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
for sub in ('','fair_x_xp_xpf','xpf16_fullpercent_study','xpf16_02grid_2048','github/DiPred'):
    sys.path.insert(0,str(ROOT/sub))
from fair_x_xp_xpf.evaluate import OUT, ARMS, SEEDS, LEVELS, ALPHA, load_model, input_batch
from clean_reliability_common import configure,manifest
from experiment_reliability_transport_e12 import geometry
from reliability.selected16_predict import RequestedThreePredictor
from reliability.features import forecast_features
from calibration_grid import GridCalibrator,MedianCalibrator
from distribution_grid import rearrange_quantiles

@torch.no_grad()
def main():
    configure();predictor=RequestedThreePredictor(geometry(manifest()['files']['development'][:1]),'selected16','cuda')
    out=[]
    for seed in SEEDS:
        for arm in ARMS:
            model,_=load_model(arm,seed);x,p,f,mask=input_batch('development',0,arm)
            x,p,f,mask=x[:1],p[:1],f[:1],mask[:1]
            cal=torch.load(OUT/'calibration'/f'{arm}_{seed}.pt',map_location='cuda',weights_only=False)
            for mode in ('raw','fixed_cqr','online'):
                online=GridCalibrator(ALPHA.cuda(),cal['initial'],1)
                median=MedianCalibrator(cal['median'],1)
                def inference():
                    if arm=='X':
                        pp=torch.zeros_like(p);ff=torch.zeros_like(f)
                    elif arm=='XP':
                        point=predictor.model.rollout(x,500000,sign=0)
                        pp=((point-predictor.stats['base_mean'][:,:2])/predictor.stats['base_std'][:,:2]).clamp(-10,10)
                        ff=torch.zeros_like(f)
                    else:
                        point,last=forecast_features(predictor.model,x)
                        pp=((point-predictor.stats['base_mean'][:,:2])/predictor.stats['base_std'][:,:2]).clamp(-10,10)
                        ff=((last[:,-48:].half().float()-predictor.stats['last_mean'])/predictor.stats['last_std']).clamp(-10,10)
                    q=model.quantiles(x,pp,ff,mask,LEVELS.cuda().float())
                    if mode=='fixed_cqr':
                        lo,hi=q[:249],q[250:].flip(0);delta=cal['fixed'][:,None,:,None,None,None]
                        a,b=lo-delta,hi+delta;invalid=a>b;mid=(a+b)/2
                        q=torch.cat((torch.where(invalid,mid,a),q[249:250],torch.where(invalid,mid,b).flip(0)))
                        q=rearrange_quantiles(q)[0]
                    elif mode=='online':
                        a,b=online.issue(q[:249],q[250:].flip(0));m=median.issue(q[249])
                        q=torch.cat((a,m[None],b.flip(0)))
                        online.pending=None;median.pending=None
                        q=rearrange_quantiles(q)[0]
                    return q*predictor.norm['std']+predictor.norm['mean']
                for _ in range(2):inference()
                torch.cuda.synchronize();times=[]
                for _ in range(5):
                    t=time.perf_counter();inference();torch.cuda.synchronize();times.append(time.perf_counter()-t)
                out.append(dict(method=arm,seed=seed,calibration=mode,seconds=sum(times)/len(times),samples=times))
                print('MODE_BENCHMARK',arm,seed,mode,out[-1]['seconds'],flush=True)
    (OUT/'benchmark_modes.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
