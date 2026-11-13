"""Synchronized end-to-end inference latency for one 20-frame history."""
import json
import sys
import time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
for sub in ('', 'fair_x_xp_xpf', 'xpf16_fullpercent_study','github/DiPred'):
    sys.path.insert(0,str(ROOT/sub))
from fair_x_xp_xpf.evaluate import OUT, ARMS, SEEDS, LEVELS, load_model, input_batch
from clean_reliability_common import configure, manifest
from experiment_reliability_transport_e12 import geometry
from reliability.selected16_predict import RequestedThreePredictor
from reliability.features import forecast_features

@torch.no_grad()
def main():
    configure(); rec=manifest()['files']['development'][:1]
    predictor=RequestedThreePredictor(geometry(rec),'selected16','cuda')
    values=[]
    for seed in SEEDS:
        for arm in ARMS:
            m,state=load_model(arm,seed)
            x,p,f,mask=input_batch('development',0,arm)
            x,p,f,mask=x[:1],p[:1],f[:1],mask[:1]
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
                return m.quantiles(x,pp,ff,mask,LEVELS.cuda().float())
            for _ in range(3): inference()
            torch.cuda.synchronize();samples=[]
            for _ in range(10):
                t=time.perf_counter();inference();torch.cuda.synchronize();samples.append(time.perf_counter()-t)
            values.append(dict(method=arm,seed=seed,seconds=sum(samples)/len(samples),samples=samples,
                               parameters=state['parameters'],gpu=torch.cuda.get_device_name()))
            print('BENCHMARK',arm,seed,values[-1]['seconds'],flush=True)
    (OUT/'benchmark.json').write_text(json.dumps(values,indent=2)+'\n')

if __name__=='__main__':main()
