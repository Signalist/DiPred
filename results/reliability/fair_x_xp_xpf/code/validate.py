"""Fixed-protocol development evaluation; results never select weights."""
import csv
import sys
import time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'fair_x_xp_xpf'))
sys.path.insert(0,str(ROOT))
from evaluate import OUT, ARMS, SEEDS, load_model, input_batch
from cache_inputs import fields
from clean_reliability_common import configure, manifest
from experiment_reliability_transport_e12 import normalization

@torch.no_grad()
def main():
    configure(); norm=normalization(); records=manifest()['files']['development']
    results=[]
    for seed in SEEDS:
        for arm in ARMS:
            model,_=load_model(arm,seed)
            value=0.
            for start in range(0,161,20):
                x,p,f,mask=input_batch('development',start,arm)
                tau=torch.tensor([.05,.5,.95],device='cuda')
                prediction=model.quantiles(x,p,f,mask,tau)
                y=(fields('development',start,records,True)-norm['mean'])/norm['std']
                error=y[None]-prediction;tau=tau[:,None,None,None,None,None]
                value+=float((torch.maximum(tau*error,(tau-1)*error)*norm['std']).mean())/9
            results.append((arm,seed,value))
            print('DEVELOPMENT',arm,seed,value,flush=True)
    path=OUT/'validation.csv'
    with path.open('w',newline='') as handle:
        w=csv.writer(handle);w.writerow(('method','seed','mean_pinball_05_50_95'));w.writerows(results)
    (OUT/'validation_complete.txt').write_text(str(time.time())+'\n')

if __name__=='__main__':main()
