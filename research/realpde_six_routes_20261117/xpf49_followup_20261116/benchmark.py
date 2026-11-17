from followcommon import *
from query_xpf import Head,mask_for
from experiment_selected16_quantile import new_head
from features import AblatedHead
from query import inputs
import csv
@torch.no_grad()
def main():
 setup();d=cache('development');x=d['x'][:1];pmodel=model();oldstats=load(PRIOR/'query/statistics.pt','cuda');qstats=load(OUT/'query/qstats.pt','cuda');heads=[]
 for k in (5,25,49):
  ck=load(OLD/'models'/f'XPF16-Q{k:02d}'/'model.pt');h=new_head('compact16',None);h.load_state_dict(ck['model']);heads.append(AblatedHead(h.eval(),'XPF'))
 models={}
 for a in ('XP','XPF','XPQ','XPFQ'):
  h=Head().cuda().eval();h.load_state_dict(load(OUT/'query'/f'{a}.pt')['model']);models[a]=h
 fstats=load(PRIOR/'feature_stats.pt','cuda');levels=torch.tensor([.05,.5,.95],device='cuda');rows=[]
 for count in (1,8,32,160):
  qs=queries()[:count]
  for arm in models:
   times=[]
   for rep in range(12):
    torch.cuda.synchronize();t=time.perf_counter();p,f=feature(pmodel,x,'diffusion');normf=((f-fstats['f'][0])/fstats['f'][1]).clamp(-10,10)
    z=inputs(dict(x=x,p=p,f=normf),qs).flatten(0,1);z[:,:384]=((z[:,:384]-oldstats['mean'])/oldstats['std']).clamp(-10,10)
    if 'Q' in arm:
     base,last=features(x,p,f);bounds=[h(base,last) for h in heads];fields=torch.stack([v for pair in bounds for v in pair],1).flatten(1,3);byrect={}
     for q in qs:
      rect=tuple(q['rect'])
      if rect not in byrect:byrect[rect]=pool(fields,rect)
     zq=torch.stack([byrect[tuple(q['rect'])] for q in qs],1).flatten(0,1);zq=((zq-qstats['mean'])/qstats['std']).clamp(-10,10)
    else:zq=torch.zeros(count,720,device='cuda')
    out=models[arm].quantiles(torch.cat((z,zq),1),mask_for(arm),levels);torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
    if rep>=2:times.append(elapsed)
   rows.append(dict(method=arm,queries=count,median_ms=float(np.median(times)),p10_ms=float(np.quantile(times,.1)),p90_ms=float(np.quantile(times,.9)),batch=1,repeats=10));print('LATENCY',rows[-1],flush=True)
 with (OUT/'query/latency.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 dump(OUT/'query/latency_protocol.json',dict(gpu=torch.cuda.get_device_name(),batch=1,warmup=2,repeats=10,includes='four call DiPred; actual 3 original XPF heads if Q; query pooling and 3 predicted quantiles',excludes='model loading, CPU transfer, feedback',noise='prior noise latency generated once per group of queries; separate run under different load'))
if __name__=='__main__':main()
