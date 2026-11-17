"""Extract 3 frozen actual XPF16-head outputs for query on RealPDE only."""
from followcommon import *
from experiment_selected16_quantile import new_head
from features import AblatedHead
from query import inputs
@torch.no_grad()
def main():
 setup();predictor=model();heads=[]
 for k in (5,25,49):
  ck=load(OLD/'models'/f'XPF16-Q{k:02d}'/'model.pt');h=new_head('compact16',None).cuda().eval();h.load_state_dict(ck['model']);heads.append(AblatedHead(h,'XPF'))
 hashes={f'Q{k:02d}':digest(OLD/'models'/f'XPF16-Q{k:02d}'/'model.pt') for k in (5,25,49)}
 for split in ('train','calibration','development'):
  dest=OUT/'query'/f'{split}.pt'
  if dest.exists():continue
  data=cache(split);prior=load(PRIOR/'query'/f'{split}.pt');qs=prior['queries'];z=[]
  for i in range(0,len(data['x']),8):
   x=data['x'][i:i+8];p,f=feature(predictor,x,'diffusion')
   if not torch.allclose(p,data['p'][i:i+8],atol=2e-5,rtol=1e-5):raise RuntimeError('Predictor differs from actual historical cache')
   base,last=features(x,p,f);bounds=[h(base,last) for h in heads]
   fields=torch.stack([v for pair in bounds for v in pair],1).flatten(1,3)
   byrect={}
   for q in qs:
    r=tuple(q['rect'])
    if r not in byrect:byrect[r]=pool(fields,r)
   z.append(torch.stack([byrect[tuple(q['rect'])] for q in qs],1).cpu())
   if i%80==0:print('XPF49_QUERY_EXTRACT',split,i,len(data['x']),flush=True)
  qfeatures=torch.cat(z)
  if split=='train':
   mean=qfeatures.mean((0,1));std=qfeatures.std((0,1)).clamp_min(.05);save(dict(mean=mean,std=std),OUT/'query/qstats.pt')
  stats=load(OUT/'query/qstats.pt');qfeatures=((qfeatures-stats['mean'])/stats['std']).clamp(-10,10)
  save(dict(z=torch.cat((prior['z'],qfeatures),2),y=prior['y'],queries=qs,meta=prior['meta']),dest)
  print('XPF49_QUERY_SAVED',split,qfeatures.shape,flush=True)
 dump(OUT/'query/extraction.json',dict(heads=[5,25,49],description='Actual original XPF16 head output residuals 05/95, 25/75, 49/51. Spatial mean/std/max; no future targets.',hashes=hashes,split=['train','calibration','development'],seed=42))
if __name__=='__main__':main()
