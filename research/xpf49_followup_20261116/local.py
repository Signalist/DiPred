"""Calibrate the unchanged actual Q05 head, comparing XPF versus XP."""
from followcommon import *
from evaluate_risk import groups
from experiment_selected16_quantile import new_head
from features import AblatedHead
import csv
@torch.no_grad()
def main():
 setup();pmodel=model();threshold=load(PRIOR/'gradient_threshold.pt','cuda');data={s:cache(s) for s in ('calibration','development')};rows=[];parameters={}
 for arm in ('XPF','XP'):
  path=OLD/'models/XPF16-Q05/model.pt' if arm=='XPF' else ABL/'models/XP/Q05.pt';ck=load(path);h=new_head('compact16',None);h.load_state_dict(ck['model']);h=AblatedHead(h.eval(),arm)
  b={}
  for split in data:
   pieces=[]
   for i in range(0,len(data[split]['x']),8):
    x=data[split]['x'][i:i+8];p,f=feature(pmodel,x,'diffusion');base,last=features(x,p,f);lo,hi=h(base,last);pieces.append(torch.stack((lo,hi)).cpu())
   b[split]=torch.cat(pieces,1).cuda()
  cal=data['calibration'];dev=data['development'];r=cal['y']-cal['p'];lo,hi=b['calibration'];center=(lo+hi)/2;half=((hi-lo)/2).clamp_min(1e-5);score=(r-center).abs()/half
  cl=(b['development'][0]+b['development'][1])/2;dh=((b['development'][1]-b['development'][0])/2).clamp_min(1e-5);cm=groups(cal['p'],threshold);dm=groups(dev['p'],threshold)
  for target in (.8,.85,.9,.95):
   factors={g:torch.stack([torch.quantile(score[:,c][s[:,c]],target) for c in range(2)]) for g,s in cm.items()}
   for strategy in ('global','groups'):
    scale=torch.zeros_like(dev['p'])
    for g in (['all'] if strategy=='global' else dm):scale=torch.maximum(scale,torch.where(dm[g],factors[g][None,:,None,None,None],torch.zeros_like(scale)))
    lower=cl-dh*scale;upper=cl+dh*scale;y=dev['y']-dev['p'];values=dict(coverage=((y>=lower)&(y<=upper)).float(),width=upper-lower,interval=(upper-lower)+20*((lower-y).clamp_min(0)+(y-upper).clamp_min(0)))
    for group,sel in dm.items():
     count=sel.sum((2,3,4));v={k:((val*sel).sum((2,3,4))/count.clamp_min(1)).cpu().numpy() for k,val in values.items()};ct=count.cpu().numpy()
     for i,meta in enumerate(dev['meta']):
      for c in range(2):
       if ct[i,c]:rows.append(dict(arm=arm,strategy=strategy,target=target,region=group,channel=c,**meta,**{k:float(val[i,c]) for k,val in v.items()}))
    parameters[f'{arm}_{strategy}_{target}']={g:v.tolist() for g,v in factors.items()}
  print('LOCAL',arm,flush=True)
 folder=OUT/'local';folder.mkdir(exist_ok=True)
 with (folder/'per_group.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 dump(folder/'parameters.json',parameters)
if __name__=='__main__':main()
