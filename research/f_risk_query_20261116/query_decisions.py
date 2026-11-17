"""Choose alert cutoffs on calibration negatives, then audit development only."""
from common import *
from query import QueryHead,mask,cdf,queries,target_queries
from query_ensemble import ecc
from risk import RiskHead,features
import pandas as pd
@torch.no_grad()
def main():
    setup();folder=OUT/'query';levels=torch.arange(1,100,device='cuda')/100;stats=load(folder/'statistics.pt','cuda');thresholds=stats['thresholds']
    path=folder/'calibration_event_probabilities.pt'
    if path.exists():packed=load(path)
    else:
        d=load(folder/'calibration.pt','cuda');ids=[q['id'] for q in d['queries'] if q['category']=='seen'];z=d['z'][:,ids].flatten(0,1);y=d['y'][:,ids].flatten(0,1);truth=(y[:,None,2:]>thresholds[None]).cpu();probs={}
        for a in ('X','P','F','XPF'):
            m=QueryHead().cuda().eval();m.load_state_dict(load(folder/(a+'.pt'))['model']);q=torch.cat([m.quantiles(z[i:i+1024],mask(a),levels) for i in range(0,len(z),1024)],1)
            probs[a]=cdf(q[:,:,2:],thresholds,levels).cpu()
        data=cache('calibration');m=model();qs=queries(True);hck=load(OUT/'risk/main/F.pt');h=RiskHead(hck['width']).cuda().eval();h.load_state_dict(hck['model']);v={a:[] for a in ('noise8','noise32','F_ECC8','F_ECC32')}
        for i in range(0,len(data['x']),3):
            ix=torch.arange(i,min(i+3,len(data['x'])),device='cuda');x=data['x'][ix];paths=torch.stack([m.rollout(x,500000+k,sign=1) for k in range(32)]);w=h.weights(*features(data,'F',ix))
            for count in (8,32):
                for a,fields in [(f'noise{count}',paths[:count]),(f'F_ECC{count}',ecc(paths[:count],data['p'][ix],w))]:
                    yy=target_queries(fields.flatten(0,1),qs).reshape(count,len(ix)*len(qs),4);p=(yy[:,:,None,2:]>thresholds[None,None]).float().mean(0);v[a].append(p.cpu())
            if i%15==0:print('DECISION_CAL',i,flush=True)
        probs.update({a:torch.cat(parts) for a,parts in v.items()});packed=dict(probs=probs,truth=truth);save(packed,path)
    dev=load(folder/'predictions.pt');ep=load(folder/'ensemble_predictions.pt');preds={a:v['event'] for a,v in dev['predictions'].items()};preds.update({a:v['event'] for a,v in ep.items()})
    y=dev['targets'];truth=(y[:,:,None,2:]>dev['thresholds'][None,None]);qs=dev['queries'];rows=[];baseline=[]
    for a,prob in preds.items():
        cp=packed['probs'][a];ct=packed['truth'];cut=torch.zeros(3,2)
        for t in range(3):
            for c in range(2):
                neg=cp[:,t,c][~ct[:,t,c]];cut[t,c]=torch.quantile(neg,.9) if len(neg) else 1.
        alert=prob>cut[None,None]
        for cat in ('seen','new_space','new_time','new_both'):
            ids=[q['id'] for q in qs if q['category']==cat]
            for c in range(2):
                for t in range(3):
                    yy=truth[:,ids,t,c];aa=alert[:,ids,t,c];pos=int(yy.sum());neg=int((~yy).sum());tp=int((aa&yy).sum());fp=int((aa&~yy).sum());fn=pos-tp
                    rows.append(dict(method=a,category=cat,channel=c,threshold_index=t,cutoff=float(cut[t,c]),positive=pos,negative=neg,false_positive=fp,false_negative=fn,fpr=fp/neg if neg else None,fnr=fn/pos if pos else None,calibration_fpr=float((cp[:,t,c][~ct[:,t,c]]>cut[t,c]).float().mean())))
    for cat in ('seen','new_space','new_time','new_both'):
        ids=[q['id'] for q in qs if q['category']==cat]
        for c in range(2):
            for t in range(3):
                rate=float(packed['truth'][:,t,c].float().mean());yy=truth[:,ids,t,c].float()
                baseline.append(dict(category=cat,channel=c,threshold_index=t,prevalence=float(yy.mean()),zero_probability_brier=float(yy.mean()),calibration_climatology=rate,calibration_climatology_brier=float(((rate-yy)**2).mean())))
    pd.DataFrame(rows).to_csv(folder/'decisions_calibration_fpr10.csv',index=False);pd.DataFrame(baseline).to_csv(folder/'event_baselines.csv',index=False)
    dump(folder/'decision_protocol.json',dict(cutoff='90th percentile of negative calibration probabilities, strict >; ties may be conservative',calibration_queries='seen only',selection_uses_development=False,missing_fnr='null when no positive events',no_coverage_guarantee=True))
if __name__=='__main__':main()
