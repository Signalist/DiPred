"""Recompute features from the actual frozen models; split by trajectory/Re."""
from common import *
import time
@torch.no_grad()
def main():
    setup();n=norm(); models={k:model(k) for k in ('diffusion','deterministic','random')}
    groups=json.loads(MANIFEST.read_text())['files']
    for a in groups:
        for b in groups:
            if a!=b:
                assert {r['file'] for r in groups[a]}.isdisjoint({r['file'] for r in groups[b]})
                assert {r['group'] for r in groups[a]}.isdisjoint({r['group'] for r in groups[b]})
    for split in ('train','calibration','development'):
        dest=OUT/'data'/f'{split}.pt'
        if dest.exists():continue
        pairs=[(r,s) for s in (list(range(200,361,20))+list(range(400,561,20)) if split=='train' else range(0,161,20)) for r in records(split) if (r['frames'] >= (400 if s<400 else 600) if split=='train' else s+40<=r['frames'])]
        values={k:[] for k in ('x','y','p','f','fd','fr','pd')};meta=[]
        for j in range(0,len(pairs),12):
            batch=pairs[j:j+12]; raw=torch.stack([read(r,s) for r,s in batch]).cuda()
            x=(raw[:,:,:20]-n['mean'])/n['std']
            # Future is never passed into any frozen model or feature function.
            p,f=feature(models['diffusion'],x,'diffusion')
            pd,fd=feature(models['deterministic'],x,'deterministic')
            _,fr=feature(models['random'],x,'random')
            y=(raw[:,:,20:]-n['mean'])/n['std']
            for k,v in zip(values,(x,y,p,f,fd,fr,pd)):values[k].append(v.cpu())
            meta.extend([dict(file=r['file'],group=r['group'],start=s) for r,s in batch])
            if j%120==0:print('PREPARE',split,j,len(pairs),flush=True)
        value={k:torch.cat(v) for k,v in values.items()};value['meta']=meta
        if split=='train':
            stats={}
            for k in ('f','fd','fr'):
                a=value[k];stats[k]=(a.mean((0,2,3),keepdim=True),a.std((0,2,3),keepdim=True).clamp_min(.05))
            save(stats,OUT/'feature_stats.pt')
        for k,(mu,std) in load(OUT/'feature_stats.pt').items():value[k]=((value[k]-mu)/std).clamp(-10,10)
        assert len(meta)==(819 if split=='train' else len(records(split))*9)
        save(value,dest);print('SAVED',split,len(meta),flush=True)
    dump(OUT/'data_audit.json',dict(seed=42,dataset=str(DATA),splits=['train','calibration','development'],test_read=False,
        train_windows=819,feature_source='fourth up2 of frozen original DiPred; no target inputs',
        deterministic_control='same architecture, 6000 MSE updates, ONE call; not a matched four-call objective isolation',
        sources={str(p):digest(p) for p in (SHARED,DET,MANIFEST)}))
if __name__=='__main__':main()
