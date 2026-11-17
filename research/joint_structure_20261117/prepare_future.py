from experiment import *
from common import read,records
def main():
    setup();n=norm('cpu');path=OUT/'future_data.pt'
    if not path.exists():
        values=[];meta=[]
        for start in range(600,801,40):
            for r in records('development'):
                if start+40<=r['frames']:
                    values.append(read(r,start));meta.append(dict(file=r['file'],group=r['group'],start=start))
        raw=torch.stack(values);save(dict(x=(raw[:,:,:20]-n['mean'])/n['std'],y=(raw[:,:,20:]-n['mean'])/n['std'],meta=meta),path)
    d=load(path)
    for stage in ('physics','helmholtz_filter'):
        link=OUT/stage/'future_data.pt'
        if not link.exists():link.symlink_to(path)
    dump(OUT/'future_protocol.json',dict(starts=list(range(600,801,40)),windows=len(d['meta']),trajectories=len({m['file'] for m in d['meta']}),selection_hashes={k:digest(OUT/k/'selection.json') for k in ('physics','helmholtz_filter')},scope='temporally disjoint later windows of existing development trajectories; selected methods frozen before extraction; not a fresh independent Re test'))
    print('FUTURE_READY',len(d['meta']),flush=True)
if __name__=='__main__':main()
