"""Fixed later-window confirmation on the same development trajectories."""
from experiment import *
from common import read,records,DATA
import h5py

@torch.no_grad()
def main():
    setup();n=norm('cpu');source=OUT/'late_data.pt'
    if not source.exists():
        values=[];meta=[]
        for start in range(200,561,40):
            for r in records('development'):
                if start+40>r['frames']:continue
                values.append(read(r,start));meta.append(dict(file=r['file'],group=r['group'],start=start))
        raw=torch.stack(values);save(dict(x=(raw[:,:,:20]-n['mean'])/n['std'],y=(raw[:,:,20:]-n['mean'])/n['std'],meta=meta),source)
    d=load(source)
    for stage in ('physics','advection'):
        link=OUT/stage/'late_data.pt'
        if not link.exists():link.symlink_to(source)
    grid=[]
    for split in ('train','calibration','development'):
        for r in records(split):
            with h5py.File(DATA/r['file'],'r') as f:
                x=f['x'][::2,::2];y=f['y'][::2,::2];t=f['t'][:]
                dx=np.diff(x,axis=1);dy=np.diff(y,axis=0);dt=np.diff(t)
                assert np.max(np.abs(dx-.003421664))<1e-8 and np.max(np.abs(dy+.003421664))<1e-8
                assert np.max(np.abs(dt-.02))<2e-6
                grid.append(dict(file=r['file'],split=split,dx=float(dx.mean()),dy=float(dy.mean()),dt=float(dt.mean())))
    dump(OUT/'late_protocol.json',dict(windows=len(d['meta']),trajectories=len({v['file'] for v in d['meta']}),starts=list(range(200,561,40)),source='same development Re and trajectories, later disjoint forecast times; not a fresh independent test',selection='calibration only, no later-window tuning',seed=42,grid_checks=grid))
    print('LATE_READY',len(d['meta']),flush=True)

if __name__=='__main__':main()
