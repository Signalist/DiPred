"""Only fixed native-feedback keys; W computed once, no training here."""
import time
from .core import *


def build(split,xpf,scales=None):
    norm={k:v.cpu() for k,v in xpf.norm.items()};rs=records();files=[r['file'] for r in rs[split]]
    if split=='test':
        rows=read(AUDIT/'TEST_CATALOG.json')
    else:rows=[r for r in read(AUDIT/'WINDOW_MANIFEST.json') if r['split']==split]
    entries=[];manifest=[];audit=[];started=time.perf_counter()
    lock=read(AUDIT/'CACHE_MANIFEST.json');reader=H5Reader(DATA)
    for s in sorted({r['start'] for r in rows}):
        p=ROOT/f'.cache/error_guided_spatial_routing/{split}_{s-20}_native.pt'
        assert p.exists(),f'missing feedback {p}'
        # Test feedback was locked in original protected-cache manifest.
        sha=digest(p)
        expected=(lock.get(str(p.relative_to(ROOT))) or {}).get('sha256')
        if expected is None:expected=read(AUDIT/'PROTECTED_BEFORE.json')[str(p.relative_to(ROOT))]
        assert sha==expected
        index=CacheIndex(torch.load(p,map_location='cpu',weights_only=False),split,s-20,files,str(p.relative_to(ROOT)))
        for row in rows:
            if row['start']!=s:continue
            key=Key(split,row['trajectory_id'],s);previous=index.lookup(Key(split,key.trajectory_id,s-20))
            item=release(reader,key,previous,norm,xpf)
            if split!='test':
                assert item['history_sha256']==row['history_sha256'] and item['previous_path_sha256']==row['previous_path_sha256']
            if len(audit)<2:
                audit.append(dict(split=split,trajectory_id=key.trajectory_id,start=s,**xpf.verify_equivalence(reader(key.trajectory_id,s,s+20))))
            # Target read occurs after release; never enters XPF or E.
            item['target']=reader(key.trajectory_id,s+20,s+40)
            item['Re']=rs[split][files.index(key.trajectory_id)]['group']
            item['cache_key']=f'{split}__{key.trajectory_id}__{s}'
            entries.append(item)
            manifest.append(dict(split=split,trajectory_id=key.trajectory_id,start=s,Re=item['Re'],cache_key=item['cache_key'],
               history_interval=[s,s+20],previous_target_interval=[s,s+20],target_interval=[s+20,s+40],
               history_sha256=item['history_sha256'],E_sha256=item['E_sha256'],W_sha256=item['W_sha256'],
               previous_path_sha256=item['previous_path_sha256'],target_sha256=tensor_hash(item['target']),
               feedback_cache=str(p.relative_to(ROOT)),feedback_cache_sha256=sha,feedback_cache_row=previous.cache_row,
               xpf_checkpoint_sha256=digest(ROOT/'reliability/checkpoints/selected16_comparison.pt'),
               W_source='frozen native zero-noise four-stage point/features; raw XPF16 half width; no online calibrator'))
        del index
    if split=='train':
        assert len(entries)==137
        e=torch.cat([i['E'].flatten() for i in entries]);w=torch.stack([i['W'] for i in entries])
        scales=dict(E_clip=float(torch.quantile(e.abs(),.995).clamp_min(.1)),
           W_scale=[float(v) for v in w.square().mean((0,2,3,4)).sqrt().clamp_min(1e-6)],
           W_clip=10.,source_split='train',train_windows=137,E_clip_quantile=.995,
           W_normalization='fixed training RMS per velocity channel; no window normalization')
        dump(OUT/'CONDITION_SCALES.json',scales)
    elif split=='development':assert len(entries)==36
    else:assert len(entries)==12
    torch.save(entries,CACHE/f'{split}_conditions.pt')
    dump(OUT/f'{split}_DATA_MANIFEST.json',manifest)
    dump(OUT/f'{split}_XPF_AUDIT.json',dict(passed=True,examples=audit,entries=len(entries),calls_per_window=4,
       xpf_weight_state_sha256=xpf.weight_hash,seconds=time.perf_counter()-started,cache_sha256=digest(CACHE/f'{split}_conditions.pt')))
    return entries,scales


def main():
    if (CACHE/'train_conditions.pt').exists():raise RuntimeError('refuse to overwrite conditions')
    xpf=FrozenXPF();train,scales=build('train',xpf);dev,_=build('development',xpf,scales)
    dump(OUT/'DATA_MANIFEST.json',read(OUT/'train_DATA_MANIFEST.json')+read(OUT/'development_DATA_MANIFEST.json'))
    dump(OUT/'XPF_CONDITION_AUDIT.json',dict(passed=True,train=read(OUT/'train_XPF_AUDIT.json'),development=read(OUT/'development_XPF_AUDIT.json')))
    print('CONDITIONS READY: train137 development36; zero updates',flush=True)

if __name__=='__main__':main()
