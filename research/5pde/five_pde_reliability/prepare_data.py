"""Materialize read-only Arrow fields on a common grid, without split leakage."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import pyarrow as pa

ROOT=Path(__file__).resolve().parent.parent
DATA=Path(os.environ['FIVE_PDE_DATA']).resolve()
OUT=ROOT/'runs/five_pde_reliability_20261114'


def basename(record):return hashlib.sha256(record['id'].encode()).hexdigest()[:16]+'.npy'


def preprocess(name,records):
    folder=OUT/'fields'/name;folder.mkdir(parents=True,exist_ok=True)
    by_shard={}
    for r in records:by_shard.setdefault(r['shard'],[]).append(r)
    channels=('observed',) if name=='combustion' else ('u','v')
    for shard,entries in sorted(by_shard.items()):
        if all((folder/basename(r)).exists() for r in entries):continue
        with pa.memory_map(str(DATA/shard),'r') as source:
            batches=pa.ipc.open_stream(source)
            for batch_index,batch in enumerate(batches):
                for record in (r for r in entries if r['batch']==batch_index):
                    target=folder/basename(record)
                    if target.exists():continue
                    t,h,w=record['frames'],record['height'],record['width']
                    if h%32 or w%64:raise ValueError(f'Unsupported grid {h}x{w}')
                    temp=target.with_suffix('.tmp')
                    dest=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(t,len(channels),32,64))
                    for c,key in enumerate(channels):
                        scalar=batch.column(batch.schema.get_field_index(key))[record['row']]
                        arr=np.frombuffer(scalar.as_buffer(),dtype=np.float32)
                        if arr.size!=t*h*w:raise ValueError(f'Field shape mismatch {name} {record["id"]} {key}')
                        arr=arr.reshape(t,h,w)
                        for start in range(0,t,256):
                            block=arr[start:start+256,::h//32,::w//64]
                            if not np.isfinite(block).all():raise ValueError(f'Nonfinite field {name} {record["id"]}')
                            dest[start:start+256,c]=block
                    dest.flush();del dest
                    os.replace(temp,target)
                    print('PREPARED',name,record['id'],record['split'],flush=True)
    # Compute normalization exclusively over all frames of training trajectories.
    count=0;sums=np.zeros(len(channels),np.float64);squares=np.zeros_like(sums);diffsquares=np.zeros_like(sums);diffcount=0
    for record in records:
        if record['split']!='train':continue
        arr=np.load(folder/basename(record),mmap_mode='r')
        previous=None
        for start in range(0,len(arr),256):
            block=arr[start:start+256].astype(np.float64)
            sums+=block.sum(axis=(0,2,3));squares+=(block*block).sum(axis=(0,2,3));count+=block.shape[0]*32*64
            if previous is not None:
                diff=block[:1]-previous;diffsquares+=(diff*diff).sum(axis=(0,2,3));diffcount+=32*64
            diff=np.diff(block,axis=0);diffsquares+=(diff*diff).sum(axis=(0,2,3));diffcount+=len(diff)*32*64
            previous=block[-1:]
    mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean*mean,0))
    if (std<1e-8).any():raise ValueError(f'Degenerate scale {name}')
    metadata=dict(channels=list(channels),mean=mean.tolist(),std=std.tolist(),
                  increment_rms_normalized=(np.sqrt(diffsquares/diffcount)/std).clip(1e-4).tolist(),
                  grid=[32,64],grid_method='deterministic stride from stored grid',
                  stats='training trajectories only, all their frames',count_per_channel=count)
    (OUT/f'normalization_{name}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('PREPARATION_COMPLETE',name,metadata,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scenario');args=parser.parse_args()
    if args.scenario:
        preprocess(args.scenario,json.loads((OUT/f'inventory_{args.scenario}.json').read_text()))
    else:
        for name,records in json.loads((OUT/'splits.json').read_text())['scenarios'].items():preprocess(name,records)
        (OUT/'preparation_complete.json').write_text(json.dumps({'complete':True})+'\n')

if __name__=='__main__':main()
