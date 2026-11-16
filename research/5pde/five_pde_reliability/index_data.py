"""Read-only inventory of real RealPDEBench Arrow trajectories and strict splits."""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pyarrow as pa

ROOT=Path(__file__).resolve().parent.parent
DATA=Path(os.environ['FIVE_PDE_DATA']).resolve()
OUT=ROOT/'runs/five_pde_reliability_20261114'
NAMES=('cylinder','controlled_cylinder','fsi','foil','combustion')


def inventory(name):
    files=sorted(glob.glob(str(DATA/name/'hf_dataset/real/*.arrow')))
    expected={}
    for group in ('remain','in_dist_test','out_dist_test'):
        for key,value in json.loads((DATA/name/f'{group}_params_real.json').read_text()).items():
            # Test-group precedence prevents overlapping IDs entering training.
            expected[key]=(group,value)
    records=[]
    for path in files:
        with pa.memory_map(path,'r') as source:
            reader=pa.ipc.open_stream(source)
            for batch_index,batch in enumerate(reader):
                for row in range(batch.num_rows):
                    def value(key):return batch.column(batch.schema.get_field_index(key))[row].as_py()
                    sim=value('sim_id')
                    group,parameters=expected.get(sim,('unlisted_test',None))
                    records.append(dict(id=sim,official_group=group,parameters=parameters,
                        Re=(None if name=='combustion' else float(sim[:-3].split('_')[0])),
                        shard=str(Path(path).relative_to(DATA)),batch=batch_index,row=row,
                        frames=value('shape_t'),height=value('shape_h'),width=value('shape_w')))
    ids=[r['id'] for r in records]
    if len(ids)!=len(set(ids)):
        raise ValueError(f'{name}: duplicate Arrow IDs')
    missing=set(expected)-set(ids)
    if missing:print('MISSING_METADATA_IDS',name,sorted(missing),flush=True)
    rng=np.random.default_rng(20261114)
    remaining=sorted(r['id'] for r in records if r['official_group']=='remain')
    rng.shuffle(remaining)
    # Official in/out groups are kept entirely for test. Combustion has none,
    # so reserve 20% of its real trajectories before assigning train/val/cal.
    test=set(r['id'] for r in records if r['official_group']!='remain')
    if not test:
        n=max(1,round(.2*len(remaining)));test.update(remaining[:n]);remaining=remaining[n:]
    nval=max(1,round(.15*len(remaining)));ncal=max(1,round(.15*len(remaining)))
    valid=set(remaining[:nval]);cal=set(remaining[nval:nval+ncal]);train=set(remaining[nval+ncal:])
    for r in records:
        r['split']='test' if r['id'] in test else 'development' if r['id'] in valid else 'calibration' if r['id'] in cal else 'train'
    assert all(any(r['split']==s for r in records) for s in ('train','development','calibration','test'))
    return records


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result={}
    for name in NAMES:
        cached=OUT/f'inventory_{name}.json'
        records=json.loads(cached.read_text()) if cached.exists() else inventory(name)
        cached.write_text(json.dumps(records,indent=2)+'\n');result[name]=records
        counts={s:sum(r['split']==s for r in records) for s in ('train','development','calibration','test')}
        print(name,counts,'frames',sorted(set(r['frames'] for r in records)),
              'shapes',sorted(set((r['height'],r['width']) for r in records)),flush=True)
    (OUT/'splits.json').write_text(json.dumps(dict(seed=20261114,dataset='real',
        source_revision='0ae4426085d250787f96d5456bc0d30ce5f277d2',scenarios=result),indent=2)+'\n')

if __name__=='__main__':main()
