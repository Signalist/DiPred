"""Verify train-only cached grid values against the original Arrow bytes."""
import argparse,json
import numpy as np
import pyarrow as pa
from common import ROOT,OUT,records
from five_pde_reliability.prepare_data import DATA,basename

def main():
    parser=argparse.ArgumentParser();parser.add_argument('scenario');args=parser.parse_args();name=args.scenario
    rs=records(name);channels=('observed',) if name=='combustion' else ('u','v');checks=[]
    for split in ('train','development','calibration'):
        r=next(a for a in rs if a['split']==split)
        cached=np.load(OUT/'fields'/name/basename(r),mmap_mode='r')
        with pa.memory_map(str(DATA/r['shard']),'r') as source:
            for i,batch in enumerate(pa.ipc.open_stream(source)):
                if i!=r['batch']:continue
                for c,key in enumerate(channels):
                    raw=np.frombuffer(batch.column(batch.schema.get_field_index(key))[r['row']].as_buffer(),dtype=np.float32).reshape(r['frames'],r['height'],r['width'])
                    for start in (0,(r['frames']-40)//40*20):
                        expected=raw[start:start+20,::r['height']//32,::r['width']//64]
                        checks.append(bool(np.array_equal(expected,cached[start:start+20,c])))
                break
    result=dict(scenario=name,passed=all(checks),comparisons=len(checks),test_targets_read=False)
    (OUT/'field_audit').mkdir(parents=True,exist_ok=True)
    (OUT/'field_audit'/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True);assert result['passed']

if __name__=='__main__':main()
