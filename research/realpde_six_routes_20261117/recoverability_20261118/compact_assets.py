"""Remove shared-storage bloat without changing tensors or frozen originals."""
from pathlib import Path
import sys,json
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'recoverability_20261118'))
import evaluation_recoverability as ev
O=ev.OUT

def compact(x):
 if isinstance(x,torch.Tensor):return x.clone().contiguous()
 if isinstance(x,dict):return {k:compact(v) for k,v in x.items()}
 if isinstance(x,list):return [compact(v) for v in x]
 if isinstance(x,tuple):return tuple(compact(v) for v in x)
 return x

def main():
 rows={}
 for seed in ev.SEEDS:
  p=O/f'router_{seed}.pt';q=O/f'router_{seed}_compact.pt';old=ev.ctl.load(p);new=compact(old);ev.ctl.save(new,q);check=ev.ctl.load(q)
  for k in old['projection']:assert torch.equal(old['projection'][k],check['projection'][k])
  for role in old['models']:assert torch.equal(old['models'][role]['coef'],check['models'][role]['coef'])
  rows[str(seed)]=dict(original_bytes=p.stat().st_size,compact_bytes=q.stat().st_size,original_sha256=ev.ctl.digest(p),compact_sha256=ev.ctl.digest(q),exact_tensor_equality=True)
 ev.ctl.dump(O/'compact_assets.json',rows);print(rows,flush=True)
if __name__=='__main__':main()
