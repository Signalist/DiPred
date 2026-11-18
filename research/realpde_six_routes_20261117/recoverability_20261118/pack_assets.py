"""Package the frozen primary experiment weights without duplicating PCA storage."""
from pathlib import Path
import sys,json,shutil,tarfile,hashlib
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'recoverability_20261118'))
import evaluation_recoverability as ev
O=ev.OUT;P=O/'primary_assets';P.mkdir(exist_ok=True)

def main():
 routers={s:ev.ctl.load(O/f'router_{s}_compact.pt') for s in ev.SEEDS};common=routers[42]['projection']
 for seed in ev.SEEDS:
  for k in common:assert torch.equal(common[k],routers[seed]['projection'][k]);
  for role in routers[seed]['models']:assert torch.isfinite(routers[seed]['models'][role]['coef']).all()
  for method in ['warp','rescrct_original']:shutil.copyfile(O/f'{method}_{seed}.pt',P/f'{method}_{seed}.pt')
 torch.save(dict(projection=common,routers={str(s):routers[s]['models'] for s in ev.SEEDS}),P/'routers_shared.pt')
 for path in [O/'selection.json',O/'compact_assets.json',W/'runs/clean_direct20_20261107/manifest.json',W/'recoverability_20261118/PROTOCOL.md']:
  shutil.copyfile(path,P/path.name)
 manifest={'backbone_repository_path':'reliability/checkpoints/current.pt','backbone_git_blob':'3730649c35c62b5abbcf9d12165d2e4522fc634e','backbone_sha256':ev.ctl.digest(ev.ctl.SHARED),'data':'RealPDE Competition train_real, public release; held-out identities in manifest.json','files':{p.name:ev.ctl.digest(p) for p in sorted(P.iterdir()) if p.is_file() and p.name!='asset_manifest.json'}}
 (P/'asset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 with tarfile.open(O/'primary_assets.tar.gz','w:gz') as tar:
  for p in sorted(P.iterdir()):tar.add(p,arcname=p.name,recursive=False)
 archive=O/'primary_assets.tar.gz';print('PRIMARY_ASSETS',archive.stat().st_size,ev.ctl.digest(archive),flush=True)
if __name__=='__main__':main()
