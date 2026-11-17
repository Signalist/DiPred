"""Restore archived predictions and compressed CSVs, verifying SHA-256 first."""
from pathlib import Path
import hashlib,json,shutil,tarfile,gzip
R=Path(__file__).resolve().parent
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def join(info,target):
 with target.open('wb') as out:
  for p in info['parts']:
   with (R/p).open('rb') as src:shutil.copyfileobj(src,out)
 if sha(target)!=info['sha256']:raise ValueError('SHA-256 mismatch: '+str(target))
def main():
 info=json.loads((R/'prediction_archive.json').read_text());archive=R/'archives/predictions.tar.gz';join(info,archive)
 with tarfile.open(archive,'r:gz') as tar:
  assert sorted(tar.getnames())==sorted(info['contents'])
  tar.extractall(R)
 for name,entry in json.loads((R/'split_tables.json').read_text()).items():join(entry,R/name)
 for p in R.rglob('*.csv.gz'):
  with gzip.open(p,'rb') as src,p.with_suffix('').open('wb') as dst:shutil.copyfileobj(src,dst)
 print('Restored predictions and CSV tables; archive hashes verified.')
if __name__=='__main__':main()
