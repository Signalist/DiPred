"""Reconstruct published split CSV.gz files and verify their SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('result_directory',type=Path)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.result_directory/'split_data_manifest.json').read_text())
    args.output_directory.mkdir(parents=True,exist_ok=True)
    for name,entry in manifest.items():
        directory=args.result_directory/'split_data'/name
        parts=sorted(directory.glob('part[0-9][0-9][0-9][0-9]'))
        if len(parts)!=entry['parts']:
            raise RuntimeError(f'{name}: expected {entry["parts"]} parts, got {len(parts)}')
        digest=hashlib.sha256();target=args.output_directory/name
        with target.open('wb') as out:
            for part in parts:
                data=part.read_bytes();digest.update(data);out.write(data)
        if target.stat().st_size!=entry['size'] or digest.hexdigest()!=entry['sha256']:
            target.unlink()
            raise RuntimeError(f'{name}: verification failed')
        print('verified',target)

if __name__=='__main__':main()
