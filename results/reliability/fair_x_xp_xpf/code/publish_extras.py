"""Attach protocol, calibrators and exact small source dependencies to results."""
import ast
import csv
import hashlib
import json
import shutil
import zipfile
import sys
import torch
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'runs/fair_x_xp_xpf_20261113'
DEST=ROOT/'github/DiPred/results/reliability/fair_x_xp_xpf'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol=json.loads((OUT/'frozen_protocol.json').read_text())
    for name,expected in protocol['source_sha256'].items():
        if sha(ROOT/'fair_x_xp_xpf'/name)!=expected:
            raise RuntimeError('Protocol source changed after preregistration: '+name)
    shutil.copy2(OUT/'frozen_protocol.json',DEST/'frozen_protocol.json')
    shutil.copy2(OUT/'train.log',DEST/'training.log')
    with (DEST/'causal_events.jsonl').open('w') as handle:
        for seed in (42,123,2026):
            for arm in ('X','XP','XPF'):
                part=torch.load(OUT/'evaluated'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
                for event in part['events']:
                    handle.write(json.dumps(dict(method=arm,seed=seed,**event))+'\n')
    (DEST/'calibration').mkdir(exist_ok=True)
    for path in (OUT/'calibration').glob('*.pt'):shutil.copy2(path,DEST/'calibration'/path.name)
    # Archive the exact workspace modules imported by the experiment; datasets
    # and the large historical cached input tensors remain external.
    queue=list((ROOT/'fair_x_xp_xpf').glob('*.py'));seen=set()
    for sub in ('xpf16_fullpercent_study','xpf16_02grid_2048','xpf16_cdf_fulltest'):
        queue.extend((ROOT/sub).glob('*.py'))
    while queue:
        path=queue.pop()
        if path in seen:continue
        seen.add(path)
        try:tree=ast.parse(path.read_text())
        except (SyntaxError,UnicodeDecodeError):continue
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names=[v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom) and node.level==0:names=[node.module or '']
            else:continue
            for name in names:
                candidate=ROOT/(name.replace('.','/')+'.py')
                if candidate.is_file() and candidate not in seen:queue.append(candidate)
    sources=[p for p in seen if 'fair_x_xp_xpf' not in p.parts and 'github' not in p.parts]
    with zipfile.ZipFile(DEST/'research_dependencies.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(sources):z.write(path,path.relative_to(ROOT))
        z.write(ROOT/'env.sh','env.sh')
    shared=torch.load(ROOT/'github/DiPred/reliability/checkpoints/current.pt',map_location='cpu',weights_only=False)
    sys.path.insert(0,str(ROOT/'github/DiPred'))
    from reliability.networks import DynamicsDiffusion20
    frozen_model=DynamicsDiffusion20(shared['dynamics_scale'])
    backbone_parameters=sum(p.numel() for p in frozen_model.parameters())
    timings=json.loads((OUT/'benchmark_modes.json').read_text())
    index={(r['method'],str(r['seed']),r['calibration']):r['seconds'] for r in timings}
    with (DEST/'summary.csv').open(newline='') as handle:
        reader=csv.DictReader(handle);names=reader.fieldnames+['head_parameters','backbone_parameters','total_parameters'];rows=list(reader)
    for row in rows:
        row['inference_seconds_per_trajectory']=index[(row['method'],row['seed'],row['calibration'])]
        row['head_parameters']=int(row['parameters'])
        row['backbone_parameters']=0 if row['method']=='X' else backbone_parameters
        row['total_parameters']=row['head_parameters']+row['backbone_parameters']
    with (DEST/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=names);writer.writeheader();writer.writerows(rows)
    config=json.loads((DEST/'config.json').read_text())
    config['inference']='one trajectory, 499 physical-unit quantiles; XP/XPF include four DiPred calls; fixed/online include calibration and quantile sorting; excludes loading and reveal updates'
    shared=torch.load(ROOT/'github/DiPred/reliability/checkpoints/current.pt',map_location='cpu',weights_only=False)
    config['excluded_duplicate_first200']=json.loads((ROOT/'runs/clean_direct20_20261107/manifest.json').read_text())['duplicates']
    config['loss']='quantile Huber kappa=0.1; tau sampled Uniform(0,1), clamped to [0.002,0.998]'
    config['gradient_clip_l2_per_head']=1.0
    config['bootstrap_seeds']={'aggregate':20261113,'by_window':20261114}
    config['bootstrap_repetitions']=10000
    config['calibration_and_development_starts']=list(range(0,161,20))
    config['online_gain']=0.5
    config['calibration_spatial_stride']=4
    config['cdf_axis']={'minimum':-.35,'maximum':.35,'points':701,'unit':'dataset original scale'}
    config['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'
    config['torch_version']=torch.__version__
    config['cuda_version']=torch.version.cuda
    config['cudnn_deterministic']=True
    config['tf32']=False
    config['head_architecture']={'f_projection':'Conv2d 48->160, reshape 8x20', 'encoder':'Conv3d 12->32->32->32 with GroupNorm and SiLU, middle spatial dilation 2', 'decoder':'Conv3d 32->32->14', 'quantile_bases':'six positive-weight monotone bases: linear, logit and four soft hinges'}
    config['normalization_mean']=shared['normalization']['mean'].flatten().tolist()
    config['normalization_std']=shared['normalization']['std'].flatten().tolist()
    config['input_feature_statistics_source']='reliability/checkpoints/current.pt: statistics'
    (DEST/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest=json.loads((DEST/'manifest.json').read_text())
    manifest.update(frozen_backbone_parameters=backbone_parameters,mode_benchmarks=timings,protocol_sha256=sha(DEST/'frozen_protocol.json'),
                    research_dependencies_sha256=sha(DEST/'research_dependencies.zip'),
                    source_sha256={str(p.relative_to(DEST)):sha(p) for p in sorted((DEST/'code').glob('*.py'))},
                    calibration_sha256={p.name:sha(p) for p in sorted((DEST/'calibration').glob('*.pt'))})
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Published extras, dependency source count:',len(sources))

if __name__=='__main__':main()
