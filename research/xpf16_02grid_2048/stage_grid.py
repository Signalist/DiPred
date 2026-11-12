"""Stage completed audited 0.2%-grid study as compact Git artifacts."""
import hashlib,json,shutil,tarfile
from pathlib import Path
import numpy as np
from train_grid import ROOT,OUT,configs

REPO=ROOT/'github/DiPred'
DEST=REPO/'results/reliability/xpf16_02grid_2048'
SRC=REPO/'research/xpf16_02grid_2048'


def split_file(path,target):
    target.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256();size=0;parts=0
    with path.open('rb') as stream:
        while True:
            data=stream.read(600000)
            if not data:break
            (target/f'part{parts:04d}').write_bytes(data)
            h.update(data);size+=len(data);parts+=1
    return dict(filename=path.name,size=size,sha256=h.hexdigest(),parts=parts)


def main():
    assert json.loads((OUT/'independent_audit.json').read_text())['passed']
    assert (OUT/'summary/REPORT.md').exists()
    DEST.mkdir(parents=True,exist_ok=True);SRC.mkdir(parents=True,exist_ok=True)
    for name in ('protocol.json','training_complete.json','evaluation_complete.json','independent_audit.json','portable_audit.json','calibration.pt','latency.json'):
        shutil.copy2(OUT/name,DEST/name)
    for name in ('REPORT.md','source_hashes.json','quantiles499.csv','intervals249.csv','distribution_summary.json',
                 'quantile499.png','quantile499.pdf','interval249.png','interval249.pdf','pit_summary.json','pit.png','pit.pdf',
                 'lead_heatmap.png','lead_heatmap.pdf','re_channel.png','re_channel.pdf',
                 'distribution_diagnostics.png','distribution_diagnostics.pdf','point_cdf.png','point_cdf.pdf'):
        shutil.copy2(OUT/'summary'/name,DEST/name)
    for split in ('development','test'):
        shutil.copy2(OUT/'evaluation'/split/'complete.json',DEST/(split+'_protocol_and_timing.json'))
        source=np.load(OUT/'summary'/f'{split}_per_window.npz')
        iv=source['intervals'];q=source['marginals']
        compact=dict(starts=source['starts'],window_intervals=iv.mean(axis=(3,4)),window_marginals=q.mean(axis=(3,4)),
                     trajectory_intervals=iv[1:].mean(axis=0),trajectory_marginals=q[1:].mean(axis=0))
        for key in source.files:
            if key not in ('starts','intervals','marginals'):
                value=source[key]
                compact['window_'+key]=value.mean(axis=(1,2))
                compact['trajectory_'+key]=value[1:].mean(axis=0)
        np.savez_compressed(OUT/'summary'/f'{split}_compact.npz',**compact)
    test=np.load(OUT/'evaluation/test/0000.npz')
    names=('pixels','truth_snapshot','point_snapshot','sample_snapshot','noise_snapshot','xpf_snapshot','xpf_rearranged_snapshot')
    subset={k:(test[k][:8] if k=='pixels' else test[k][...,:8]) for k in names}
    np.savez_compressed(DEST/'test_snapshot8.npz',**subset)
    archive=OUT/'summary/xpf16_grid_250_weights.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as t:
        for c in configs():t.add(OUT/'models'/c['id']/'model.pt',arcname=c['id']+'.pt',recursive=False)
    artifacts=[archive]
    for split in ('development','test'):
        artifacts += [OUT/'summary'/f'{split}_compact.npz',OUT/'summary'/f'{split}_group_lead.npz']
    parts={p.name:split_file(p,DEST/'split_data'/p.name) for p in artifacts}
    (DEST/'split_data_manifest.json').write_text(json.dumps(parts,indent=2)+'\n')
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,SRC/path.name)
    # Publish the small full-test CDF product and its streaming implementation.
    cdf_source=OUT/'full_test_cdf'
    assert (cdf_source/'config.json').exists(), 'Full-test CDF output is required.'
    shutil.copytree(cdf_source,DEST/'full_test_cdf',dirs_exist_ok=True)
    shutil.copy2(ROOT/'xpf16_cdf_fulltest/README.md',DEST/'full_test_cdf/README.md')
    shutil.copytree(ROOT/'xpf16_cdf_fulltest',SRC/'cdf',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'xpf16_fullpercent_study/restore_result_parts.py',SRC/'restore_result_parts.py')
    (SRC/'README.md').write_text('250 个独立 XPF16 头和 2048 路 Diffusion 的 0.2% 网格研究。重训与完整重放需要原实验工作目录、数据和历史缓存。具体指标、校准口径和历史测试限制见结果 REPORT.md。\n')
    with (DEST/'REPORT.md').open('a') as f:
        f.write('\n权重归档和聚合数组位于 `split_data/`。运行 `python research/xpf16_02grid_2048/restore_result_parts.py results/reliability/xpf16_02grid_2048 --output-directory reconstructed` 可拼接并验证 SHA-256，再将 `xpf16_grid_250_weights.tar.gz` 解压至权重目录。\n')
        f.write('\n仓库推理接口为 `reliability.grid02_predict.Grid02Predictor`；传入权重目录与网格几何后，先 `predict(history)`，获得完整 20 帧观测后调用 `reveal(truth)`。默认计算全部 499 个分位点，可通过 `tail_ids` 选择部分区间降低成本。\n')
        f.write('\n`*_compact.npz` 同时保留每窗口跨轨迹汇总和每轨迹跨窗口汇总；`*_group_lead.npz` 保留 Re/u-v/预测帧分组结果，但不含每轨迹每窗口的完整交叉明细。`test_snapshot8.npz` 是少量对齐位置的实际样本和分位快照，无法重算全测试集。原实验目录保留完整逐轨迹逐窗口结果。\n')
    print('GRID_RELEASE_STAGED',sum(v['parts'] for v in parts.values()),'parts',flush=True)

if __name__=='__main__':main()
