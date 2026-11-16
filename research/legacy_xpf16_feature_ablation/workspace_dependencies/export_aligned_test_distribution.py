"""Export aligned full test observations, actual samples, and online quantiles.

The original evaluation covers starts 0..760. The source test trajectories
have 868 frames, allowing three further full 20-frame blocks at 780..820.
All arrays are float32 in the original data scale and share window-major order.
"""
import csv
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch

from clean_reliability_common import DATA, ROOT, configure, digest, dump, manifest
from experiment_reliability_transport_e12 import geometry, history, issued, normalization, target
from xpf_crps_metrics import ensemble_crps

sys.path.insert(0, str(ROOT / 'github/DiPred'))
from reliability.quantiles19_predict import Quantiles19Predictor


OUT = ROOT / 'runs/aligned_test_distribution_20261111'
ARCHIVE = ROOT / 'DiPred_完整测试集_真实值_32样本_19分位_20261111.zip'
STARTS = tuple(range(0, 821, 20))


def from_hdf(records, start):
    """Read source data without modifying the dataset or its metadata."""
    frames = []
    for record in records:
        with h5py.File(DATA / record['file'], 'r') as source:
            field = np.stack([source[c][start:start + 20, ::2, ::2] for c in ('u', 'v')]).astype(np.float32)
        assert field.shape == (2, 20, 32, 64)
        frames.append(torch.from_numpy(field))
    return torch.stack(frames).cuda()


def event(kind, start):
    with (OUT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event=kind, history_start=start, time=time.time())) + '\n')


def chunked_copy_to_zip(archive, path):
    hasher = hashlib.sha256()
    with path.open('rb') as source, archive.open(path.name, 'w', force_zip64=True) as target_file:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            target_file.write(chunk)
            hasher.update(chunk)
    return hasher.hexdigest()


@torch.no_grad()
def main():
    configure()
    OUT.mkdir(exist_ok=False)
    if ARCHIVE.exists():
        raise FileExistsError('Archive already exists: ' + str(ARCHIVE))
    records = manifest()['files']['test']
    assert len(records) == 12 and all(r['frames'] == 868 for r in records)
    assert STARTS[-1] + 40 <= min(r['frames'] for r in records)
    assert STARTS[-1] + 60 > min(r['frames'] for r in records)
    count = len(records) * len(STARTS)
    norm = normalization()
    predictor = Quantiles19Predictor(geometry(records), 'cuda')
    assert torch.equal(predictor.norm['mean'], norm['mean']) and torch.equal(predictor.norm['std'], norm['std'])
    shape = (count, 2, 20, 32, 64)
    y_out = np.lib.format.open_memmap(OUT / 'Y.npy', mode='w+', dtype=np.float32, shape=shape)
    samples_out = np.lib.format.open_memmap(OUT / 'diffusion_samples32.npy', mode='w+', dtype=np.float32,
                                            shape=(count, 32, 2, 20, 32, 64))
    q_out = np.lib.format.open_memmap(OUT / 'xpf_quantiles19.npy', mode='w+', dtype=np.float32,
                                     shape=(count, 2, 19, 20, 32, 64))
    index, q_checks, ensemble_checks = [], 0, 0
    q_rows = json.loads((ROOT / 'runs/xpf16_quantiles19_20261111/per_case.json').read_text())
    q_reference = {tuple(r[k] for k in ('start', 'file', 'channel', 'percentile')): r
                   for r in q_rows if r['split'] == 'test' and r['mode'] == 'online'}
    e_rows = json.loads((ROOT / 'runs/native_ensemble8_32_crps_20261111/per_case.json').read_text())
    e_reference = {tuple(r[k] for k in ('start', 'file', 'channel', 'samples', 'estimator')): r
                   for r in e_rows if r['split'] == 'test'}
    max_q_error, max_crps_error = 0., 0.
    for window_index, start in enumerate(STARTS):
        observed = history('test', start) if start <= 760 else from_hdf(records, start)
        prediction = predictor.predict(observed)
        h = (observed - norm['mean']) / norm['std']
        physical = torch.stack([predictor.model.rollout(h, seed, sign=1) * norm['std'] + norm['mean']
                                for seed in range(500000, 500032)])
        assert torch.isfinite(physical).all() and torch.isfinite(prediction['quantiles']).all()
        assert torch.all(prediction['quantiles'][:, :, 1:] >= prediction['quantiles'][:, :, :-1] - 1e-7)
        if start <= 760:
            saved = issued('diffusion', 'test', start)
            assert torch.equal(physical.std(0), saved['std32']) and torch.equal(physical.mean(0), saved['mean'])
        # Both sets of issued predictions exist before reading current truth.
        event('all_quantiles_and32_samples_issued', start)
        truth = target('test', start) if start <= 760 else from_hdf(records, start + 20)
        assert torch.isfinite(truth).all()
        event('complete20_truth_revealed', start)
        q_values = prediction['quantiles']
        begin, end = window_index * len(records), (window_index + 1) * len(records)
        y_out[begin:end] = truth.cpu().numpy()
        samples_out[begin:end] = physical.permute(1, 0, 2, 3, 4, 5).cpu().numpy()
        q_out[begin:end] = q_values.cpu().numpy()
        if start <= 760:
            # Check every previous test-window/trajectory/channel/level result.
            levels = torch.arange(1, 20, device='cuda').reshape(1, 1, 19, 1, 1, 1) / 20
            error = (truth.unsqueeze(2) - q_values) / norm['std'].unsqueeze(2)
            pinball = torch.maximum(levels * error, (levels - 1) * error).mean((3, 4, 5)).cpu()
            hit = (truth.unsqueeze(2) <= q_values).float().mean((3, 4, 5)).cpu()
            paths_norm = (physical - norm['mean']) / norm['std']
            truth_norm = (truth - norm['mean']) / norm['std']
            scores = {n: {k: v.mean((2, 3, 4)).cpu() for k, v in ensemble_crps(paths_norm[:n], truth_norm).items()}
                      for n in (8, 32)}
            for i, record in enumerate(records):
                for c, channel in enumerate(('u', 'v')):
                    for k in range(19):
                        old = q_reference[start, record['file'], channel, (k + 1) * 5]
                        error_p = abs(float(pinball[i, c, k]) - old['pinball'])
                        error_c = abs(float(hit[i, c, k]) - old['empirical_cdf'])
                        assert error_p < 2e-6 and error_c <= 2 / 40960 + 1e-7
                        max_q_error = max(max_q_error, error_p)
                        q_checks += 1
                    for n in (8, 32):
                        for estimator in ('empirical', 'fair'):
                            old = e_reference[start, record['file'], channel, n, estimator]
                            error_e = abs(float(scores[n][estimator][i, c]) - old['crps'])
                            assert error_e < 2e-6
                            max_crps_error = max(max_crps_error, error_e)
                            ensemble_checks += 1
        for i, record in enumerate(records):
            index.append(dict(index=begin + i, trajectory_id=record['file'], history_start=start,
                history_end_exclusive=start + 20, target_start=start + 20, target_end_exclusive=start + 40,
                Re_group=record['group'], Reynolds_number=record['re'], angle_of_attack=record['aoa'],
                prior_39_window_evaluation=start <= 760, cold_start=start == 0))
        predictor.reveal(truth)
        if window_index % 5 == 0 or window_index == len(STARTS) - 1:
            print('EXPORTED', window_index + 1, len(STARTS), 'last_history_start', start, flush=True)
    assert len(index) == count and [r['index'] for r in index] == list(range(count))
    for array in (y_out, samples_out, q_out):
        array.flush()
    del y_out, samples_out, q_out
    with (OUT / 'index.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    dump(OUT / 'metadata.json', dict(sample_count=count, test_trajectories=len(records),
        windows_per_trajectory=len(STARTS), sample_order='window-major: history_start ascending, then trajectory manifest order',
        window_indexing='Zero-based, start inclusive and end exclusive; history_start is first of 20 input frames.',
        first_history_start=0, last_history_start=820, last_target_end_exclusive=860,
        unused_tail_frames='860..867 cannot make a complete additional20-frame target.',
        prior_evaluation='Starts0..760 are historical39-window comparison; starts780,800,820 newly exported and marked false in index.',
        arrays=dict(Y=dict(shape=list(shape), dtype='float32', channels=['u', 'v']),
            diffusion_samples32=dict(shape=[count, 32, 2, 20, 32, 64], dtype='float32', samples='actual stochastic diffusion; first8 match8-noise'),
            xpf_quantiles19=dict(shape=[count, 2, 19, 20, 32, 64], dtype='float32',
                quantiles=[i / 20 for i in range(1, 20)], calibration='existing causal memory_dynamic online calibration; first window cold start')),
        units='Original physical field data scale for all arrays. The dataset SI unit labels have not been independently verified.',
        normalization_for_reference=dict(mean=norm['mean'].flatten().cpu().tolist(), std=norm['std'].flatten().cpu().tolist(),
            formula='normalized=(original-mean)/std; arrays here are original, no inverse transform needed'),
        seeds=list(range(500000, 500032)), feedback='Each complete 20-frame truth is revealed only after both model forecasts are issued; only XPF uses prior revealed errors.',
        historical_data_warning='Existing test trajectories used repeatedly during development; no fresh independent generalization claim.',
        hashes=dict(source_sha256=digest(ROOT / 'export_aligned_test_distribution.py'),
            endpoint_original_sha256=digest(ROOT / 'runs/selected16_quantile_20261110/compact16.pt'),
            interior_checkpoint_sha256=digest(ROOT / 'github/DiPred/reliability/checkpoints/quantiles19.pt'))))
    (OUT / 'README.md').write_text('''# 对齐的完整测试集：真实场、扩散样本、19 分位预测\n\n'''
        '''三个 NPY 数组全部为 float32、原始物理场数据尺度，无需反归一化；数据集的 SI 单位未独立确认。N=504，按窗口起点升序、每个窗口内按 index.csv 轨迹顺序排列。\n\n'''
        '''- `Y.npy`: [504,2,20,32,64]，未来 20 帧真实值。\n'''
        '''- `diffusion_samples32.npy`: [504,32,2,20,32,64]，真实随机扩散预测；前 8 条即 8-noise。\n'''
        '''- `xpf_quantiles19.npy`: [504,2,19,20,32,64]，5%、10%、…、95% 的在线校准分位数。\n'''
        '''- `index.csv`、`metadata.json`: 零基窗口、轨迹、Re、角度、采样种子、尺度及完整顺序。\n\n'''
        '''XPF 在线校准在每个窗口完整 20 帧真实值揭示后更新；扩散随机样本为原生未校准采样。各轨迹第一个窗口为冷启动，两个方法的信息/在线反馈条件因此不同。\n\n'''
        '''0..760 的 39 窗口是既往实验范围，780/800/820 为补充的完整尾部窗口；后续 860..867 的 8 帧不够再构成 20 帧目标。历史轨迹已用于多轮研究。\n\n'''
        '''分位点只覆盖 5%–95%，未规定两端尾部及节点间插值，因此多分位 CRPS 是需要明确积分/尾部假设的近似；这些点也不能唯一确定时空联合分布。PIT 需要先声明分位插值与尾部处理。\n\n'''
        '''加载示例（先从 ZIP 解压至本地）：\n\n```python\nimport numpy as np\ny = np.load('Y.npy', mmap_mode='r')\nsamples = np.load('diffusion_samples32.npy', mmap_mode='r')\nq = np.load('xpf_quantiles19.npy', mmap_mode='r')\n```\n''')
    dump(OUT / 'generation_audit.json', dict(passed=True, aligned_rows=count, full_test_trajectories=len(records),
        complete_windows_per_trajectory=len(STARTS), prior_quantile_records_checked=q_checks,
        prior_ensemble_score_records_checked=ensemble_checks, max_pinball_difference=max_q_error,
        max_standardized_crps_difference=max_crps_error, finite_values_and_quantile_order_checked=True,
        quantiles_and_samples_issued_before_truth=True, full_feedback_only=True))
    files = [OUT / name for name in ('Y.npy', 'diffusion_samples32.npy', 'xpf_quantiles19.npy',
              'index.csv', 'metadata.json', 'README.md', 'events.jsonl', 'generation_audit.json')]
    checksums = {}
    with zipfile.ZipFile(ARCHIVE, mode='x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in files:
            print('PACKING', path.name, path.stat().st_size, flush=True)
            checksums[path.name] = chunked_copy_to_zip(archive, path)
        archive.writestr('SHA256SUMS.txt', ''.join('{}  {}\n'.format(value, name) for name, value in checksums.items()))
    print('VERIFYING_ARCHIVE', ARCHIVE.stat().st_size, flush=True)
    with zipfile.ZipFile(ARCHIVE, 'r', allowZip64=True) as archive:
        assert set(archive.namelist()) == {p.name for p in files} | {'SHA256SUMS.txt'}
        assert archive.testzip() is None
    dump(OUT / 'archive_audit.json', dict(passed=True, archive=str(ARCHIVE), bytes=ARCHIVE.stat().st_size,
        members=len(files) + 1, full_archive_crc_checked=True, sha256_per_member=checksums))
    print('ALIGNED_TEST_DISTRIBUTION_COMPLETE', ARCHIVE, flush=True)


if __name__ == '__main__':
    main()
