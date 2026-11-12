"""Stage audited XP baseline, curves and reproducible checkpoints."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from evaluate_xp import ROOT,OUT


def split_file(path,dest):
    dest.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256();size=0;parts=0
    with path.open('rb') as stream:
        while True:
            block=stream.read(600000)
            if not block:break
            (dest/f'part{parts:04d}').write_bytes(block)
            h.update(block);size+=len(block);parts+=1
    return dict(filename=path.name,size=size,sha256=h.hexdigest(),parts=parts)


def main():
    assert json.loads((OUT/'results/audit.json').read_text())['passed']
    repo=ROOT/'github/DiPred'
    result=repo/'results/reliability/xp02_cqr'
    source=repo/'research/xp02_cqr'
    result.mkdir(parents=True,exist_ok=True);source.mkdir(parents=True,exist_ok=True)
    for path in (OUT/'results').iterdir():
        if path.is_file():shutil.copy2(path,result/path.name)
    # Preserve all per-window values while avoiding unnecessary JSON whitespace.
    summary=result/'summary.json'
    summary.write_text(json.dumps(json.loads(summary.read_text()),separators=(',',':'))+'\n')
    for name in ('cqr_offsets.pt','training_complete.json'):
        shutil.copy2(OUT/name,result/name)
    for path in (ROOT/'xp02_cqr').glob('*.py'):shutil.copy2(path,source/path.name)
    archive=OUT/'xp02_cqr_weights.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for k in range(1,250):
            tar.add(OUT/'models'/f'XP-P{k:03d}'/'model.pt',arcname=f'XP-P{k:03d}.pt')
    manifest={archive.name:split_file(archive,result/'split_data'/archive.name)}
    (result/'split_data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    text='''# XP-0.2 + split CQR 基线\n\n在冻结四步 DiPred 上，对 0.2%–99.8% 每对对称分位使用独立训练的 XP 网络；输入是完整 20 帧历史场 X 和 20 帧预测场 P，不使用 F 或坐标。X 使用训练集归一化；P 使用原系统的训练集特征标准化并截断至 [-10,10]。249 个风险头独立训练，seed 42，819 个训练样本，各 2000 步。\n\n按 Romano、Patterson、Candès, NeurIPS 2019 的 CQR 方法，在单独校准轨迹上计算非一致性分数 `max(lower - y, y - upper)`，按总失覆盖率 `2 alpha` 使用秩 `ceil((n+1)*(1-2 alpha))` 的校准分位数，将同一修正量向两侧施加。该基线固定校准，图中的 XPF 和 Noise 则使用在线校准；应同时注意模型输入和校准方式均有差异。\n\n每通道将 10 条校准轨迹中 9 个窗口的时空采样点合并，像素与时间高度相关，训练/校准/test Re 组虽然分离，不能自动满足论文的可交换样本假设，因此此处不声称有限样本覆盖保证。若以轨迹为独立单位，10 条不足以在 99.6% 目标下得到有限的标准 split-CQR 修正。\n\n将 249 组 CQR 上下界逐位置投影消除反向区间，再将 498 个分位值重排、线性插值并以最外分位值承载两端质量，从而构成用于比较的 CDF；投影和重排会改变原始逐区间 CQR，见 summary.json 中频率。该 CDF 不是 CQR 原论文单次输出。\n\n`cdf_u.npz` / `cdf_v.npz` 包含共同 x 坐标、先前的六条在线校准曲线、`xp02_cqr` 和观测数；CSV 列相同。平均覆盖全部 12 条测试轨迹、41 个非冷启动窗口，每通道 20,152,320 个网格观测。数据尺度和原图一致，测试集此前反复用于探索。\n\n已保存固定校准系数、独立头权重分片与复现实验源码。可用仓库 `research/xpf16_02grid_2048/restore_result_parts.py` 拼接分片并验证哈希；完整重训/重放需原研究工作目录、数据及历史缓存。\n\n参考：[Romano et al., Conformalized Quantile Regression, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)。\n'''
    (result/'README.md').write_text(text)
    (source/'README.md').write_text(text)
    print('XP_CQR_STAGED',manifest,flush=True)

if __name__=='__main__':main()
