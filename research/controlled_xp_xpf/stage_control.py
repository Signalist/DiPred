"""Build the matched comparison report and stage compact reproducible outputs."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'runs/controlled_xp_xpf_20261112'


def split_file(path,target):
    target.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256();size=0;parts=0
    with path.open('rb') as stream:
        while True:
            block=stream.read(600000)
            if not block:break
            (target/f'part{parts:04d}').write_bytes(block)
            h.update(block);size+=len(block);parts+=1
    return dict(filename=path.name,sha256=h.hexdigest(),size=size,parts=parts)


def main():
    assert json.loads((OUT/'audit.json').read_text())['passed']
    summary=json.loads((OUT/'summary.json').read_text());data=np.load(OUT/'results.npz')
    names=list(data['arms']);intervals=data['intervals'];crps=data['crps']
    labels=('XP + 固定 CQR','XP + 在线','XPF + 固定 CQR','XPF + 在线','XP 原始','XPF 原始')
    lines=['# XP 与 XPF：校准方式控制对照','',
      '冻结同一四步 DiPred，输入 20 帧并一次预测未来 20 帧。六组使用相同 12 条测试轨迹、41 个非冷启动窗口、每通道 20,152,320 个位置。',
      '', '| 方法 | CRPS u | CRPS v | 90% 覆盖 u | 90% 覆盖 v | 90% 区间评分 u | 90% 区间评分 v |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for i,label in enumerate(labels):
        z=intervals[:,i,24].mean((0,1));q=crps[:,i].mean((0,1))
        lines.append(f'| {label} | {q[0]:.6f} | {q[1]:.6f} | {z[0,0]:.2%} | {z[1,0]:.2%} | {z[0,2]:.6f} | {z[1,2]:.6f} |')
    trajectory_crps=crps.mean((0,3)) # [method,trajectory], windows and channels equally weighted
    lines.extend(['','轨迹配对 CRPS：'])
    for first,second,description in ((0,2,'固定 CQR：XPF 相对 XP'),(1,3,'在线：XPF 相对 XP'),(0,1,'XP 在线相对固定'),(2,3,'XPF 在线相对固定')):
        delta=trajectory_crps[first]-trajectory_crps[second]
        lines.append(f'- {description}：{int((delta>0).sum())} / 12 条轨迹更低；平均变化 {delta.mean():+.6f}。')
    lines.extend(['',
      '固定 CQR：分别为 XP 与 XPF 在同一 10 条校准轨迹上拟合 Romano 非一致性分数，按目标置信水平使用有限样本秩；在线组使用同一逐尾、逐轨迹状态更新规则，每次只能在完整 20 帧揭示后更新。所有分位按对应概率独立维护状态。原始组完全不校准。',
      '', 'CRPS 以 498 个分位点重排后插值、以相邻分位中点估计 50%，按 0.2% 网格积分；两端使用最外分位的常数尾部。分数和区间宽度按训练通道标准差标准化。CDF 横轴使用数据集原始数值尺度，四组方法共同网格。',
      '', '本对照将校准方式分别匹配，但 XP 与 XPF 风险头在输入、网络结构、初始化和参数量上也不同，不能将全部差异归因于 F。校准点在时间和空间上相关，测试轨迹以前用于探索，因此没有严格有限样本或全场联合覆盖保证。',
      '', '完整逐窗口 × 轨迹 × 通道数据和校准参数已分片保存，可用 `research/xpf16_02grid_2048/restore_result_parts.py` 拼接验证。',
      '', '脚本：`research/controlled_xp_xpf/evaluate_control.py`。重放需要原始数据、冻结权重与原研究工作目录。'])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    repo=ROOT/'github/DiPred';dest=repo/'results/reliability/controlled_xp_xpf';src=repo/'research/controlled_xp_xpf'
    dest.mkdir(parents=True,exist_ok=True);src.mkdir(parents=True,exist_ok=True)
    for name in ('REPORT.md','summary.json','audit.json','events.json','intervals.csv','distribution.csv','cdf_u.png','cdf_v.png','cdf_u.pdf','cdf_v.pdf','calibration.pt'):
        shutil.copy2(OUT/name,dest/name)
    for path in (ROOT/'controlled_xp_xpf').glob('*.py'):shutil.copy2(path,src/path.name)
    parts={'results.npz':split_file(OUT/'results.npz',dest/'split_data/results.npz')}
    (dest/'split_data_manifest.json').write_text(json.dumps(parts,indent=2)+'\n')
    (src/'README.md').write_text('结果与实验协议见 results/reliability/controlled_xp_xpf/REPORT.md。\n')
    print('CONTROL_STAGED',parts,flush=True)

if __name__=='__main__':main()
