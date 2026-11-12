"""Stage the audited IQN adaptation, aligned CDFs, and compact report."""
import json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'runs/xonly_iqn_20261112'


def main():
    assert json.loads((OUT/'audit.json').read_text())['passed']
    s=json.loads((OUT/'summary.json').read_text());training=json.loads((OUT/'training.json').read_text());data=np.load(OUT/'metrics.npz')
    old=np.load(ROOT/'runs/controlled_xp_xpf_20261112/results.npz')
    references=json.loads((ROOT/'runs/controlled_xp_xpf_20261112/summary.json').read_text())
    assert [r['file'] for r in s['records']]==[r['file'] for r in references['records']]
    rows=[('X-only IQN 原始',*s['mean_crps'][0],*s['mean_coverage90'][0]),
          ('X-only IQN 在线',*s['mean_crps'][1],*s['mean_coverage90'][1])]
    for label,idx in (('XP + 固定 CQR',0),('XP + 在线',1),('XPF + 固定 CQR',2),('XPF + 在线',3)):
        q=old['crps'][:,idx].mean((0,1));v=old['intervals'][:,idx,24].mean((0,1))[:,0]
        rows.append((label,*q,*v))
    text=['# X-only IQN 风格 PDE 预测基线','',
      '单模型读取历史 20 帧 X，直接预测完整未来 20 帧的条件分位函数；不调用 Diffusion，也不使用 P 或 F。利用余弦分位编码、条件特征相乘和分位 Huber 损失（kappa=0.1）。原 IQN 研究分布式强化学习，本实验是其方法到 PDE 回归的适配，并非复现原论文的强化学习结果。',
      '', '| 方法 | 标准化 CRPS u ↓ | 标准化 CRPS v ↓ | 90% 覆盖 u | 90% 覆盖 v |',
      '|---|---:|---:|---:|---:|']
    for name,u,v,cu,cv in rows:text.append(f'| {name} | {u:.6f} | {v:.6f} | {cu:.2%} | {cv:.2%} |')
    text.extend(['',f"模型参数 {training['parameters']}；使用同一 819 个训练窗口、seed42、2000 步，输入是真值历史；以最后观测帧作为网络的残差连接中心。单一网络训练时的 tau 从 Uniform(0,1) 采样并截断到 [0.002,0.998]。",
      '', '训练后冻结网络；在线版本逐对分位、逐轨迹、逐通道维护偏置，完整 20 帧揭示后更新。原始版本没有校准。使用相同的 12 条测试轨迹和 41 个非冷启动窗口；每通道 20,152,320 个位置。',
      '', 'CDF 使用 498 个分位点逐位置单调重排后线性插值，最外分位值承载两端概率质量；中位数由相邻分位插值供 CRPS 积分。在线状态只影响当前窗口之前已经揭示的信息。图中旧 XP 使用固定 CQR，XPF 和 Noise 使用在线校准，应结合表格中的匹配对照分析。',
      '', f"原始/在线相邻分位交叉比例（重排之前）为 {s['crossing_fraction'][0]:.3%} / {s['crossing_fraction'][1]:.3%}。共享 tau 在整个场上并不建立可信的多变量联合分布，因此这里报告逐点边际 CDF，而不是全场联合可靠性。",
      '', '训练、校准、测试的 Re 组分离，但测试轨迹此前参与过探索；在线校准点时空相关，不主张严格有限样本保证。模型结构、计算预算与 DiPred 不同，本实验是 X-only 概率基线而不是精确归因。',
      '', '来源：[Dabney et al., Implicit Quantile Networks for Distributional Reinforcement Learning, ICML 2018](https://proceedings.mlr.press/v80/dabney18a.html)。'])
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
    repo=ROOT/'github/DiPred';dest=repo/'results/reliability/xonly_iqn';src=repo/'research/xonly_iqn'
    dest.mkdir(parents=True,exist_ok=True);src.mkdir(parents=True,exist_ok=True)
    for name in ('REPORT.md','summary.json','training.json','audit.json','events.json','model.pt','calibration.pt','metrics.npz','cdf_u.npz','cdf_v.npz','cdf_u.csv','cdf_v.csv','cdf_u.png','cdf_v.png','cdf_u.pdf','cdf_v.pdf'):
        shutil.copy2(OUT/name,dest/name)
    for path in (ROOT/'xonly_iqn').glob('*.py'):shutil.copy2(path,src/path.name)
    (src/'README.md').write_text('结果和训练评估口径见 results/reliability/xonly_iqn/REPORT.md。完整重放需原数据和原研究工作目录。\n')
    print('XONLY_IQN_STAGED',flush=True)

if __name__=='__main__':main()
