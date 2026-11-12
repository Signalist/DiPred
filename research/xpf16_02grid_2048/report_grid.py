"""Generate compact report only after complete study passes independent audit."""
import csv,json,statistics
from pathlib import Path
from clean_reliability_common import digest
from train_grid import OUT


def main():
    audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['passed']
    summary=OUT/'summary'
    q=[r for r in csv.DictReader((summary/'quantiles499.csv').open()) if r['split']=='test']
    iv=[r for r in csv.DictReader((summary/'intervals249.csv').open()) if r['split']=='test']
    dist=json.loads((summary/'distribution_summary.json').read_text())['test']
    def row(method,percent):
        return next(v for v in q if v['method']==method and abs(float(v['percent'])-percent)<1e-7)
    def interval(method,lower):
        return next(v for v in iv if v['method']==method and abs(float(v['lower_pct'])-lower)<1e-7)
    lines=['# XPF16 0.2%–99.8% 及 2048-noise 全分位对照','',
      '固定四步 DiPred，输入 20 帧并一次联合预测之后 20 帧。249 个独立 XPF16 双分位头覆盖对称的上下分位；第 250 个头独立学习 50% 中位数，总计 499 个水平，步长 0.2 个百分点。所有头使用相同 819 个训练样本、初始权重、seed42、2,000 步及优化器协议。49 个此前独立训练的整数百分位头和 0.2% 头在核验完全相同训练协议后复用，其他 200 个头单独训练。', '',
      '测试结果为每条轨迹第 2–42 窗口，Re 工况与 u/v 通道等权；测试轨迹此前多次参与探索，不能作为全新盲测，覆盖为逐点边际覆盖。', '',
      '| 方法 | CRPS | Fair CRPS 统计量 | WIS |', '|---|---:|---:|---:|',
      f"| XPF 在线，重排后 | {dist['xpf_online_rearranged_crps']:.6f}（分位积分近似） | — | {dist['xpf_online_rearranged_wis']:.6f} |",
      f"| 2048 原生样本 | {dist['noise_raw_empirical']:.6f} | {dist['noise_raw_fair']:.6f} | {dist['noise_raw_quantiles_wis']:.6f} |",
      f"| 2048 代理尺度校准样本 | {dist['noise_proxy_empirical']:.6f} | {dist['noise_proxy_fair']:.6f} | {dist['noise_proxy_quantiles_wis']:.6f} |", '',
      f"在线 XPF 重排前，相邻分位交叉比例 {100*dist['xpf_crossing_fraction']:.3f}%，至少一处交叉的位置占 {100*dist['xpf_any_crossing']:.3f}%；重排后按定义不再交叉。", '',
      '| 目标分位 | XPF 在线覆盖 | XPF 重排后覆盖 | 2048 原始覆盖 | 2048 代理尺度覆盖 |',
      '|---:|---:|---:|---:|---:|']
    for level in (.2,1,5,10,25,49.8,50,50.2,75,90,95,99,99.8):
        vals=[row(method,level) for method in ('xpf_online','xpf_online_rearranged','noise_raw','noise_proxy')]
        lines.append(f"| {level:g}% | "+' | '.join(f"{100*float(v['coverage']):.3f}%" for v in vals)+' |')
    lines+=['','| 目标区间 | 方法 | 实际覆盖 | 标准化宽度 | 区间分数 |','|---:|---|---:|---:|---:|']
    for lower in (.2,1,5,10,25,49.8):
        for method,label in (('xpf_online','XPF 在线'),('noise_proxy','2048 代理尺度'),('noise_online_two_tail','2048 双尾在线')):
            r=interval(method,lower)
            lines.append(f"| {100-2*lower:g}% | {label} | {100*float(r['coverage']):.3f}% | {float(r['width']):.5f} | {float(r['interval_score']):.5f} |")
    lines+=['','`quantiles499.csv` 和 `intervals249.csv` 包含全部 Raw/Online、2048 样本与重排版本，逐窗口和逐工况预测帧指标另存 NPZ。每个分位独立预测，联合结果不是一个 499 输出神经网络。', '',
      '所有 XPF 和 2048 双尾区间分别保存独立校准状态，每个完整 20 帧揭示后更新；对应上下尾各自校准。2048 代理尺度路径沿用此前 90% 代理策略，是与 499 个单独双尾区间不同的完整样本分布。原生和代理样本 CRPS 是实际样本经验分布的分数，Fair 统计量使用 K(K−1)，共同样本尺度可能破坏其独立同分布无偏解释。', '',
      'PIT 从单调分位函数逐段反演，在截断尾部形成的概率原子处使用中点 PIT，结果见 `pit_summary.json` 和图表。XPF 近似 CRPS 使用 0.2% 网格的梯形积分与线性分位插值，0%–0.2% 和 99.8%–100% 采用常数分位尾部假设。单调重排后才能将跨头输出解释为有效分位函数；50% 来自独立训练的中位数头。2048 条路径每侧 0.2% 只有约 4 条极端样本，尾部仍受有限样本限制。不存在自动的全场联合覆盖或未见工况覆盖保证。']
    (summary/'REPORT.md').write_text('\n'.join(lines)+'\n')
    src=Path(__file__).resolve().parent
    (summary/'source_hashes.json').write_text(json.dumps({p.name:digest(p) for p in sorted(src.glob('*.py'))},indent=2)+'\n')
    print('GRID_REPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
